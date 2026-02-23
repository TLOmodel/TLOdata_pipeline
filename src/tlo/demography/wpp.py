#!/usr/bin/env python3
"""
WPP demographic processing:
- 5-year age-group population
- births (total births, frac male births, ASFR)
- deaths (total deaths)
- life-table death rates + expanded annual age death rates
- optional annual population + initial population split by district from census
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tlo.demography.demography_io import (
    WPPReader,
    ensure_dir,
    reformat_date_period_for_wpp,
)
from tlo.demography.utils import (
    create_age_range_lookup,
    load_cfg,
    make_calendar_period_lookup,
)


@dataclass(frozen=True)
class WPPPaths:
    """
    Representation of paths used in WPP, including resources directory.

    Attributes:
        resources_dir: Path representing the directory where resources for
                       WPP are located.
    """

    resources_dir: Path


@dataclass(frozen=True)
class WPPReaderConfig:
    """
    Configuration settings for the WPPReader.

    This class represents the configuration required for the WPPReader, including
    attributes to identify label names, header rows, and specific index for
    country data.

    Attributes:
        country_label: The label or header name used to identify the country column.
        header_row: The index of the row in the dataset where the header is located.
        country_col_index: The index of the column representing the country data.
    """

    country_label: str
    header_row: int
    country_col_index: int


@dataclass(frozen=True)
class WPPInitConfig:
    """
    Dataclass that contains the initialization configuration for a World Population
    Projection (WPP) model.

    This class encapsulates the essential configuration parameters required to
    initialize a WPP model, including the census year, the initialization population
    year, and the maximum age considered in the model.

    Attributes:
        census_year: int
            The year in which a census was conducted. This serves as the base year
            for population data.
        init_population_year: int
            The year corresponding to the initial population data that will be used
            as a starting point for projections.
        max_age: int
            The maximum age to be considered in the population data and projections.
    """

    census_year: int
    init_population_year: int
    max_age: int


@dataclass(frozen=True)
class WPPContext:
    """
    Represents a context for World Population Prospects (WPP) processing.

    This class encapsulates all the necessary data and configurations needed for
    processing WPP data. It includes mappings for configuration and WPP-specific
    data, directories for resources, details about the data reader, and key
    population-related parameters. Immutability ensures that the context remains
    consistent throughout its usage.

    Attributes:
        cfg: Configuration mapping with string keys and values of any type.
        wpp: Mapping containing WPP-specific data with string keys and values of
            any type.
        resources_dir: Path to the directory where resources are stored.
        reader: Instance of WPPReader used for reading WPP data.
        country_label: String label representing the country being processed.
        header_row: Row number (integer) in the dataset where the header is located.
        max_age: Maximum age (integer) considered in the WPP data.
        init_population_year: Initial year (integer) for population data.
        census_year: Census year (integer) referenced in the context.
    """

    wpp: Mapping[str, Any]
    paths: WPPPaths
    reader: WPPReader
    reader_cfg: WPPReaderConfig
    init: WPPInitConfig


def expand_frac_births_male_per_year(
    births_df: pd.DataFrame,
    year_lo: int = 1950,
    year_hi: int = 2100,
) -> pd.DataFrame:
    """
    Expand fraction of male births to annual series.

    Input births_df must contain: Period, Variant, M_to_F_Sex_Ratio.
    Output: Year, frac_births_male (annual).
    """
    df = births_df.copy()
    df["frac_births_male"] = df["M_to_F_Sex_Ratio"] / (1.0 + df["M_to_F_Sex_Ratio"])

    v = df["Variant"].astype(str).str.lower()
    keep = v.str.contains("estimate") | v.str.contains("medium")
    df = df.loc[keep, ["Variant", "Period", "frac_births_male"]].copy()

    df[["low_year", "high_year"]] = df["Period"].astype(str).str.split("-", n=1, expand=True)
    df["low_year"] = df["low_year"].astype(int)
    df["high_year"] = df["high_year"].astype(int)

    def _variant_priority(name: str) -> int:
        n = name.lower()
        if "estimate" in n:
            return 0
        if "medium" in n:
            return 1
        return 99

    df["priority"] = df["Variant"].astype(str).map(_variant_priority)

    records: list[dict[str, object]] = []
    for year in range(year_lo, year_hi):
        hits = df.loc[(year >= df["low_year"]) & (year <= df["high_year"])].copy()
        if hits.empty:
            raise ValueError(f"WPP frac_births_male lookup failed: year not covered: {year}")

        hits = hits.sort_values(["priority", "low_year"], ascending=[True, True])
        frac = float(hits["frac_births_male"].iloc[0])
        records.append({"Year": year, "frac_births_male": frac})

    return pd.DataFrame(records)


def _make_context(cfg: Mapping[str, Any]) -> WPPContext:
    wpp = cfg["wpp"]

    resources_dir = Path(cfg["outputs"]["resources_dir"])
    ensure_dir(resources_dir)

    reader_cfg = WPPReaderConfig(
        country_label=str(wpp["country_label"]),
        header_row=int(wpp.get("header_row", 16)),
        country_col_index=int(wpp.get("country_col_index", 2)),
    )

    reader = WPPReader(
        country_label=reader_cfg.country_label,
        header_row=reader_cfg.header_row,
        country_col_index=reader_cfg.country_col_index,
    )

    init_cfg = WPPInitConfig(
        census_year=int(cfg.get("census", {}).get("year", 2018)),
        init_population_year=int(wpp.get("init_population_year", 2010)),
        max_age=int(cfg.get("model", {}).get("max_age", 120)),
    )

    return WPPContext(
        wpp=wpp,
        paths=WPPPaths(resources_dir=resources_dir),
        reader=reader,
        reader_cfg=reader_cfg,
        init=init_cfg,
    )


def _write_pop_agegrp(ctx: WPPContext) -> None:
    wpp = ctx.wpp
    r = ctx.reader

    males = r.read_country_table(
        wpp["pop_agegrp_male"],
        sheets=wpp["pop_agegrp_sheets"],
        extra_cols={"Sex": "M"},
    )
    females = r.read_country_table(
        wpp["pop_agegrp_female"],
        sheets=wpp["pop_agegrp_sheets"],
        extra_cols={"Sex": "F"},
    )
    ests = pd.concat([males, females], ignore_index=True)

    ests[ests.columns[2:23]] = ests[ests.columns[2:23]] * float(
        wpp.get("pop_agegrp_multiplier", 1000)
    )
    ests["Variant"] = "WPP_" + ests["Variant"].astype(str)
    ests = ests.rename(columns={ests.columns[1]: "Year"})

    pop_wpp = ests.melt(id_vars=["Variant", "Year", "Sex"], value_name="Count", var_name="Age_Grp")
    pop_wpp.to_csv(ctx.paths.resources_dir / "ResourceFile_Pop_WPP.csv", index=False)


def _build_births_tables(ctx: WPPContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (births_df, asfr_melt)."""
    wpp = ctx.wpp
    r = ctx.reader

    tot_births = r.read_country_table(wpp["total_births_file"], sheets=wpp["fert_sheets_all"])
    tot_births = tot_births.melt(
        id_vars=["Variant"], var_name="Period", value_name="Total_Births"
    ).dropna()
    tot_births["Total_Births"] = 1000 * tot_births["Total_Births"]

    sex_ratio = r.read_country_table(wpp["sex_ratio_file"], sheets=wpp["fert_sheets_est_med"])
    sex_ratio = sex_ratio.melt(
        id_vars=["Variant"], var_name="Period", value_name="M_to_F_Sex_Ratio"
    ).dropna()

    med = sex_ratio.loc[
        sex_ratio["Variant"] == "Medium variant", ["Period", "M_to_F_Sex_Ratio"]
    ].copy()
    sex_ratio = pd.concat(
        [sex_ratio, med.assign(Variant="High variant"), med.assign(Variant="Low variant")],
        ignore_index=True,
    )

    births = tot_births.merge(sex_ratio, on=["Variant", "Period"], validate="1:1")
    reformat_date_period_for_wpp(births, period_col="Period")

    asfr = r.read_country_table(wpp["asfr_file"], sheets=wpp["fert_sheets_all"])
    asfr[asfr.columns[2:9]] = asfr[asfr.columns[2:9]] / 1000
    reformat_date_period_for_wpp(asfr, period_col="Period")
    asfr["Variant"] = "WPP_" + asfr["Variant"].astype(str)
    asfr_melt = asfr.melt(id_vars=["Variant", "Period"], value_name="asfr", var_name="Age_Grp")

    return births, asfr_melt


def _write_births_outputs(ctx: WPPContext) -> None:
    births, asfr_melt = _build_births_tables(ctx)

    births.to_csv(ctx.paths.resources_dir / "ResourceFile_TotalBirths_WPP.csv", index=False)

    frac_birth_male = expand_frac_births_male_per_year(births, year_lo=1950, year_hi=2100)
    frac_birth_male.to_csv(
        ctx.paths.resources_dir / "ResourceFile_Pop_Frac_Births_Male.csv", index=False
    )

    asfr_melt.to_csv(ctx.paths.resources_dir / "ResourceFile_ASFR_WPP.csv", index=False)


def _write_deaths(ctx: WPPContext) -> None:
    wpp = ctx.wpp
    r = ctx.reader

    deaths_m = r.read_country_table(
        wpp["deaths_male_file"],
        sheets=wpp["deaths_sheets"],
        extra_cols={"Sex": "M"},
    )
    deaths_f = r.read_country_table(
        wpp["deaths_female_file"],
        sheets=wpp["deaths_sheets"],
        extra_cols={"Sex": "F"},
    )
    deaths = pd.concat([deaths_m, deaths_f], ignore_index=True)

    deaths[deaths.columns[2:22]] = deaths[deaths.columns[2:22]] * float(
        wpp.get("deaths_multiplier", 1000)
    )
    reformat_date_period_for_wpp(deaths, period_col="Period")

    deaths_melt = deaths.melt(
        id_vars=["Variant", "Period", "Sex"], value_name="Count", var_name="Age_Grp"
    )
    deaths_melt["Variant"] = "WPP_" + deaths_melt["Variant"].astype(str)
    deaths_melt.to_csv(ctx.paths.resources_dir / "ResourceFile_TotalDeaths_WPP.csv", index=False)


def _read_lifetable(ctx: WPPContext, file_path: str, sex: str) -> pd.DataFrame:
    wpp = ctx.wpp

    df = pd.concat(
        [
            pd.read_excel(
                file_path,
                sheet_name=s,
                header=ctx.reader_cfg.header_row,
                usecols=wpp["lifetable_usecols"],
            )
            for s in wpp["lifetable_sheets"]
        ],
        sort=False,
        ignore_index=True,
    )

    # country label for lifetable is in column index 1 (keeps your original behavior)
    df = df.loc[df[df.columns[1]] == ctx.reader_cfg.country_label].copy().reset_index(drop=True)
    df = df.drop(columns=[df.columns[1]])
    df["Sex"] = sex
    return df


def _lifetable_to_death_rates(ctx: WPPContext) -> pd.DataFrame:
    wpp = ctx.wpp

    lt_m = _read_lifetable(ctx, wpp["lifetable_male_file"], "M")
    lt_f = _read_lifetable(ctx, wpp["lifetable_female_file"], "F")
    lt = pd.concat([lt_m, lt_f], ignore_index=True)

    lt.loc[lt["Variant"].astype(str).str.contains("Medium", case=False, na=False), "Variant"] = (
        "Medium"
    )
    lt = lt.rename(columns={"Central death rate m(x,n)": "death_rate"})
    lt["Variant"] = "WPP_" + lt["Variant"].astype(str)

    lt = lt.loc[lt["Age (x)"] != 100.0].copy()
    lt["Age_Grp"] = (
        lt["Age (x)"].astype(int).astype(str)
        + "-"
        + (lt["Age (x)"] + lt["Age interval (n)"] - 1).astype(int).astype(str)
    )

    reformat_date_period_for_wpp(lt, period_col="Period")
    return lt[["Variant", "Period", "Sex", "Age_Grp", "death_rate"]].copy()


def _expand_death_rates(ctx: WPPContext, lt_out: pd.DataFrame) -> pd.DataFrame:
    mort_sched = lt_out.copy()
    mort_sched[["low_age", "high_age"]] = mort_sched["Age_Grp"].str.split("-", n=1, expand=True)
    mort_sched["low_age"] = mort_sched["low_age"].astype(int)
    mort_sched["high_age"] = mort_sched["high_age"].astype(int)

    expanded: list[dict[str, object]] = []
    for period in pd.unique(mort_sched["Period"]):
        fallbackyear = int(str(period).split("-", maxsplit=1)[0])
        for sex in ["M", "F"]:
            for age_years in range(ctx.init.max_age):
                age_lookup = 99 if age_years > 99 else age_years
                mask = (
                    (mort_sched["Period"] == period)
                    & (mort_sched["Sex"] == sex)
                    & (age_lookup >= mort_sched["low_age"])
                    & (age_lookup <= mort_sched["high_age"])
                )
                if mask.sum() != 1:
                    raise AssertionError(
                        "LifeTable lookup failed: "
                        f"period={period} sex={sex} age={age_years} matches={mask.sum()}"
                    )
                dr = float(mort_sched.loc[mask, "death_rate"].values[0])
                expanded.append(
                    {
                        "fallbackyear": fallbackyear,
                        "sex": sex,
                        "age_years": age_years,
                        "death_rate": dr,
                    }
                )

    return pd.DataFrame(expanded, columns=["fallbackyear", "sex", "age_years", "death_rate"])


def _write_lifetable_outputs(ctx: WPPContext) -> None:
    lt_out = _lifetable_to_death_rates(ctx)
    lt_out.to_csv(ctx.paths.resources_dir / "ResourceFile_Pop_DeathRates_WPP.csv", index=False)

    expanded = _expand_death_rates(ctx, lt_out)
    expanded.to_csv(
        ctx.paths.resources_dir / "ResourceFile_Pop_DeathRates_Expanded_WPP.csv", index=False
    )


def _read_annual_population(
    ctx: WPPContext, file_path: str, sex: str, sheets: list[str]
) -> pd.DataFrame:
    dat = pd.concat(
        [pd.read_excel(file_path, sheet_name=s, header=ctx.reader_cfg.header_row) for s in sheets],
        sort=False,
        ignore_index=True,
    )
    out = dat.loc[dat[dat.columns[2]] == ctx.reader_cfg.country_label].copy().reset_index(drop=True)
    out["Sex"] = sex
    # scale happens later once we know slice range
    return out


def _write_annual_population(ctx: WPPContext) -> pd.DataFrame:
    """Write ResourceFile_Pop_Annual_WPP.csv and return the
    long-format dataframe used downstream."""
    wpp = ctx.wpp
    calendar_period_lookup = make_calendar_period_lookup()
    age_grp_lookup = create_age_range_lookup(min_age=0, max_age=100, range_size=5)

    sheets = wpp.get("pop_annual_sheets", ["ESTIMATES", "MEDIUM VARIANT"])
    multiplier = float(wpp.get("pop_annual_multiplier", 1000))
    age_cols_slice_end = int(wpp.get("pop_annual_age_cols_slice_end", 103))

    males = _read_annual_population(ctx, wpp["pop_annual_male_file"], "M", sheets)
    females = _read_annual_population(ctx, wpp["pop_annual_female_file"], "F", sheets)
    ests = pd.concat([males, females], ignore_index=True)

    ests = ests.drop(ests.columns[[0, 2, 3, 4, 5, 6]], axis=1)
    ests[ests.columns[2:age_cols_slice_end]] = ests[ests.columns[2:age_cols_slice_end]] * multiplier
    ests = ests.rename(columns={ests.columns[1]: "Year"})
    ests = ests.drop_duplicates(subset=["Year", "Sex"], keep="first")
    ests["Variant"] = "WPP_" + ests["Variant"].astype(str)

    ests_melt = ests.melt(id_vars=["Variant", "Year", "Sex"], value_name="Count", var_name="Age")
    ests_melt["Period"] = ests_melt["Year"].map(calendar_period_lookup)

    ests_melt["Age"] = pd.to_numeric(ests_melt["Age"], errors="coerce")
    ests_melt = ests_melt.loc[ests_melt["Age"].notna()].copy()
    ests_melt["Age"] = ests_melt["Age"].astype(int)
    ests_melt["Age_Grp"] = ests_melt["Age"].map(age_grp_lookup)

    ests_melt.to_csv(ctx.paths.resources_dir / "ResourceFile_Pop_Annual_WPP.csv", index=False)
    return ests_melt


def _load_census_district_tables(
    resources_dir: Path, census_year: int
) -> tuple[pd.DataFrame, pd.Series]:
    census_resource_filename = f"ResourceFile_PopulationSize_{census_year}Census.csv"
    census_path = resources_dir / census_resource_filename
    if not census_path.exists():
        raise FileNotFoundError(
            f"Required census resource file not found: {census_path}\n"
            "Run build census first, or adjust configuration."
        )

    census_df = pd.read_csv(census_path)
    district_breakdown = census_df.groupby("District", as_index=True)["Count"].sum() / float(
        census_df["Count"].sum()
    )
    return census_df, district_breakdown


def _district_nums_from_census(census_df: pd.DataFrame) -> pd.DataFrame:
    return (
        census_df[["District", "District_Num", "Region"]]
        .drop_duplicates(subset=["District"])
        .set_index("District")
        .sort_values("District_Num")
    )


def _build_init_population_by_district(
    *,
    pop_annual: pd.DataFrame,
    district_breakdown: pd.Series,
    district_nums: pd.DataFrame,
    init_year: int,
) -> pd.DataFrame:
    pop_init = pop_annual.loc[pop_annual["Year"] == init_year, ["Sex", "Age", "Count"]].copy()
    if pop_init.empty:
        available = sorted(pd.unique(pop_annual["Year"]))
        raise ValueError(
            f"WPP annual population has no rows for Year={init_year}. "
            f"Available years: {available[:10]}..."
        )

    # cross join via key (portable across pandas versions)
    pop_init["_k"] = 1
    db = district_breakdown.reset_index().rename(columns={"Count": "district_share"})
    db["_k"] = 1
    base = pop_init.merge(db, on="_k").drop(columns=["_k"])
    base["Count"] = base["Count"] * base["district_share"]

    init_pop = base.merge(
        district_nums.reset_index(),
        on="District",
        how="left",
        validate="many_to_one",
    )

    if init_pop[["District_Num", "Region"]].isna().any().any():
        raise AssertionError(
            "District_Num/Region missing after merge "
            "— district name mismatch between census and district table."
        )

    init_pop = init_pop[["District", "District_Num", "Region", "Sex", "Age", "Count"]].copy()

    total_init = float(init_pop["Count"].sum())
    total_wpp = float(pop_init["Count"].sum())
    if not np.isclose(total_init, total_wpp, rtol=0, atol=1e-6):
        raise AssertionError(f"Init pop sum mismatch: init_pop={total_init} vs WPP={total_wpp}")

    sex_order = pd.CategoricalDtype(categories=["M", "F"], ordered=True)
    init_pop["Sex"] = init_pop["Sex"].astype(sex_order)
    init_pop["Age"] = init_pop["Age"].astype(int)
    init_pop["District_Num"] = init_pop["District_Num"].astype(int)

    init_pop = init_pop.sort_values(
        by=["District_Num", "Sex", "Age"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return init_pop


def _maybe_write_annual_and_init_pop(ctx: WPPContext) -> None:
    if not ctx.wpp.get("enable_annual_pop", True):
        return

    pop_annual = _write_annual_population(ctx)

    census_df, district_breakdown = _load_census_district_tables(
        resources_dir=ctx.paths.resources_dir, census_year=ctx.init.census_year
    )
    district_nums = _district_nums_from_census(census_df)

    init_pop = _build_init_population_by_district(
        pop_annual=pop_annual,
        district_breakdown=district_breakdown,
        district_nums=district_nums,
        init_year=ctx.init.init_population_year,
    )
    init_pop.to_csv(
        ctx.paths.resources_dir / f"ResourceFile_Population_{ctx.init.init_population_year}.csv",
        index=False,
    )


def main() -> None:
    """
    Main function to execute the workflow for generating and writing WPP resources outputs.

    This function orchestrates several helper functions to process configuration,
    create a working context, and write various demographic outputs.

    Raises:
        KeyError: If a required key is missing in the configuration
        FileNotFoundError: If required file paths specified in the configuration are not found
        ValueError: If the configuration contains invalid values
    """
    cfg = load_cfg()
    ctx = _make_context(cfg)

    _write_pop_agegrp(ctx)
    _write_births_outputs(ctx)
    _write_deaths(ctx)
    _write_lifetable_outputs(ctx)
    _maybe_write_annual_and_init_pop(ctx)

    print(f"[OK] WPP resources written to: {ctx.paths.resources_dir}")


if __name__ == "__main__":
    main()
