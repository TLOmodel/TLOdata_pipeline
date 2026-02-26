#!/usr/bin/env python3
"""
WPP demographic processing -> ResourceFiles (ResourceBuilder format)

Produces (always):
  - ResourceFile_Pop_WPP.csv
  - ResourceFile_TotalBirths_WPP.csv
  - ResourceFile_Pop_Frac_Births_Male.csv
  - ResourceFile_ASFR_WPP.csv
  - ResourceFile_TotalDeaths_WPP.csv
  - ResourceFile_Pop_DeathRates_WPP.csv
  - ResourceFile_Pop_DeathRates_Expanded_WPP.csv

Produces (optional, if enable_annual_pop=True):
  - ResourceFile_Pop_Annual_WPP.csv
  - ResourceFile_Population_<init_population_year>.csv
    (requires census resource output to exist in ctx.output_dir)

This module defines a WPPBuilder (no main()).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.components.common.fixes import WPPReader, reformat_date_period_for_wpp
from pipeline.components.resource_builder import BuildContext, ResourceBuilder
from pipeline.components.utils import (
    create_age_range_lookup,
    make_calendar_period_lookup,
    resolve_input_path,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class WPPConfig:
    # Reader controls
    country_label: str
    header_row: int = 16
    country_col_index: int = 2

    # Age-group population (5-year)
    pop_agegrp_male: Path = Path()
    pop_agegrp_female: Path = Path()
    pop_agegrp_sheets: list[str] = None  # type: ignore[assignment]
    pop_agegrp_multiplier: float = 1000.0

    # Fertility
    total_births_file: Path = Path()
    sex_ratio_file: Path = Path()
    asfr_file: Path = Path()
    fert_sheets_all: list[str] = None  # type: ignore[assignment]
    fert_sheets_est_med: list[str] = None  # type: ignore[assignment]

    # Deaths (5-year)
    deaths_male_file: Path = Path()
    deaths_female_file: Path = Path()
    deaths_sheets: list[str] = None  # type: ignore[assignment]
    deaths_multiplier: float = 1000.0

    # Life table
    lifetable_male_file: Path = Path()
    lifetable_female_file: Path = Path()
    lifetable_sheets: list[str] = None  # type: ignore[assignment]
    lifetable_usecols: Any | None = None

    # Annual pop (optional)
    enable_annual_pop: bool = True
    init_population_year: int = 2010
    pop_annual_male_file: Path | None = None
    pop_annual_female_file: Path | None = None
    pop_annual_sheets: list[str] = None  # type: ignore[assignment]
    pop_annual_multiplier: float = 1000.0
    pop_annual_age_cols_slice_end: int = 103

    # Model/census tie-in
    census_year: int = 2018
    max_age: int = 120

    @staticmethod
    def from_ctx(ctx: BuildContext) -> WPPConfig:
        wpp = ctx.cfg["wpp"]

        def _p(key: str) -> Path:
            return resolve_input_path(ctx, wpp[key])

        census_year = int(ctx.cfg.get("census", {}).get("year", 2018))
        max_age = int(ctx.cfg.get("model", {}).get("max_age", 120))

        return WPPConfig(
            country_label=str(wpp["country_label"]),
            header_row=int(wpp.get("header_row", 16)),
            country_col_index=int(wpp.get("country_col_index", 2)),
            pop_agegrp_male=_p("pop_agegrp_male"),
            pop_agegrp_female=_p("pop_agegrp_female"),
            pop_agegrp_sheets=list(wpp["pop_agegrp_sheets"]),
            pop_agegrp_multiplier=float(wpp.get("pop_agegrp_multiplier", 1000)),
            total_births_file=_p("total_births_file"),
            sex_ratio_file=_p("sex_ratio_file"),
            asfr_file=_p("asfr_file"),
            fert_sheets_all=list(wpp["fert_sheets_all"]),
            fert_sheets_est_med=list(wpp["fert_sheets_est_med"]),
            deaths_male_file=_p("deaths_male_file"),
            deaths_female_file=_p("deaths_female_file"),
            deaths_sheets=list(wpp["deaths_sheets"]),
            deaths_multiplier=float(wpp.get("deaths_multiplier", 1000)),
            lifetable_male_file=_p("lifetable_male_file"),
            lifetable_female_file=_p("lifetable_female_file"),
            lifetable_sheets=list(wpp["lifetable_sheets"]),
            lifetable_usecols=wpp.get("lifetable_usecols"),
            enable_annual_pop=bool(wpp.get("enable_annual_pop", True)),
            init_population_year=int(wpp.get("init_population_year", 2010)),
            pop_annual_male_file=(
                _p("pop_annual_male_file") if "pop_annual_male_file" in wpp else None
            ),
            pop_annual_female_file=(
                _p("pop_annual_female_file") if "pop_annual_female_file" in wpp else None
            ),
            pop_annual_sheets=list(wpp.get("pop_annual_sheets", ["ESTIMATES", "MEDIUM VARIANT"])),
            pop_annual_multiplier=float(wpp.get("pop_annual_multiplier", 1000)),
            pop_annual_age_cols_slice_end=int(wpp.get("pop_annual_age_cols_slice_end", 103)),
            census_year=census_year,
            max_age=max_age,
        )


# ---------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------
def expand_frac_births_male_per_year(
    births_df: pd.DataFrame,
    year_lo: int = 1950,
    year_hi: int = 2100,
) -> pd.DataFrame:
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


def _norm_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\u00a0", " ", regex=False).str.strip().str.casefold()


def _filter_country_by_col(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    series = _norm_text_series(df[col])
    lab = str(label).replace("\u00a0", " ").strip().casefold()
    out = df.loc[series == lab].copy().reset_index(drop=True)
    if out.empty:
        sample = series.dropna().unique()[:20]
        raise ValueError(
            "Country filter returned empty.\n"
            f"- label={label!r}\n"
            f"- col={col!r}\n"
            f"- sample_values={list(sample)!r}"
        )
    return out


def _read_lifetable(*, cfg: WPPConfig, file_path: Path, sex: str) -> pd.DataFrame:
    df = pd.concat(
        [
            pd.read_excel(
                file_path, sheet_name=s, header=cfg.header_row, usecols=cfg.lifetable_usecols
            )
            for s in cfg.lifetable_sheets
        ],
        sort=False,
        ignore_index=True,
    )

    # Original WPP lifetable has the location in column index 1
    loc_col = df.columns[1]
    df = _filter_country_by_col(df, loc_col, cfg.country_label)
    df = df.drop(columns=[loc_col])
    df["Sex"] = sex
    return df


def _lifetable_to_death_rates(*, cfg: WPPConfig) -> pd.DataFrame:
    lt_m = _read_lifetable(cfg=cfg, file_path=cfg.lifetable_male_file, sex="M")
    lt_f = _read_lifetable(cfg=cfg, file_path=cfg.lifetable_female_file, sex="F")
    lt = pd.concat([lt_m, lt_f], ignore_index=True)

    lt.loc[lt["Variant"].astype(str).str.contains("medium", case=False, na=False), "Variant"] = (
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


def _expand_death_rates(*, cfg: WPPConfig, lt_out: pd.DataFrame) -> pd.DataFrame:
    mort_sched = lt_out.copy()
    mort_sched[["low_age", "high_age"]] = mort_sched["Age_Grp"].str.split("-", n=1, expand=True)
    mort_sched["low_age"] = mort_sched["low_age"].astype(int)
    mort_sched["high_age"] = mort_sched["high_age"].astype(int)

    expanded: list[dict[str, object]] = []
    for period in pd.unique(mort_sched["Period"]):
        fallbackyear = int(str(period).split("-", maxsplit=1)[0])
        for sex in ["M", "F"]:
            for age_years in range(cfg.max_age):
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


def _read_annual_population(
    *, cfg: WPPConfig, file_path: Path, sex: str, sheets: list[str]
) -> pd.DataFrame:
    dat = pd.concat(
        [pd.read_excel(file_path, sheet_name=s, header=cfg.header_row) for s in sheets],
        sort=False,
        ignore_index=True,
    )
    # WPP annual files typically have location in column index 2
    loc_col = dat.columns[2]
    out = _filter_country_by_col(dat, loc_col, cfg.country_label)
    out["Sex"] = sex
    return out


def _load_census_district_tables(
    *, output_dir: Path, census_year: int
) -> tuple[pd.DataFrame, pd.Series]:
    census_resource_filename = f"ResourceFile_PopulationSize_{census_year}Census.csv"
    census_path = output_dir / census_resource_filename
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
            "District_Num/Region missing after merge — district name mismatch between census and district table."
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


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------
class WPPBuilder(ResourceBuilder):
    COMPONENT = "demography"

    EXPECTED_OUTPUTS = (
        "ResourceFile_Pop_WPP.csv",
        "ResourceFile_TotalBirths_WPP.csv",
        "ResourceFile_Pop_Frac_Births_Male.csv",
        "ResourceFile_ASFR_WPP.csv",
        "ResourceFile_TotalDeaths_WPP.csv",
        "ResourceFile_Pop_DeathRates_WPP.csv",
        "ResourceFile_Pop_DeathRates_Expanded_WPP.csv",
    )

    def preflight(self) -> None:
        super().preflight()
        cfg = WPPConfig.from_ctx(self.ctx)

        required_paths = [
            ("pop_agegrp_male", cfg.pop_agegrp_male),
            ("pop_agegrp_female", cfg.pop_agegrp_female),
            ("total_births_file", cfg.total_births_file),
            ("sex_ratio_file", cfg.sex_ratio_file),
            ("asfr_file", cfg.asfr_file),
            ("deaths_male_file", cfg.deaths_male_file),
            ("deaths_female_file", cfg.deaths_female_file),
            ("lifetable_male_file", cfg.lifetable_male_file),
            ("lifetable_female_file", cfg.lifetable_female_file),
        ]
        missing = [f"{k}: {p}" for k, p in required_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required WPP inputs:\n" + "\n".join(f"- {m}" for m in missing)
            )

        if cfg.enable_annual_pop:
            ann_missing = []
            if cfg.pop_annual_male_file is None or not cfg.pop_annual_male_file.exists():
                ann_missing.append(f"pop_annual_male_file: {cfg.pop_annual_male_file}")
            if cfg.pop_annual_female_file is None or not cfg.pop_annual_female_file.exists():
                ann_missing.append(f"pop_annual_female_file: {cfg.pop_annual_female_file}")
            if ann_missing:
                raise FileNotFoundError(
                    "Annual population enabled, but inputs missing:\n"
                    + "\n".join(f"- {m}" for m in ann_missing)
                )

    def load_data(self) -> Mapping[str, Any]:
        cfg = WPPConfig.from_ctx(self.ctx)
        reader = WPPReader(
            country_label=cfg.country_label,
            header_row=cfg.header_row,
            country_col_index=cfg.country_col_index,
        )

        males = reader.read_country_table(
            str(cfg.pop_agegrp_male), sheets=cfg.pop_agegrp_sheets, extra_cols={"Sex": "M"}
        )
        females = reader.read_country_table(
            str(cfg.pop_agegrp_female), sheets=cfg.pop_agegrp_sheets, extra_cols={"Sex": "F"}
        )
        pop_agegrp = pd.concat([males, females], ignore_index=True)

        tot_births = reader.read_country_table(
            str(cfg.total_births_file), sheets=cfg.fert_sheets_all
        )
        sex_ratio = reader.read_country_table(
            str(cfg.sex_ratio_file), sheets=cfg.fert_sheets_est_med
        )
        asfr = reader.read_country_table(str(cfg.asfr_file), sheets=cfg.fert_sheets_all)

        deaths_m = reader.read_country_table(
            str(cfg.deaths_male_file), sheets=cfg.deaths_sheets, extra_cols={"Sex": "M"}
        )
        deaths_f = reader.read_country_table(
            str(cfg.deaths_female_file), sheets=cfg.deaths_sheets, extra_cols={"Sex": "F"}
        )
        deaths = pd.concat([deaths_m, deaths_f], ignore_index=True)

        annual_raw = None
        if cfg.enable_annual_pop:
            assert cfg.pop_annual_male_file is not None and cfg.pop_annual_female_file is not None
            males_a = _read_annual_population(
                cfg=cfg, file_path=cfg.pop_annual_male_file, sex="M", sheets=cfg.pop_annual_sheets
            )
            females_a = _read_annual_population(
                cfg=cfg, file_path=cfg.pop_annual_female_file, sex="F", sheets=cfg.pop_annual_sheets
            )
            annual_raw = pd.concat([males_a, females_a], ignore_index=True)

        return {
            "cfg": cfg,
            "pop_agegrp": pop_agegrp,
            "tot_births": tot_births,
            "sex_ratio": sex_ratio,
            "asfr": asfr,
            "deaths": deaths,
            "annual_raw": annual_raw,
        }

    def build(self, raw: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        cfg: WPPConfig = raw["cfg"]
        pop_agegrp: pd.DataFrame = raw["pop_agegrp"]
        tot_births: pd.DataFrame = raw["tot_births"]
        sex_ratio: pd.DataFrame = raw["sex_ratio"]
        asfr: pd.DataFrame = raw["asfr"]
        deaths: pd.DataFrame = raw["deaths"]
        annual_raw: pd.DataFrame | None = raw["annual_raw"]

        outputs: dict[str, pd.DataFrame] = {}

        # -------------------------
        # Pop age-group (5-year)
        # -------------------------
        ests = pop_agegrp.copy()
        ests[ests.columns[2:23]] = ests[ests.columns[2:23]] * cfg.pop_agegrp_multiplier
        ests["Variant"] = "WPP_" + ests["Variant"].astype(str)
        ests = ests.rename(columns={ests.columns[1]: "Year"})

        pop_wpp = ests.melt(
            id_vars=["Variant", "Year", "Sex"], value_name="Count", var_name="Age_Grp"
        )
        outputs["ResourceFile_Pop_WPP.csv"] = pop_wpp

        # -------------------------
        # Births + ASFR
        # -------------------------
        tb = tot_births.melt(
            id_vars=["Variant"], var_name="Period", value_name="Total_Births"
        ).dropna()
        tb["Total_Births"] = 1000 * tb["Total_Births"]

        sr = sex_ratio.melt(
            id_vars=["Variant"], var_name="Period", value_name="M_to_F_Sex_Ratio"
        ).dropna()

        med = sr.loc[
            sr["Variant"].astype(str).str.contains("medium", case=False, na=False),
            ["Period", "M_to_F_Sex_Ratio"],
        ].copy()
        if not med.empty:
            sr = pd.concat(
                [sr, med.assign(Variant="High variant"), med.assign(Variant="Low variant")],
                ignore_index=True,
            )

        births = tb.merge(sr, on=["Variant", "Period"], validate="1:1")
        reformat_date_period_for_wpp(births, period_col="Period")
        outputs["ResourceFile_TotalBirths_WPP.csv"] = births

        frac_birth_male = expand_frac_births_male_per_year(births, year_lo=1950, year_hi=2100)
        outputs["ResourceFile_Pop_Frac_Births_Male.csv"] = frac_birth_male

        asfr2 = asfr.copy()
        asfr2[asfr2.columns[2:9]] = asfr2[asfr2.columns[2:9]] / 1000.0
        reformat_date_period_for_wpp(asfr2, period_col="Period")
        asfr2["Variant"] = "WPP_" + asfr2["Variant"].astype(str)
        asfr_melt = asfr2.melt(id_vars=["Variant", "Period"], value_name="asfr", var_name="Age_Grp")
        outputs["ResourceFile_ASFR_WPP.csv"] = asfr_melt

        # -------------------------
        # Deaths (5-year)
        # -------------------------
        d = deaths.copy()
        d[d.columns[2:22]] = d[d.columns[2:22]] * cfg.deaths_multiplier
        reformat_date_period_for_wpp(d, period_col="Period")

        deaths_melt = d.melt(
            id_vars=["Variant", "Period", "Sex"], value_name="Count", var_name="Age_Grp"
        )
        deaths_melt["Variant"] = "WPP_" + deaths_melt["Variant"].astype(str)
        outputs["ResourceFile_TotalDeaths_WPP.csv"] = deaths_melt

        # -------------------------
        # Life table -> death rates + expanded
        # -------------------------
        lt_out = _lifetable_to_death_rates(cfg=cfg)
        outputs["ResourceFile_Pop_DeathRates_WPP.csv"] = lt_out

        expanded = _expand_death_rates(cfg=cfg, lt_out=lt_out)
        outputs["ResourceFile_Pop_DeathRates_Expanded_WPP.csv"] = expanded

        # -------------------------
        # Annual population + init pop (optional)
        # -------------------------
        if cfg.enable_annual_pop and annual_raw is not None:
            calendar_period_lookup = make_calendar_period_lookup()
            age_grp_lookup = create_age_range_lookup(min_age=0, max_age=100, range_size=5)

            ests_a = annual_raw.copy()
            ests_a = ests_a.drop(ests_a.columns[[0, 2, 3, 4, 5, 6]], axis=1)

            end = cfg.pop_annual_age_cols_slice_end
            ests_a[ests_a.columns[2:end]] = (
                ests_a[ests_a.columns[2:end]] * cfg.pop_annual_multiplier
            )

            ests_a = ests_a.rename(columns={ests_a.columns[1]: "Year"})
            ests_a = ests_a.drop_duplicates(subset=["Year", "Sex"], keep="first")
            ests_a["Variant"] = "WPP_" + ests_a["Variant"].astype(str)

            ests_melt = ests_a.melt(
                id_vars=["Variant", "Year", "Sex"], value_name="Count", var_name="Age"
            )
            ests_melt["Period"] = ests_melt["Year"].map(calendar_period_lookup)

            ests_melt["Age"] = pd.to_numeric(ests_melt["Age"], errors="coerce")
            ests_melt = ests_melt.loc[ests_melt["Age"].notna()].copy()
            ests_melt["Age"] = ests_melt["Age"].astype(int)
            ests_melt["Age_Grp"] = ests_melt["Age"].map(age_grp_lookup)

            outputs["ResourceFile_Pop_Annual_WPP.csv"] = ests_melt

            census_df, district_breakdown = _load_census_district_tables(
                output_dir=self.ctx.output_dir,
                census_year=cfg.census_year,
            )
            district_nums = _district_nums_from_census(census_df)

            init_pop = _build_init_population_by_district(
                pop_annual=ests_melt.rename(
                    columns={"Age": "Age"}
                ),  # Age column is already named "Age"
                district_breakdown=district_breakdown,
                district_nums=district_nums,
                init_year=cfg.init_population_year,
            )
            outputs[f"ResourceFile_Population_{cfg.init_population_year}.csv"] = init_pop

        return outputs

    def validate(self, outputs: Mapping[str, pd.DataFrame]) -> None:
        super().validate(outputs)

        # Core outputs must be non-empty
        for name in self.EXPECTED_OUTPUTS:
            if outputs[name].empty:
                raise AssertionError(f"{name} is empty")

        # Pop_WPP must have Count and non-negative
        pop = outputs["ResourceFile_Pop_WPP.csv"]
        if "Count" not in pop.columns:
            raise AssertionError("ResourceFile_Pop_WPP.csv missing 'Count' column")
        if (pd.to_numeric(pop["Count"], errors="coerce").dropna() < 0).any():
            raise AssertionError("ResourceFile_Pop_WPP.csv has negative counts")

        # ASFR within plausible bounds
        asfr = outputs["ResourceFile_ASFR_WPP.csv"]
        vals = pd.to_numeric(asfr["asfr"], errors="coerce").dropna()
        if not vals.empty:
            if (vals < 0).any():
                raise AssertionError("ASFR_WPP has negative values")
            if (vals > 2).any():
                raise AssertionError("ASFR_WPP has implausibly large values (>2 per-woman)")

        # Death rates non-negative
        dr = outputs["ResourceFile_Pop_DeathRates_WPP.csv"]
        dr_vals = pd.to_numeric(dr["death_rate"], errors="coerce").dropna()
        if (dr_vals < 0).any():
            raise AssertionError("DeathRates_WPP has negative death_rate")

        # If init population exists, ensure it sums to WPP init-year totals
        if "ResourceFile_Pop_Annual_WPP.csv" in outputs:
            pop_annual = outputs["ResourceFile_Pop_Annual_WPP.csv"]
            init_keys = [
                k
                for k in outputs
                if k.startswith("ResourceFile_Population_") and k.endswith(".csv")
            ]
            for key in init_keys:
                init_year = int(key.split("_")[-1].replace(".csv", ""))
                init = outputs[key]

                wpp_init = pop_annual.loc[pop_annual["Year"] == init_year, "Count"]
                if wpp_init.empty:
                    raise AssertionError(f"Pop_Annual_WPP has no rows for init year {init_year}")

                if not np.isclose(
                    float(init["Count"].sum()), float(wpp_init.sum()), rtol=0, atol=1e-6
                ):
                    raise AssertionError(f"Init population sum mismatch for {key}")
