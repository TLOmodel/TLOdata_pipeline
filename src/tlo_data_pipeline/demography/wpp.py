#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tlo.analysis.utils import make_calendar_period_lookup
from tlo.util import create_age_range_lookup

from utils import load_cfg
from demography_io import (
    ensure_dir,
    reformat_date_period_for_wpp,
    WPPReader,
    melt_year_age_groups,
)

try:
    import yaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from e


def expand_frac_births_male_per_year(
    births_df: pd.DataFrame,
    year_lo: int = 1950,
    year_hi: int = 2100,
) -> pd.DataFrame:
    """
    births_df must contain:
      - Period (inclusive range already, e.g. 2010-2014)
      - Variant (e.g., 'Estimates', 'Medium variant', etc.)
      - M_to_F_Sex_Ratio

    Produces:
      - Year, frac_births_male
    """
    df = births_df.copy()

    # Compute fraction male from sex ratio at birth
    df["frac_births_male"] = df["M_to_F_Sex_Ratio"] / (1.0 + df["M_to_F_Sex_Ratio"])

    # Keep only Estimates + Medium variant (these are the ones that give full year coverage)
    v = df["Variant"].astype(str).str.lower()
    keep = v.str.contains("estimate") | v.str.contains("medium")
    df = df.loc[keep, ["Variant", "Period", "frac_births_male"]].copy()

    # Parse inclusive year bounds from Period
    df[["low_year", "high_year"]] = df["Period"].astype(str).str.split("-", n=1, expand=True)
    df["low_year"] = df["low_year"].astype(int)
    df["high_year"] = df["high_year"].astype(int)

    # Preference: Estimates first for historical coverage, then Medium for projections
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
            raise ValueError(f"WPP frac_births_male lookup failed: year not covered by any Period: {year}")

        # Choose best available variant for that year
        hits = hits.sort_values(["priority", "low_year"], ascending=[True, True])
        frac = float(hits["frac_births_male"].iloc[0])

        records.append({"Year": year, "frac_births_male": frac})

    return pd.DataFrame(records)

def build_wpp_annual_and_init_population(
    *,
    wpp: dict,
    resources_dir: Path,
    country: str,
    header: int,
    census_year: int,
    census_resource_filename: str | None = None,
    init_year: int = 2010,
) -> None:
    """
    Recreates the original pipeline block:
      - Read annual single-year population by age (ESTIMATES + MEDIUM VARIANT) for M/F
      - Output ResourceFile_Pop_Annual_WPP.csv
      - Use init_year (default 2010) national age/sex totals and split by district using census breakdown
      - Output ResourceFile_Population_2010.csv

    Requires in config (wpp dict):
      - pop_annual_male_file
      - pop_annual_female_file
      - pop_annual_sheets (default ['ESTIMATES','MEDIUM VARIANT'])
      - pop_annual_multiplier (default 1000)
      - pop_annual_age_cols_slice_end (default 103) to match original [2:103]
    """
    (__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
    (__tmp2__, age_grp_lookup) = create_age_range_lookup(min_age=0, max_age=100, range_size=5)

    male_file = wpp["pop_annual_male_file"]
    female_file = wpp["pop_annual_female_file"]
    sheets = wpp.get("pop_annual_sheets", ["ESTIMATES", "MEDIUM VARIANT"])
    multiplier = float(wpp.get("pop_annual_multiplier", 1000))
    age_cols_slice_end = int(wpp.get("pop_annual_age_cols_slice_end", 103))  # original used 103 (0..100 + some cols)

    def read_annual(file_path: str, sex: str) -> pd.DataFrame:
        dat = pd.concat(
            [pd.read_excel(file_path, sheet_name=s, header=header) for s in sheets],
            sort=False,
            ignore_index=True,
        )
        out = dat.loc[dat[dat.columns[2]] == country].copy().reset_index(drop=True)
        out["Sex"] = sex
        return out

    ests_males = read_annual(male_file, "M")
    ests_females = read_annual(female_file, "F")

    ests = pd.concat([ests_males, ests_females], sort=False, ignore_index=True)

    # Tidy up exactly like original
    ests = ests.drop(ests.columns[[0, 2, 3, 4, 5, 6]], axis=1)

    # Scale age columns (original: ests.columns[2:103])
    ests[ests.columns[2:age_cols_slice_end]] = ests[ests.columns[2:age_cols_slice_end]] * multiplier

    ests = ests.rename(columns={ests.columns[1]: "Year"})

    # Remove duplicates across concatenated datasets (e.g., 2020 present twice)
    ests = ests.drop_duplicates(subset=["Year", "Sex"], keep="first")

    ests["Variant"] = "WPP_" + ests["Variant"].astype(str)

    # Melt: annual single ages are in column names
    ests_melt = ests.melt(id_vars=["Variant", "Year", "Sex"], value_name="Count", var_name="Age")
    ests_melt["Period"] = ests_melt["Year"].map(calendar_period_lookup)

    # Age_Grp mapping 0..100 -> 0-4,5-9,...
    ests_melt["Age"] = pd.to_numeric(ests_melt["Age"], errors="coerce")
    ests_melt = ests_melt.loc[ests_melt["Age"].notna()].copy()
    ests_melt["Age"] = ests_melt["Age"].astype(int)
    ests_melt["Age_Grp"] = ests_melt["Age"].map(age_grp_lookup)

    ests_melt.to_csv(resources_dir / "ResourceFile_Pop_Annual_WPP.csv", index=False)

    # --- Build init population split by district using census district breakdown
    # Load census output to get district weights
    if census_resource_filename is None:
        census_resource_filename = f"ResourceFile_PopulationSize_{census_year}Census.csv"

    census_path = resources_dir / census_resource_filename
    if not census_path.exists():
        raise FileNotFoundError(
            f"Required census resource file not found: {census_path}\n"
            f"Run build_census_resources.py first, or pass census_resource_filename explicitly."
        )

    census_df = pd.read_csv(census_path)

    # district_nums derived from census file itself
    # (District, District_Num, Region should exist; census file has one row per (District,Age_Grp,Sex))
    district_nums = (
        census_df[["District", "District_Num", "Region"]]
        .drop_duplicates(subset=["District"])
        .set_index("District")
        .sort_values("District_Num")
    )

    # district breakdown weights: sum counts by district / total
    district_breakdown = (
        census_df.groupby("District", as_index=True)["Count"].sum() / float(census_df["Count"].sum())
    )

    # National age/sex totals for init_year from annual WPP
    pop_init = ests_melt.loc[ests_melt["Year"] == init_year, ["Sex", "Age", "Count"]].copy()
    if pop_init.empty:
        available = sorted(pd.unique(ests_melt["Year"]))
        raise ValueError(f"WPP annual population has no rows for Year={init_year}. Available years: {available[:10]}...")

    # Vectorized split by district:
    # create all combos of (Sex,Age) x District and multiply by district share
    base = pop_init.merge(
        district_breakdown.rename("district_share"),
        how="cross"
    )
    base["Count"] = base["Count"] * base["district_share"]
    base = base.rename(columns={"District": "District"})  # no-op for clarity

    # after cross merge, the district key is the index name from district_breakdown; bring it back
    base = base.rename(columns={"index": "District"})
    if "District" not in base.columns:
        # pandas cross merge result keeps right series name as column if it had one; handle both cases
        # district_breakdown.rename("district_share") keeps index; after cross, index becomes column named 'District' in newer pandas
        pass

    # If the District column didn't appear, rebuild via explicit key
    if "District" not in base.columns:
        # Fallback: do explicit cartesian product without how='cross'
        pop_init["_k"] = 1
        db = district_breakdown.reset_index().rename(columns={"Count": "district_share"})
        db["_k"] = 1
        base = pop_init.merge(db, on="_k").drop(columns=["_k"])
        base["Count"] = base["Count"] * base["district_share"]

    # Merge district_nums
    init_pop = base.merge(
        district_nums.reset_index(),
        on="District",
        how="left",
        validate="many_to_one",
    )

    if init_pop[["District_Num", "Region"]].isna().any().any():
        raise AssertionError("District_Num/Region missing after merge — district name mismatch between census and district table.")

    # Reorder
    init_pop = init_pop[["District", "District_Num", "Region", "Sex", "Age", "Count"]].copy()

    # Exact sum check (allow tiny float error)
    total_init = float(init_pop["Count"].sum())
    total_wpp = float(pop_init["Count"].sum())
    if not np.isclose(total_init, total_wpp, rtol=0, atol=1e-6):
        raise AssertionError(f"Init pop sum mismatch: init_pop={total_init} vs WPP={total_wpp}")

    # Match original output ordering:
    # District order = District_Num ascending (A1 order)
    # Sex order = M then F
    # Age order = 0..100
    sex_order = pd.CategoricalDtype(categories=["M", "F"], ordered=True)

    init_pop["Sex"] = init_pop["Sex"].astype(sex_order)
    init_pop["Age"] = init_pop["Age"].astype(int)
    init_pop["District_Num"] = init_pop["District_Num"].astype(int)

    init_pop = init_pop.sort_values(
        by=["District_Num", "Sex", "Age"],
        ascending=[True, True, True],
        kind="mergesort",  # stable sort
    ).reset_index(drop=True)

    init_pop.to_csv(resources_dir / f"ResourceFile_Population_{init_year}.csv", index=False)


def main() -> None:
    cfg = load_cfg()
    wpp = cfg["wpp"]
    resources_dir = Path(cfg["outputs"]["resources_dir"])
    ensure_dir(resources_dir)

    reader = WPPReader(
        country_label=str(wpp["country_label"]),
        header_row=int(wpp.get("header_row", 16)),
        country_col_index=int(wpp.get("country_col_index", 2)),
    )

    # -------------------------
    # 1) Population (5-year age groups)
    # -------------------------
    males = reader.read_country_table(
        wpp["pop_agegrp_male"],
        sheets=wpp["pop_agegrp_sheets"],
        extra_cols={"Sex": "M"},
    )
    females = reader.read_country_table(
        wpp["pop_agegrp_female"],
        sheets=wpp["pop_agegrp_sheets"],
        extra_cols={"Sex": "F"},
    )
    ests = pd.concat([males, females], ignore_index=True)

    # Match your old behavior: age group columns are [2:23] after drops
    ests[ests.columns[2:23]] = ests[ests.columns[2:23]] * float(wpp.get("pop_agegrp_multiplier", 1000))
    ests["Variant"] = "WPP_" + ests["Variant"].astype(str)
    ests = ests.rename(columns={ests.columns[1]: "Year"})

    pop_wpp = ests.melt(id_vars=["Variant", "Year", "Sex"], value_name="Count", var_name="Age_Grp")
    pop_wpp.to_csv(resources_dir / "ResourceFile_Pop_WPP.csv", index=False)

    # -------------------------
    # 2) Births: TotalBirths_WPP + FracBirthsMale + ASFR_WPP
    # -------------------------
    tot_births = reader.read_country_table(
        wpp["total_births_file"],
        sheets=wpp["fert_sheets_all"],
    )
    tot_births = tot_births.melt(id_vars=["Variant"], var_name="Period", value_name="Total_Births").dropna()
    tot_births["Total_Births"] = 1000 * tot_births["Total_Births"]

    sex_ratio = reader.read_country_table(
        wpp["sex_ratio_file"],
        sheets=wpp["fert_sheets_est_med"],
    )
    sex_ratio = sex_ratio.melt(id_vars=["Variant"], var_name="Period", value_name="M_to_F_Sex_Ratio").dropna()

    # Copy Medium to Low/High (keeps your existing output semantics)
    med = sex_ratio.loc[sex_ratio["Variant"] == "Medium variant", ["Period", "M_to_F_Sex_Ratio"]].copy()
    high = med.assign(Variant="High variant")
    low = med.assign(Variant="Low variant")
    sex_ratio = pd.concat([sex_ratio, high, low], ignore_index=True)

    births = tot_births.merge(sex_ratio, on=["Variant", "Period"], validate="1:1")
    reformat_date_period_for_wpp(births, period_col="Period")
    births.to_csv(resources_dir / "ResourceFile_TotalBirths_WPP.csv", index=False)

    frac_birth_male_for_export = expand_frac_births_male_per_year(births, year_lo=1950, year_hi=2100)
    frac_birth_male_for_export.to_csv(resources_dir / "ResourceFile_Pop_Frac_Births_Male.csv", index=False)

    # ASFR
    asfr = reader.read_country_table(
        wpp["asfr_file"],
        sheets=wpp["fert_sheets_all"],
    )
    # columns [2:9] per your script after drops
    asfr[asfr.columns[2:9]] = asfr[asfr.columns[2:9]] / 1000
    reformat_date_period_for_wpp(asfr, period_col="Period")
    asfr["Variant"] = "WPP_" + asfr["Variant"].astype(str)
    asfr_melt = asfr.melt(id_vars=["Variant", "Period"], value_name="asfr", var_name="Age_Grp")
    asfr_melt.to_csv(resources_dir / "ResourceFile_ASFR_WPP.csv", index=False)

    # -------------------------
    # 3) Deaths: TotalDeaths_WPP
    # -------------------------
    deaths_m = reader.read_country_table(
        wpp["deaths_male_file"],
        sheets=wpp["deaths_sheets"],
        extra_cols={"Sex": "M"},
    )
    deaths_f = reader.read_country_table(
        wpp["deaths_female_file"],
        sheets=wpp["deaths_sheets"],
        extra_cols={"Sex": "F"},
    )
    deaths = pd.concat([deaths_m, deaths_f], ignore_index=True)

    deaths[deaths.columns[2:22]] = deaths[deaths.columns[2:22]] * float(wpp.get("deaths_multiplier", 1000))
    reformat_date_period_for_wpp(deaths, period_col="Period")
    deaths_melt = deaths.melt(id_vars=["Variant", "Period", "Sex"], value_name="Count", var_name="Age_Grp")
    deaths_melt["Variant"] = "WPP_" + deaths_melt["Variant"].astype(str)
    deaths_melt.to_csv(resources_dir / "ResourceFile_TotalDeaths_WPP.csv", index=False)

    # -------------------------
    # 4) Life table: Pop_DeathRates_WPP + Expanded
    # -------------------------
    def read_lifetable(file_path: str, sex: str) -> pd.DataFrame:
        df = pd.concat(
            [
                pd.read_excel(
                    file_path,
                    sheet_name=s,
                    header=int(wpp.get("header_row", 16)),
                    usecols=wpp["lifetable_usecols"],
                )
                for s in wpp["lifetable_sheets"]
            ],
            sort=False,
            ignore_index=True,
        )

        # IMPORTANT: as in the original script, country-label is in column index 1 for life tables
        df = df.loc[df[df.columns[1]] == str(wpp["country_label"])].copy().reset_index(drop=True)

        # Drop the country column (same as your script)
        df = df.drop(columns=[df.columns[1]])

        # Add sex
        df["Sex"] = sex
        return df

    lt_m = read_lifetable(wpp["lifetable_male_file"], "M")
    lt_f = read_lifetable(wpp["lifetable_female_file"], "F")
    lt = pd.concat([lt_m, lt_f], ignore_index=True)

    # Normalize variant naming
    lt.loc[lt["Variant"].astype(str).str.contains("Medium", case=False, na=False), "Variant"] = "Medium"

    lt = lt.rename(columns={"Central death rate m(x,n)": "death_rate"})
    lt["Variant"] = "WPP_" + lt["Variant"].astype(str)

    # Drop age 100 row (same as original)
    lt = lt.loc[lt["Age (x)"] != 100.0].copy()

    lt["Age_Grp"] = (
        lt["Age (x)"].astype(int).astype(str)
        + "-"
        + (lt["Age (x)"] + lt["Age interval (n)"] - 1).astype(int).astype(str)
    )

    reformat_date_period_for_wpp(lt, period_col="Period")

    lt_out = lt[["Variant", "Period", "Sex", "Age_Grp", "death_rate"]].copy()
    lt_out.to_csv(resources_dir / "ResourceFile_Pop_DeathRates_WPP.csv", index=False)

    mort_sched = lt_out.copy()
    mort_sched[["low_age", "high_age"]] = mort_sched["Age_Grp"].str.split("-", n=1, expand=True)
    mort_sched["low_age"] = mort_sched["low_age"].astype(int)
    mort_sched["high_age"] = mort_sched["high_age"].astype(int)

    max_age = int(cfg.get("model", {}).get("max_age", 120))
    expanded = []
    for period in pd.unique(mort_sched["Period"]):
        for sex in ["M", "F"]:
            for age_years in range(max_age):
                age_lookup = 99 if age_years > 99 else age_years
                mask = (
                    (mort_sched["Period"] == period)
                    & (mort_sched["Sex"] == sex)
                    & (age_lookup >= mort_sched["low_age"])
                    & (age_lookup <= mort_sched["high_age"])
                )
                if mask.sum() != 1:
                    raise AssertionError(
                        f"LifeTable lookup failed: period={period} sex={sex} age={age_years} matches={mask.sum()}"
                    )
                dr = float(mort_sched.loc[mask, "death_rate"].values[0])
                expanded.append(
                    {
                        "fallbackyear": int(str(period).split("-")[0]),
                        "sex": sex,
                        "age_years": age_years,
                        "death_rate": dr,
                    }
                )

    pd.DataFrame(expanded, columns=["fallbackyear", "sex", "age_years", "death_rate"]).to_csv(
        resources_dir / "ResourceFile_Pop_DeathRates_Expanded_WPP.csv",
        index=False,
    )

    # -------------------------
    # 6) Annual population + initial district population (depends on census output)
    # -------------------------
    if wpp.get("enable_annual_pop", True):
        census_year = int(cfg["census"]["year"]) if "census" in cfg and "year" in cfg["census"] else 2018
        build_wpp_annual_and_init_population(
            wpp=wpp,
            resources_dir=resources_dir,
            country=str(wpp["country_label"]),
            header=int(wpp.get("header_row", 16)),
            census_year=census_year,
            init_year=int(wpp.get("init_population_year", 2010)),
        )

    print(f"[OK] WPP resources written to: {resources_dir}")


if __name__ == "__main__":
    main()
