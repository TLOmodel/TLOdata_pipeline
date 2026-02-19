#!/usr/bin/env python3
"""
This script processes census data to produce demographic insights such as age-sex
distribution tables, regional totals, and national aggregates. It includes data
reading, cleaning, validation, and transformation.

This script leverages configurations loaded from external files to manage file paths,
process census tables, and interpret calendar periods for demographic data preparation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fixes import (
    apply_cell_patches,
    rename_index_from_file,
)
from tlo.analysis.utils import make_calendar_period_lookup
from utils import load_cfg


def ensure_dir(p: Path) -> None:
    """
    Ensures the existence of a directory at the specified path. If the directory
    does not exist, it will be created. Parent directories will be created as well
    if they do not already exist.

    Args:
        p (Path): The path to the directory to be created.

    """
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Main function responsible for orchestrating the processing of census data and generating
    outputs.

    The function processes input configuration data (`cfg`) to extract file paths, allocate
    resources, parse population data, clean up table structures, manage calendar periods,
    compute demographic aggregates, and construct age-sex distribution tables for districts.

    Raises:
        ValueError: If unexpected table sizes, missing required data, or non-numeric values
                    are encountered.
        KeyError: If required regions or national labels are not found in the data.

    Parameters:
        None

    Returns:
        None
    """
    cfg = load_cfg()

    # Outputs
    resources_dir = Path(cfg["outputs"]["resources_dir"])
    ensure_dir(resources_dir)

    report_dir = Path(cfg["outputs"]["reports_dir"]) / str(cfg["country_code"])
    ensure_dir(report_dir)

    # Census inputs
    census_cfg = cfg["census"]
    workingfile_popsizes = str(census_cfg["population_tables"])
    census_year = int(census_cfg["year"])
    pop_totals = str(census_cfg.get("pop_totals", "pop_totals"))
    age_dist = str(census_cfg.get("age_distribution", "age_distribution"))
    region_sheet = str(census_cfg.get("regions", "regions"))
    dist_name_fixes = str(census_cfg.get("dist_name_fixes", "dist_name_fixes"))
    cell_patches = str(census_cfg.get("cell_patches", "cell_patches"))

    national_label = str(census_cfg.get("national_label", cfg.get("country_name", "National")))

    # Calendar periods
    __temp__, calendar_period_lookup = make_calendar_period_lookup()

    # -----------------------------------------
    # A1: totals by sex for each district
    # -----------------------------------------
    working_file = pd.read_excel(workingfile_popsizes, sheet_name=None)
    a1 = working_file[pop_totals]
    region_names = working_file[region_sheet]["Region"].str.strip().tolist()
    dist_names = working_file[dist_name_fixes]  # point to the CSV above
    patches_df = working_file[cell_patches]

    # Keep your original cleanup (but safer)
    a1 = a1.drop([0, 1], errors="ignore")
    if len(a1.index) > 0:
        a1 = a1.drop(a1.index[0], errors="ignore")

    # First column is district/label
    a1.index = a1.iloc[:, 0].astype(str)
    a1 = a1.drop(a1.columns[[0]], axis=1)
    a1 = a1.dropna(how="all", axis=1)

    # Determine column titles (1-year vs 2-year layout)
    other_year = census_cfg.get("other_year", None)

    if other_year is None:
        if a1.shape[1] == 6:
            other_year = census_year - 10  # legacy default (e.g., 2018/2008)
        elif a1.shape[1] == 3:
            other_year = None
        else:
            raise ValueError(
                f"A1 unexpected number of columns after cleanup: {a1.shape[1]}. "
                f"Expected 3 (single year) or 6 (two years)."
            )

    if other_year is None:
        column_titles = [f"Total_{census_year}", f"Male_{census_year}", f"Female_{census_year}"]
    else:
        column_titles = [
            f"Total_{census_year}",
            f"Male_{census_year}",
            f"Female_{census_year}",
            f"Total_{other_year}",
            f"Male_{other_year}",
            f"Female_{other_year}",
        ]

    if a1.shape[1] != len(column_titles):
        raise ValueError(
            f"A1 unexpected number of columns after cleanup: got "
            f"{a1.shape[1]} expected {len(column_titles)}. "
            "Check sheet layout / cleanup steps and/or set census.other_year."
        )

    a1.columns = column_titles
    a1 = a1.dropna(axis=0)
    a1.index = [str(name).strip() for name in list(a1.index)]

    # Coerce numeric (robust to commas/nbsp)
    a1[column_titles] = (
        a1[column_titles]
        .apply(
            lambda s: (
                s.astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("\u00a0", "", regex=False)
                .str.strip()
            )
        )
        .apply(pd.to_numeric, errors="coerce")
    )
    if a1[column_titles].isna().any().any():
        bad = a1.loc[a1[column_titles].isna().any(axis=1), column_titles].head(10)
        raise ValueError(f"A1 contains non-numeric values after coercion. Example rows:\n{bad}")

    # Canonical districts list should be AFTER you extract district rows (not regions/national)
    # First capture region + national totals from A1 as-is:
    try:
        region_totals = a1.loc[region_names].copy()
    except KeyError as e:
        raise KeyError(
            f"One or more regions in census.regions not found in A1 index: {region_names}"
        ) from e

    try:
        national_total = a1.loc[national_label].copy()
    except KeyError as e:
        raise KeyError(f"national_label '{national_label}' not found in A1 index") from e

    # Organize regional and national totals nicely
    a1 = a1.copy()
    a1["Region"] = None
    a1.loc[region_names, "Region"] = region_names
    a1["Region"] = a1["Region"].ffill()

    # Drop region rows; drop national row
    a1 = a1.drop(index=region_names, errors="ignore")
    a1 = a1.drop(index=[national_label], errors="ignore")

    # Checks (same as your intent)
    assert a1.drop(columns="Region").sum().astype(int).equals(national_total.astype(int))
    assert (
        a1.groupby("Region")
        .sum(numeric_only=True)
        .astype(int)
        .eq(region_totals.astype(int))
        .all()
        .all()
    )

    if not dist_names.empty:
        a1 = rename_index_from_file(
            a1,
            dist_names,
            canonical_districts=a1.index,  # optional
            strict=False,  # keep False to mimic original behavior (no validation)
        )

    canonical_districts = list(a1.index)
    district_names = canonical_districts

    # District numbers follow A1 order
    district_nums = pd.DataFrame(
        data={"Region": a1["Region"], "District_Num": np.arange(len(a1["Region"]))},
        index=a1.index,
    )

    # -----------------------------------------
    # A7: age breakdown for each district
    # -----------------------------------------
    # A7 read
    a7 = pd.read_excel(
        workingfile_popsizes,
        sheet_name=age_dist,
        header=1,
        index_col=0,
    )
    a7.index = a7.index.str.strip()

    if not patches_df.empty:
        a7 = apply_cell_patches(a7, patches_df, sheet=age_dist, strict=True)

    # Old logic: drop NA + numeric
    a7 = a7.dropna()
    a7 = a7.astype(int)

    if not dist_names.empty:
        a7 = rename_index_from_file(
            a7,
            dist_names,
            canonical_districts=district_nums.index,  # optional
            strict=False,  # keep False to mimic original behavior (no validation)
        )

    # Extract districts present in A1 (canonical list)
    extract = a7.loc[a7.index.isin(district_nums.index)].copy()
    extract.drop(columns=["Total"], inplace=True)

    # Checks
    assert set(extract.index) == set(district_nums.index)
    assert len(extract) == len(district_names)
    assert extract.sum(axis=1).astype(int).eq(a1[f"Total_{census_year}"]).all()

    # Fractions by age group
    frac_in_each_age_grp = extract.div(extract.sum(axis=1), axis=0)
    assert (frac_in_each_age_grp.sum(axis=1).astype("float32") == 1.0).all()

    # -----------------------------------------
    # Build district x age x sex breakdown
    # -----------------------------------------
    males = frac_in_each_age_grp.mul(a1[f"Male_{census_year}"], axis=0)
    assert (
        males.sum(axis=1).astype("float32") == a1[f"Male_{census_year}"].astype("float32")
    ).all()
    males["district"] = males.index
    males_melt = males.melt(id_vars=["district"], var_name="age_grp", value_name="number")
    males_melt["sex"] = "M"
    males_melt = males_melt.merge(a1[["Region"]], left_on="district", right_index=True)

    females = frac_in_each_age_grp.mul(a1[f"Female_{census_year}"], axis=0)
    assert (
        females.sum(axis=1).astype("float32") == a1[f"Female_{census_year}"].astype("float32")
    ).all()
    females["district"] = females.index
    females_melt = females.melt(id_vars=["district"], var_name="age_grp", value_name="number")
    females_melt["sex"] = "F"
    females_melt = females_melt.merge(a1[["Region"]], left_on="district", right_index=True)

    # Long-format
    table = pd.concat([males_melt, females_melt], ignore_index=True)
    table["number"] = table["number"].astype(float)
    table.rename(
        columns={
            "district": "District",
            "age_grp": "Age_Grp_Special",
            "sex": "Sex",
            "number": "Count",
        },
        inplace=True,
    )
    table["Age_Grp_Special"] = table["Age_Grp_Special"].replace({"Less than 1 Year": "0-1"})
    table["Variant"] = f"Census_{census_year}"
    table["Year"] = census_year
    table["Period"] = table["Year"].map(calendar_period_lookup)

    # Collapse 0-1 and 1-4 into 0-4
    table["Age_Grp"] = table["Age_Grp_Special"].replace({"0-1": "0-4", "1-4": "0-4"})
    table["Count_By_Age_Grp"] = table.groupby(by=["Age_Grp", "District", "Sex"])["Count"].transform(
        "sum"
    )
    table = table.drop_duplicates(subset=["Age_Grp", "District", "Sex"])
    table = table.rename(columns={"Count_By_Age_Grp": "Count", "Count": "Count_By_Age_Grp_Special"})

    # Merge District_Num
    table = table.merge(
        district_nums[["District_Num"]], left_on=["District"], right_index=True, how="left"
    )
    assert 0 == len(
        set(district_nums["District_Num"]).difference(set(pd.unique(table["District_Num"])))
    )

    # Re-order columns
    table = table[
        [
            "Variant",
            "District",
            "District_Num",
            "Region",
            "Year",
            "Period",
            "Age_Grp",
            "Sex",
            "Count",
        ]
    ]

    # Save resource-file
    out_path = resources_dir / f"ResourceFile_PopulationSize_{census_year}Census.csv"
    table.to_csv(out_path, index=False)

    print(f"[OK] Wrote census resource file to: {out_path}")


if __name__ == "__main__":
    main()
