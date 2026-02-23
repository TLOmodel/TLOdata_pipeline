#!/usr/bin/env python3
"""
Process census data into demographic outputs:
- totals by sex per district
- age distribution per district
- district x age x sex long table (resource file)

Uses external configuration to locate files/sheets and apply naming/cell patches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tlo.methods.fixes import apply_cell_patches, rename_index_from_file
from tlo.methods.utils import load_cfg, make_calendar_period_lookup


def ensure_dir(p: Path) -> None:
    """
    Ensures that the directory specified by the given Path object exists.
    If it does not exist, it will create the directory along with any necessary
    parent directories. If the directory already exists, the function does nothing.

    Args:
        p (Path): The path of the directory to ensure existence.

    Returns:
        None

    Raises:
        OSError: If the directory cannot be created due to an operating system-related error.
    """
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CensusInputs:
    """
    Represents the inputs required for processing census data.

    This class is a data structure intended for holding and managing data required
    for census analysis. The attributes correspond to relevant file paths, sheet
    names, and other parameters necessary for processing population and census
    details. As it is marked with `frozen=True`, the class instances are immutable,
    ensuring data integrity during operations.

    Attributes:
        workingfile_popsizes: Path to the file containing population sizes.
        census_year: The year for which the census is conducted.
        pop_totals_sheet: Name of the sheet containing population totals.
        age_dist_sheet: Name of the sheet containing age distribution data.
        region_sheet: Name of the sheet containing regional data.
        dist_name_fixes_sheet: Name of the sheet for district name fixes.
        cell_patches_sheet: Name of the sheet for cell patching data.
        national_label: Label denoting national-level data identifiers.
        other_year: Additional year of reference for the dataset, or None if not
                    applicable.
    """

    workingfile_popsizes: str
    census_year: int
    pop_totals_sheet: str
    age_dist_sheet: str
    national_label: str
    other_year: int | None


@dataclass(frozen=True)
class CensusOutputs:
    """
    Represents the output directories for census processing.

    This dataclass encapsulates the directories used for managing
    resources and reports in a census processing workflow. The
    `resources_dir` is intended to store any resource files utilized
    during processing, and `report_dir` serves as the location to
    save output reports or result files.

    Attributes:
        resources_dir: Directory path used to store resource files for census processing.
        report_dir: Directory path used for saving generated reports.
    """

    resources_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class CensusContext:
    """
    Encapsulates context information for census data processing.

    This class is designed to hold configuration settings, input data, and
    output data for census processing workflows. It ensures the necessary
    components for executing a processing pipeline are encapsulated in a
    single frozen object for immutability and easy access.

    Attributes:
        cfg: Configuration mapping containing key-value settings for the
             census process.
        outputs: The outputs object containing all data resultant from the
                 census processing.
        inputs: The inputs object holding all data required for the census
                processing.
    """

    cfg: Mapping[str, Any]
    outputs: CensusOutputs
    inputs: CensusInputs


def _parse_context(cfg: Mapping[str, Any]) -> CensusContext:
    # Outputs
    resources_dir = Path(cfg["outputs"]["resources_dir"])
    ensure_dir(resources_dir)

    report_dir = Path(cfg["outputs"]["reports_dir"]) / str(cfg["country_code"])
    ensure_dir(report_dir)

    census_cfg = cfg["census"]
    census_year = int(census_cfg["year"])

    inputs = CensusInputs(
        workingfile_popsizes=str(census_cfg["population_tables"]),
        census_year=census_year,
        pop_totals_sheet=str(census_cfg.get("pop_totals", "pop_totals")),
        age_dist_sheet=str(census_cfg.get("age_distribution", "age_distribution")),
        national_label=str(census_cfg.get("national_label", cfg.get("country_name", "National"))),
        other_year=(
            int(census_cfg["other_year"]) if census_cfg.get("other_year") is not None else None
        ),
    )

    outputs = CensusOutputs(resources_dir=resources_dir, report_dir=report_dir)
    return CensusContext(cfg=cfg, outputs=outputs, inputs=inputs)


def _load_workbook(ctx: CensusContext) -> dict[str, pd.DataFrame]:
    return pd.read_excel(ctx.inputs.workingfile_popsizes, sheet_name=None)


def _cleanup_a1(a1: pd.DataFrame) -> pd.DataFrame:
    # Keep original cleanup but safer
    a1 = a1.drop([0, 1], errors="ignore")
    if len(a1.index) > 0:
        a1 = a1.drop(a1.index[0], errors="ignore")

    # First column is district/label
    a1.index = a1.iloc[:, 0].astype(str)
    a1 = a1.drop(a1.columns[[0]], axis=1)
    a1 = a1.dropna(how="all", axis=1)
    return a1


def _infer_other_year(ctx: CensusContext, a1: pd.DataFrame) -> int | None:
    other_year = ctx.inputs.other_year
    if other_year is not None:
        return other_year

    # Infer from table width when config not provided
    if a1.shape[1] == 6:
        return ctx.inputs.census_year - 10  # legacy default
    if a1.shape[1] == 3:
        return None

    raise ValueError(
        f"A1 unexpected number of columns after cleanup: {a1.shape[1]}. "
        "Expected 3 (single year) or 6 (two years)."
    )


def _a1_column_titles(census_year: int, other_year: int | None) -> list[str]:
    if other_year is None:
        return [f"Total_{census_year}", f"Male_{census_year}", f"Female_{census_year}"]

    return [
        f"Total_{census_year}",
        f"Male_{census_year}",
        f"Female_{census_year}",
        f"Total_{other_year}",
        f"Male_{other_year}",
        f"Female_{other_year}",
    ]


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = (
        out[cols]
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
    if out[cols].isna().any().any():
        bad = out.loc[out[cols].isna().any(axis=1), cols].head(10)
        raise ValueError(f"A1 contains non-numeric values after coercion. Example rows:\n{bad}")
    return out


def _extract_region_and_national(
    a1: pd.DataFrame,
    region_names: list[str],
    national_label: str,
) -> tuple[pd.DataFrame, pd.Series]:
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

    return region_totals, national_total


def _attach_region_and_filter_district_rows(
    a1: pd.DataFrame,
    region_names: list[str],
    national_label: str,
) -> pd.DataFrame:
    out = a1.copy()
    out["Region"] = None
    out.loc[region_names, "Region"] = region_names
    out["Region"] = out["Region"].ffill()

    # Drop region rows and national row, leaving districts only
    out = out.drop(index=region_names, errors="ignore")
    out = out.drop(index=[national_label], errors="ignore")
    return out


def _build_district_nums(a1_districts: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "Region": a1_districts["Region"],
            "District_Num": np.arange(len(a1_districts["Region"])),
        },
        index=a1_districts.index,
    )


def _load_a1_and_district_nums(
    ctx: CensusContext,
    workbook: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    a1_raw = workbook[ctx.inputs.pop_totals_sheet]
    region_names = workbook["regions"]["Region"].str.strip().tolist()
    dist_names = workbook["dist_name_fixes"]

    a1 = _cleanup_a1(a1_raw)
    other_year = _infer_other_year(ctx, a1)
    titles = _a1_column_titles(ctx.inputs.census_year, other_year)

    if a1.shape[1] != len(titles):
        raise ValueError(
            "A1 unexpected number of columns after cleanup: "
            f"got {a1.shape[1]} expected {len(titles)}. "
            "Check sheet layout / cleanup steps and/or set census.other_year."
        )

    a1.columns = titles
    a1 = a1.dropna(axis=0)
    a1.index = [str(name).strip() for name in list(a1.index)]
    a1 = _coerce_numeric(a1, titles)

    region_totals, national_total = _extract_region_and_national(
        a1=a1,
        region_names=region_names,
        national_label=ctx.inputs.national_label,
    )

    a1_districts = _attach_region_and_filter_district_rows(
        a1=a1,
        region_names=region_names,
        national_label=ctx.inputs.national_label,
    )

    # Validate sums match region/national totals
    assert a1_districts.drop(columns="Region").sum().astype(int).equals(national_total.astype(int))
    assert (
        a1_districts.groupby("Region")
        .sum(numeric_only=True)
        .astype(int)
        .eq(region_totals.astype(int))
        .all()
        .all()
    )

    # Rename districts if mapping provided
    if not dist_names.empty:
        a1_districts = rename_index_from_file(
            a1_districts,
            dist_names,
            canonical_districts=a1_districts.index,  # optional
            strict=False,
        )

    district_nums = _build_district_nums(a1_districts)
    district_names = list(a1_districts.index)
    return a1_districts, district_nums, district_names


def _load_a7_age_distribution(
    ctx: CensusContext,
    workbook: dict[str, pd.DataFrame],
    district_nums: pd.DataFrame,
) -> pd.DataFrame:
    dist_names = workbook["dist_name_fixes"]
    patches_df = workbook["cell_patches"]

    a7 = pd.read_excel(
        ctx.inputs.workingfile_popsizes,
        sheet_name=ctx.inputs.age_dist_sheet,
        header=1,
        index_col=0,
    )
    a7.index = a7.index.str.strip()

    if not patches_df.empty:
        a7 = apply_cell_patches(a7, patches_df, sheet=ctx.inputs.age_dist_sheet, strict=True)

    a7 = a7.dropna().astype(int)

    if not dist_names.empty:
        a7 = rename_index_from_file(
            a7,
            dist_names,
            canonical_districts=district_nums.index,
            strict=False,
        )

    # Keep only districts that are in A1 canonical list
    extract = a7.loc[a7.index.isin(district_nums.index)].copy()
    extract = extract.drop(columns=["Total"])

    assert set(extract.index) == set(district_nums.index)
    return extract


def _age_group_fractions(extract: pd.DataFrame) -> pd.DataFrame:
    frac = extract.div(extract.sum(axis=1), axis=0)
    assert (frac.sum(axis=1).astype("float32") == 1.0).all()
    return frac


def _build_long_age_sex_table(
    ctx: CensusContext,
    a1_districts: pd.DataFrame,
    frac_in_each_age_grp: pd.DataFrame,
    calendar_period_lookup: Mapping[int, str],
) -> pd.DataFrame:
    year = ctx.inputs.census_year

    males = frac_in_each_age_grp.mul(a1_districts[f"Male_{year}"], axis=0)
    assert (
        males.sum(axis=1).astype("float32") == a1_districts[f"Male_{year}"].astype("float32")
    ).all()
    males["district"] = males.index
    males_melt = males.melt(id_vars=["district"], var_name="age_grp", value_name="number")
    males_melt["sex"] = "M"
    males_melt = males_melt.merge(a1_districts[["Region"]], left_on="district", right_index=True)

    females = frac_in_each_age_grp.mul(a1_districts[f"Female_{year}"], axis=0)
    assert (
        females.sum(axis=1).astype("float32") == a1_districts[f"Female_{year}"].astype("float32")
    ).all()
    females["district"] = females.index
    females_melt = females.melt(id_vars=["district"], var_name="age_grp", value_name="number")
    females_melt["sex"] = "F"
    females_melt = females_melt.merge(
        a1_districts[["Region"]], left_on="district", right_index=True
    )

    table = pd.concat([males_melt, females_melt], ignore_index=True)
    table["number"] = table["number"].astype(float)
    table = table.rename(
        columns={
            "district": "District",
            "age_grp": "Age_Grp_Special",
            "sex": "Sex",
            "number": "Count",
        }
    )

    table["Age_Grp_Special"] = table["Age_Grp_Special"].replace({"Less than 1 Year": "0-1"})
    table["Variant"] = f"Census_{year}"
    table["Year"] = year
    table["Period"] = table["Year"].map(calendar_period_lookup)

    return table


def _collapse_special_age_groups(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()

    # Collapse 0-1 and 1-4 into 0-4
    out["Age_Grp"] = out["Age_Grp_Special"].replace({"0-1": "0-4", "1-4": "0-4"})
    out["Count_By_Age_Grp"] = out.groupby(["Age_Grp", "District", "Sex"])["Count"].transform("sum")
    out = out.drop_duplicates(subset=["Age_Grp", "District", "Sex"])
    out = out.rename(columns={"Count_By_Age_Grp": "Count", "Count": "Count_By_Age_Grp_Special"})
    return out


def _merge_district_nums(table: pd.DataFrame, district_nums: pd.DataFrame) -> pd.DataFrame:
    out = table.merge(
        district_nums[["District_Num"]], left_on=["District"], right_index=True, how="left"
    )
    assert 0 == len(
        set(district_nums["District_Num"]).difference(set(pd.unique(out["District_Num"])))
    )
    return out


def _reorder_and_write(ctx: CensusContext, table: pd.DataFrame) -> Path:
    ordered = table[
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

    out_path = (
        ctx.outputs.resources_dir
        / f"ResourceFile_PopulationSize_{ctx.inputs.census_year}Census.csv"
    )
    ordered.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    """
    Main function to process and create a census resource file.

    This function orchestrates the workflow of loading configuration, parsing
    context, extracting data, and processing it to create a comprehensive
    census resource file. It performs consistency checks, applies transformations
    on the data, and writes the final resource file to an output path.
    The function utilizes several helper functions to achieve specific tasks.

    Raises:
        AssertionError: When the length of the extracted data does not match
            the district names count or when the sums of the extracted data
            do not align with the total census data for the specified year.
    """
    cfg = load_cfg()
    ctx = _parse_context(cfg)

    calendar_period_lookup = make_calendar_period_lookup()

    workbook = _load_workbook(ctx)
    a1_districts, district_nums, _district_names = _load_a1_and_district_nums(ctx, workbook)

    extract = _load_a7_age_distribution(ctx, workbook, district_nums)

    # Checks tying A7 to A1 totals (kept from original)
    assert len(extract) == len(_district_names)
    assert extract.sum(axis=1).astype(int).eq(a1_districts[f"Total_{ctx.inputs.census_year}"]).all()

    frac_in_each_age_grp = _age_group_fractions(extract)

    table = _build_long_age_sex_table(
        ctx=ctx,
        a1_districts=a1_districts,
        frac_in_each_age_grp=frac_in_each_age_grp,
        calendar_period_lookup=calendar_period_lookup,
    )
    table = _collapse_special_age_groups(table)
    table = _merge_district_nums(table, district_nums)

    out_path = _reorder_and_write(ctx, table)
    print(f"[OK] Wrote census resource file to: {out_path}")


if __name__ == "__main__":
    main()
