#!/usr/bin/env python3
"""
Census → demography resource builder (ResourceBuilder format).

Produces:
- ResourceFile_PopulationSize_<census_year>Census.csv

Config contract (cfg["census"]):
  year: 2012
  population_tables: "census/tanzania_census.xlsx"   # or "{paths.input_dir}/census/..."
  pop_totals: "pop_totals"
  age_dist: "age_distribution"
  regions: "regions"
  dist_name_fixes: "dist_name_fixes"     # optional sheet
  cell_patches: "cell_patches"           # optional sheet
  national_label: "{country_name}"
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.components.common.fixes import (
    apply_cell_patches,
    rename_index_from_file,
    return_wpp_columns,
)
from pipeline.components.resource_builder import BuildContext, ResourceBuilder
from pipeline.components.utils import make_calendar_period_lookup, resolve_input_path


@dataclass(frozen=True)
class CensusConfig:
    """
    Configuration class for storing census data parameters and paths.

    This class holds configuration details required for processing census data.
    It includes the year of the census, paths to the workbook, sheet names
    for various data sections, and key labels like the national label. The class
    is immutable and utilizes a dataclass design for better readability and maintainability.

    Attributes:
        year: The year of the census data, specified as an integer.
        workbook_path: The file system path to the workbook containing census data.
        total_pop_sheet: Sheet name where total population data resides.
        age_distr_sheet: Sheet name for age distribution data.
        regions_sheet: Sheet name containing region-specific census data.
        dist_name_fixes_sheet: Sheet name for district name correction mappings.
        cell_patches_sheet: Sheet name for specific cell data corrections.
        national_label: The string representing the label for national-level data.
    """

    year: int
    workbook_path: Path

    workbook_sheets: dict
    national_label: str


def _load_census_config(ctx: BuildContext) -> CensusConfig:
    """
    Loads the census configuration details and returns a CensusConfig object.

    This function retrieves census-related configuration data from the given
    build context, parses the necessary parameters, and formats specific fields.
    The returned configuration object consists of all required attributes for
    further processing of census-related data.

    Parameters:
        ctx (BuildContext): The build context containing the configuration data.

    Returns:
        CensusConfig: The loaded census configuration object.
    """
    census_cfg = ctx.cfg["census"]

    year = int(census_cfg["year"])
    workbook_path = resolve_input_path(ctx, census_cfg["population_tables"])

    total_pop_sheet = str(census_cfg["pop_totals"])
    age_distr_sheet = str(census_cfg["age_dist"])
    regions_sheet = str(census_cfg["regions"])

    dist_name_fixes_sheet = str(census_cfg.get("dist_name_fixes", "dist_name_fixes"))
    cell_patches_sheet = str(census_cfg.get("cell_patches", "cell_patches"))

    nat_raw = str(census_cfg.get("national_label", "{country_name}"))
    national_label = nat_raw.format(country_name=ctx.cfg.get("country_name", "National"))

    workbook_sheets = {
        "total_pop_sheet": total_pop_sheet,
        "age_distr_sheet": age_distr_sheet,
        "regions_sheet": regions_sheet,
        "dist_name_fixes_sheet": dist_name_fixes_sheet,
        "cell_patches_sheet": cell_patches_sheet,
    }
    return CensusConfig(
        year=year,
        workbook_path=workbook_path,
        workbook_sheets=workbook_sheets,
        national_label=national_label,
    )


class CensusBuilder(ResourceBuilder):
    """
    Builds census data resources from demographic input tables.

    This class processes raw data from input Excel workbooks to generate a
    long-format demographic table aligned with census year specifications. It manages
    necessary lifecycle steps including preflight checks, loading data, processing, and
    validation of outputs. Users can employ this class to construct census-related outputs
    with correct population and age group distributions.

    """

    COMPONENT = "demography"

    # Census produces exactly one file, but the name depends on year, so keep empty.
    EXPECTED_OUTPUTS: Sequence[str] = ()
    REQUIRED_INPUTS: Sequence[str] = ()

    def __init__(self, ctx: BuildContext, *, dry_run: bool = False) -> None:
        super().__init__(ctx, dry_run=dry_run)
        self.census = _load_census_config(self.ctx)
        # census_cfg = self.ctx.cfg["census"]
        #
        # self.census_year = int(census_cfg["year"])

        # Workbook path (robust against templated/absolute/relative cfg values)
        self.workbook_path = resolve_input_path(self.ctx, self.census.workbook_path)

        # population sheets
        self.workbook_sheets: dict = self.census.workbook_sheets
        # self.age_distr_sheet = str(self.census.age_distr_sheet)
        # self.regions_sheet = str(self.census.regions_sheet)
        #
        # # Optional sheets (cfg-driven, but can be missing in workbook)
        # self.dist_name_fixes_sheet = str(self.census.dist_name_fixes_sheet)
        # self.cell_patches_sheet = str(self.census.cell_patches_sheet)

        # National label (template-friendly)
        self.national_label = self.census.national_label

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def preflight(self) -> None:
        super().preflight()
        if not self.workbook_path.exists():
            raise FileNotFoundError(f"Missing census workbook: {self.workbook_path}")

    def load_data(self) -> Mapping[str, Any]:
        workbook = pd.read_excel(self.workbook_path, sheet_name=None)

        def _sheet_required(name: str) -> pd.DataFrame:
            try:
                return workbook[name]
            except KeyError as e:
                raise KeyError(
                    f"Missing required sheet '{name}' in census workbook: {self.workbook_path}"
                ) from e

        def _sheet_optional(name: str) -> pd.DataFrame:
            # If sheet is absent, treat as "no-op" via empty DataFrame
            return workbook.get(name, pd.DataFrame())

        total_pop_raw = _sheet_required(self.workbook_sheets["total_pop_sheet"])
        age_distr_raw = _sheet_required(self.workbook_sheets["age_distr_sheet"])
        regions_df = _sheet_required(self.workbook_sheets["regions_sheet"])
        dist_name_fixes_df = _sheet_optional(self.workbook_sheets["dist_name_fixes_sheet"])
        cell_patches_df = _sheet_optional(self.workbook_sheets["cell_patches_sheet"])

        return {
            "total_pop_raw": total_pop_raw,
            "age_distr_raw": age_distr_raw,
            "regions_df": regions_df,
            "dist_name_fixes_df": dist_name_fixes_df,
            "cell_patches_df": cell_patches_df,
            "calendar_period_lookup": make_calendar_period_lookup(),
        }

    def build(self, raw: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        """
        Builds a mapping containing processed population data for a census year.

        This method processes raw population and age distribution data, ensuring
        alignment between them, and computes fractions in age groups to build a
        long-format demographic table. The result is a mapping of file names to
        associated processed data.

        Parameters:
            raw (Mapping[str, Any]): A dictionary containing the following:
                - total_pop_raw: Raw population data as a DataFrame.
                - age_distr_raw: Raw age distribution data as a DataFrame.
                - regions_df: Regional data as a DataFrame.
                - dist_name_fixes_df: District name fixes as a DataFrame.
                - cell_patches_df: Cell patches as a DataFrame.
                - calendar_period_lookup: Mapping of calendar periods.

        Returns:
            Mapping[str, pd.DataFrame]: A dictionary mapping file names to processed
            population data.

        Raises:
            AssertionError: If the age distribution districts do not match canonical
            district numbers.
            AssertionError: If age group row sums do not align with total population
            totals within a fixed tolerance.
        """

        calendar_period_lookup: Mapping[int, str] = raw["calendar_period_lookup"]

        total_pop_districts, district_nums = _process_total_pop(
            total_pop_raw=raw["total_pop_raw"],
            regions_df=raw["regions_df"],
            dist_name_fixes_df=raw["dist_name_fixes_df"],
            national_label=self.national_label,
            census_year=self.census.year,
        )

        age_distr = _process_age_distribution(
            age_distr_raw=raw["age_distr_raw"],
            dist_name_fixes_df=raw["dist_name_fixes_df"],
            cell_patches_df=raw["cell_patches_df"],
            canonical_districts=district_nums.index,
        )

        # Coverage check: age distribution must match canonical districts (set-wise)
        if set(age_distr.index) != set(district_nums.index):
            missing = sorted(set(district_nums.index) - set(age_distr.index))
            extra = sorted(set(age_distr.index) - set(district_nums.index))
            raise AssertionError(
                "Age distribution district mismatch.\n"
                f"- missing_from_age_dist: {missing}\n"
                f"- extra_in_age_dist: {extra}"
            )

        # Totals check: age totals per district must match census totals per district
        target = total_pop_districts[f"Total_{self.census.year}"].astype(float)
        got = age_distr.sum(axis=1).astype(float).reindex(target.index)
        if not np.allclose(got.values, target.values, rtol=0, atol=1.0):
            raise AssertionError(
                "Age distribution row sums do not match total population totals (tolerance=1.0).\n"
                f"Top diffs:\n{(got - target).abs().sort_values(ascending=False).head(10)}"
            )

        frac_in_each_age_grp = _compute_age_group_fractions(age_distr)

        table = _build_long_age_sex_table(
            census_year=self.census.year,
            total_pop_districts=total_pop_districts,
            frac_in_each_age_grp=frac_in_each_age_grp,
            calendar_period_lookup=calendar_period_lookup,
        )

        table = _collapse_special_age_groups(table)
        table = _merge_district_nums(table, district_nums)
        table = _reorder_columns(table)

        out_name = f"ResourceFile_PopulationSize_{self.census.year}Census.csv"
        return {out_name: table}

    def validate(self, outputs: Mapping[str, pd.DataFrame]) -> None:
        super().validate(outputs)

        if len(outputs) != 1:
            raise AssertionError(
                f"Expected exactly 1 output for census builder, got {len(outputs)}"
            )

        name, df = next(iter(outputs.items()))

        required_cols = return_wpp_columns()
        missing = sorted(set(required_cols).difference(df.columns))
        if missing:
            raise AssertionError(f"{name}: missing columns: {missing}")

        if df[["District", "Region", "Age_Grp", "Sex", "Period"]].isna().any().any():
            raise AssertionError(f"{name}: nulls found in key columns")

        if not set(df["Sex"].unique()).issubset({"M", "F"}):
            raise AssertionError(f"{name}: unexpected Sex values: {sorted(df['Sex'].unique())}")

        if (pd.to_numeric(df["Count"], errors="coerce").fillna(0) < 0).any():
            raise AssertionError(f"{name}: negative Count values detected")

        if df.duplicated(subset=["District", "Sex", "Age_Grp", "Year"]).any():
            raise AssertionError(f"{name}: duplicate District/Sex/Age_Grp/Year rows found")


# --------------------------------------------------------------------------------------
# Helpers (logic preserved; made more robust + better errors)
# --------------------------------------------------------------------------------------
def _cleanup_total_pop(total_pop_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Template-specific cleanup. Keeps your original behavior (drop first 3 rows-ish),
    but with safer guards.
    """
    df = total_pop_raw.copy()
    df = df.drop([0, 1], errors="ignore")
    if len(df.index) > 0:
        df = df.drop(df.index[0], errors="ignore")

    df.index = df.iloc[:, 0].astype(str)
    df = df.drop(df.columns[[0]], axis=1)
    df = df.dropna(how="all", axis=1)
    return df


def _infer_other_year(census_year: int, total_pop: pd.DataFrame) -> int | None:
    if total_pop.shape[1] == 6:
        return census_year - 10
    if total_pop.shape[1] == 3:
        return None
    raise ValueError(
        f"total_pop unexpected number of columns after cleanup: {total_pop.shape[1]}. "
        "Expected 3 (single year) or 6 (two years)."
    )


def _total_pop_column_titles(census_year: int) -> list[str]:
    return [f"Total_{census_year}", f"Male_{census_year}", f"Female_{census_year}"]


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
        raise ValueError(
            f"total_pop contains non-numeric values after coercion. Example rows:\n{bad}"
        )
    return out


def _extract_region_and_national(
    total_pop: pd.DataFrame,
    region_names: list[str],
    national_label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    try:
        region_totals = total_pop.loc[region_names].copy()
    except KeyError as e:
        raise KeyError(
            f"One or more regions in census.regions not found in total_pop index: {region_names}"
        ) from e

    try:
        national_total = total_pop.loc[national_label].copy()
    except KeyError as e:
        raise KeyError(f"national_label '{national_label}' not found in total_pop index") from e

    return region_totals, national_total


def _attach_region_and_filter_district_rows(
    total_pop: pd.DataFrame,
    region_names: list[str],
    national_label: str,
) -> pd.DataFrame:
    out = total_pop.copy()
    out["Region"] = None
    out.loc[region_names, "Region"] = region_names
    out["Region"] = out["Region"].ffill()

    out = out.drop(index=region_names, errors="ignore")
    out = out.drop(index=[national_label], errors="ignore")
    return out


def _build_district_nums(total_pop_districts: pd.DataFrame) -> pd.DataFrame:
    # 0-based numbering preserved (consistent with your existing resources)
    return pd.DataFrame(
        data={
            "Region": total_pop_districts["Region"],
            "District_Num": np.arange(len(total_pop_districts["Region"])),
        },
        index=total_pop_districts.index,
    )


def _process_total_pop(
    *,
    total_pop_raw: pd.DataFrame,
    regions_df: pd.DataFrame,
    dist_name_fixes_df: pd.DataFrame,
    national_label: str,
    census_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes and validates population data, including cleaning, renaming, and validating
    sums against regional and national totals.

    Arguments:
        total_pop_raw (pd.DataFrame): Raw population data.
        regions_df (pd.DataFrame): DataFrame containing region information, including a
                                   "Region" column.
        dist_name_fixes_df (pd.DataFrame): DataFrame for mapping district names to canonical
                                           variants.
        national_label (str): Label corresponding to the national total row in the population data.
        census_year (int): Year of the census.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame with district-level population data, including regions and cleaned names.
            - A DataFrame containing aggregated district-level populations.

    Raises:
        KeyError: If 'Region' column is missing in the regions DataFrame.
        ValueError: If the number of columns in the population data after cleanup is unexpected.
        AssertionError: If district sums do not match provided national totals or regional totals.
    """
    total_pop = _cleanup_total_pop(total_pop_raw)

    if "Region" not in regions_df.columns:
        raise KeyError("regions sheet must include a 'Region' column")

    region_names = regions_df["Region"].astype(str).str.strip().tolist()

    titles = _total_pop_column_titles(census_year)

    if total_pop.shape[1] != len(titles):
        raise ValueError(
            "total_pop unexpected number of columns after cleanup: "
            f"got {total_pop.shape[1]} expected {len(titles)}. "
            "Check sheet layout / cleanup steps."
        )

    total_pop.columns = titles
    total_pop.dropna(axis=0, inplace=True)
    total_pop.index = [str(name).strip() for name in list(total_pop.index)]
    total_pop = _coerce_numeric(total_pop, titles)

    region_totals, national_total = _extract_region_and_national(
        total_pop=total_pop,
        region_names=region_names,
        national_label=national_label,
    )

    total_pop_districts = _attach_region_and_filter_district_rows(
        total_pop=total_pop,
        region_names=region_names,
        national_label=national_label,
    )

    # Validate sums match national totals
    got_nat = total_pop_districts.drop(columns="Region").sum().astype(int)
    want_nat = national_total.astype(int)
    if not got_nat.equals(want_nat):
        raise AssertionError(
            "District sums do not match national total.\n" f"got:\n{got_nat}\n\nwant:\n{want_nat}"
        )

    # Validate region groupby sums match region totals
    got_reg = total_pop_districts.groupby("Region").sum(numeric_only=True).astype(int)
    want_reg = region_totals.astype(int)
    if not got_reg.eq(want_reg).all().all():
        # show a small diff table
        raise AssertionError(
            f"District sums do not match region totals. "
            f"Example diff:\n{(got_reg - want_reg).head(20)}"
        )

    # Rename districts if mapping provided
    if dist_name_fixes_df is not None and not dist_name_fixes_df.empty:
        total_pop_districts = rename_index_from_file(
            total_pop_districts,
            dist_name_fixes_df,
            canonical_districts=total_pop_districts.index,  # best-effort
            strict=False,
        )

    return total_pop_districts, _build_district_nums(total_pop_districts)


def _process_age_distribution(
    *,
    age_distr_raw: pd.DataFrame,
    dist_name_fixes_df: pd.DataFrame,
    cell_patches_df: pd.DataFrame,
    canonical_districts: Sequence[str],
) -> pd.DataFrame:
    age_distr = age_distr_raw.copy()

    # Expect the first row to contain headers in the template
    age_distr.columns = age_distr.iloc[0]
    age_distr = age_distr.drop(age_distr.index[0])
    age_distr = age_distr.reset_index(drop=True).set_index("Area")

    age_distr.index = age_distr.index.astype(str).str.strip()

    if cell_patches_df is not None and not cell_patches_df.empty:
        age_distr = apply_cell_patches(
            age_distr, cell_patches_df, sheet="age_distribution", strict=True
        )

    age_distr = age_distr.dropna().astype(int)

    if dist_name_fixes_df is not None and not dist_name_fixes_df.empty:
        age_distr = rename_index_from_file(
            age_distr,
            dist_name_fixes_df,
            canonical_districts=canonical_districts,
            strict=False,
        )

    extract = age_distr.loc[age_distr.index.isin(list(canonical_districts))].copy()
    if "Total" in extract.columns:
        extract = extract.drop(columns=["Total"])

    return extract


def _compute_age_group_fractions(age_distr: pd.DataFrame) -> pd.DataFrame:
    denom = age_distr.sum(axis=1).astype(float)
    if (denom <= 0).any():
        bad = denom.loc[denom <= 0].head(10)
        raise AssertionError(f"Age distribution has non-positive totals for some districts:\n{bad}")

    frac = age_distr.div(denom, axis=0)
    s = frac.sum(axis=1).astype(float)

    ok = np.isclose(s.values, 1.0, rtol=0, atol=1e-6)
    if not ok.all():
        bad = s.loc[~pd.Series(ok, index=s.index)].head(10)
        raise AssertionError(f"Age fractions do not sum to 1.0 (tol=1e-6). Examples:\n{bad}")

    return frac


def _build_long_age_sex_table(
    *,
    census_year: int,
    total_pop_districts: pd.DataFrame,
    frac_in_each_age_grp: pd.DataFrame,
    calendar_period_lookup: Mapping[int, str],
) -> pd.DataFrame:
    year = census_year

    males = frac_in_each_age_grp.mul(total_pop_districts[f"Male_{year}"], axis=0)
    male_tot = males.sum(axis=1).astype(float)
    male_target = total_pop_districts[f"Male_{year}"].astype(float)
    if not np.allclose(male_tot.values, male_target.values, rtol=0, atol=1.0):
        diff = (male_tot - male_target).abs().sort_values(ascending=False).head(10)
        raise AssertionError(f"Male age-split totals mismatch (tol=1.0). Top diffs:\n{diff}")

    males["district"] = males.index
    males_melt = males.melt(id_vars=["district"], var_name="age_grp", value_name="number")
    males_melt["sex"] = "M"
    males_melt = males_melt.merge(
        total_pop_districts[["Region"]], left_on="district", right_index=True
    )

    females = frac_in_each_age_grp.mul(total_pop_districts[f"Female_{year}"], axis=0)
    fem_tot = females.sum(axis=1).astype(float)
    fem_target = total_pop_districts[f"Female_{year}"].astype(float)
    if not np.allclose(fem_tot.values, fem_target.values, rtol=0, atol=1.0):
        diff = (fem_tot - fem_target).abs().sort_values(ascending=False).head(10)
        raise AssertionError(f"Female age-split totals mismatch (tol=1.0). Top diffs:\n{diff}")

    females["district"] = females.index
    females_melt = females.melt(id_vars=["district"], var_name="age_grp", value_name="number")
    females_melt["sex"] = "F"
    females_melt = females_melt.merge(
        total_pop_districts[["Region"]], left_on="district", right_index=True
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

    # Ensure every district has a district number
    if out["District_Num"].isna().any():
        missing = sorted(out.loc[out["District_Num"].isna(), "District"].unique().tolist())
        raise AssertionError(f"Missing District_Num for districts: {missing}")

    return out


def _reorder_columns(table: pd.DataFrame) -> pd.DataFrame:
    return table[return_wpp_columns()]
