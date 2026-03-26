#!/usr/bin/env python3
"""
DHS demographic processing -> ResourceFiles (ResourceBuilder format)

Produces (when enabled/configured):
  - ResourceFile_ASFR_DHS.csv
  - ResourceFile_Under_Five_Mortality_DHS.csv

Behavior:
  - If cfg has no "dhs" section: skip (produce no artifacts)
  - If cfg has "dhs" but file missing: fail fast
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.components.framework.builder import BuildContext, ResourceArtifact, ResourceBuilder
from pipeline.components.framework.utils import resolve_input_path


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DHSConfig:
    """
    Represents the configuration for DHS (Demographic and Health Surveys).

    This class encapsulates the configuration required for DHS such as input file path,
    sheet names for age-specific fertility rates (ASFR) and under-five mortality (U5),
    and the header row index for under-five mortality data.

    Attributes:
        file (Path): The path to the DHS file.
        sheet_asfr (str): The name of the sheet containing age-specific fertility rate data.
        sheet_u5 (str): The name of the sheet containing under-five mortality data.
        u5_header (int): The header row index for the under-five mortality data.

    Methods:
        from_ctx(ctx: BuildContext) -> DHSConfig | None:
            Create a DHSConfig instance from a BuildContext.
    """

    file: Path
    sheet_asfr: str
    sheet_u5: str
    u5_header: int = 1

    @staticmethod
    def from_ctx(ctx: BuildContext) -> DHSConfig | None:
        """
        Create a DHSConfig instance from the provided BuildContext.

        This method extracts the "dhs" configuration from the given BuildContext.
        If the configuration for "dhs" is not present, it will return None. If present,
        it will create a DHSConfig instance, resolving input paths and other settings
        for the "file", "sheet_asfr", "sheet_u5", and optionally the "u5_header".

        Raises:
            KeyError: If required keys are missing in the "dhs" configuration.

        Args:
            ctx (BuildContext): The context containing the configuration and utilities
                for resolving the paths and settings for DHSConfig.

        Returns:
            DHSConfig | None: DHSConfig instance if configuration is valid, or None
            if "dhs" configuration is not present.
        """
        if "dhs" not in ctx.cfg:
            return None

        dhs_cfg = ctx.cfg["dhs"]
        return DHSConfig(
            file=resolve_input_path(ctx, dhs_cfg["file"]),
            sheet_asfr=str(dhs_cfg["sheet_asfr"]),
            sheet_u5=str(dhs_cfg["sheet_u5"]),
            u5_header=int(dhs_cfg.get("u5_header", 1)),
        )


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------
class DHSBuilder(ResourceBuilder):
    """
    Represents a builder for Demographic and Health Survey (DHS) resource data.

    This class processes raw data from DHS files to generate standardized datasets
    for demographic indicators like Age-Specific Fertility Rates (ASFR) and
    Under-Five Mortality Rates (U5). It follows a resource-building workflow
    including preflight preparation, data loading, building standardized outputs,
    and final validations.
    """

    COMPONENT = "demography"

    EXPECTED_OUTPUTS = (
        "ResourceFile_ASFR_DHS.csv",
        "ResourceFile_Under_Five_Mortality_DHS.csv",
    )

    def run(self) -> list[ResourceArtifact]:
        """
        Executes the run method for retrieving a list of ResourceArtifact objects. This method first
        checks for an optional DHS configuration. If DHS is not configured, it skips processing and
        returns an empty list. Otherwise, it calls the superclass method to execute
        the main run logic.

        Returns:
            list[ResourceArtifact]: A list of ResourceArtifact objects resulting from the run
            process.
        """
        # Optional builder: skip if not configured
        dhs = DHSConfig.from_ctx(self.ctx)
        if dhs is None:
            print("[SKIP] DHS not configured.")
            return []

        return super().run()

    def preflight(self) -> None:
        """
        Performs preparation tasks prior to the execution of the main process.

        This method ensures certain resources or configurations are properly set
        and validated before proceeding. It uses the current execution context
        to retrieve the necessary configuration and checks for the existence
        of a specific file required for further operations.

        Raises:
            FileNotFoundError: If the required DHS Excel file does not exist.
        """
        super().preflight()

        dhs = DHSConfig.from_ctx(self.ctx)
        assert dhs is not None  # for type checkers; run() guarantees this

        if not dhs.file.exists():
            raise FileNotFoundError(f"DHS Excel file not found: {dhs.file}")

    def load_data(self) -> Mapping[str, Any]:
        """
        Loads and processes data from DHSConfig and related files. This method reads and
        structures data from specified Excel sheets and sections to prepare it for further use.

        Returns:
            Mapping[str, Any]: A dictionary containing processed data. The keys in the
            dictionary are:
                - "dhs": The DHS configuration loaded using the context.
                - "asfr_raw": Raw fertility data extracted from the specified sheet in the DHS
                  file.
                - "u5_raw": Raw under-5 mortality data extracted from the specified sheet
                  with given header information in the DHS file.

        Raises:
            AssertionError: Raises an assertion error if the DHS configuration cannot be
            loaded.
        """
        dhs = DHSConfig.from_ctx(self.ctx)
        assert dhs is not None

        asfr_raw = pd.read_excel(dhs.file, sheet_name=dhs.sheet_asfr)
        u5_raw = read_dhs_u5_table(dhs_file=dhs.file, sheet=dhs.sheet_u5, header=dhs.u5_header)

        return {"dhs": dhs, "asfr_raw": asfr_raw, "u5_raw": u5_raw}

    def build(self, raw: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        """
        Builds standardized data frames from raw input data.

        This method processes raw input data to create standardized data frames.
        Specifically, it converts the ASFR (Age-Specific Fertility Rate) values from
        the per-1000 scale to a per-woman scale and retrieves the already standardized
        data for under-five mortality rates.

        Parameters:
        raw: Mapping[str, Any]
            A dictionary containing raw data as pandas DataFrames. The keys should
            include "asfr_raw" and "u5_raw", corresponding to raw ASFR and under-five
            mortality data respectively.

        Returns:
        Mapping[str, pd.DataFrame]
            A dictionary containing the processed data frames. The keys are file names
            for the resources:
            - "ResourceFile_ASFR_DHS.csv": Processed ASFR data frame with values on a
              per-woman scale.
            - "ResourceFile_Under_Five_Mortality_DHS.csv": Standardized under-five
              mortality rates data frame.
        """
        asfr_raw: pd.DataFrame = raw["asfr_raw"]
        u5_raw: pd.DataFrame = raw["u5_raw"]

        # 1) ASFR: per-1000 -> per-woman
        asfr = asfr_raw.copy()
        value_cols = list(asfr.columns[1:])
        asfr[value_cols] = asfr[value_cols].apply(pd.to_numeric, errors="coerce") / 1000.0

        # 2) U5: already standardized by reader
        u5 = u5_raw.copy()

        return {
            "ResourceFile_ASFR_DHS.csv": asfr,
            "ResourceFile_Under_Five_Mortality_DHS.csv": u5,
        }

    def validate(self, outputs: Mapping[str, pd.DataFrame]) -> None:
        """
        Validates the structure and plausibility of provided outputs.

        This method performs a series of checks to ensure that the outputs conform to expected
        structures and ranges for specific indicators. The method validates two specific outputs:
        'ASFR' (Age-Specific Fertility Rates) and 'U5' (Under-Five Mortality).

        Raises:
            AssertionError: If the ASFR output contains fewer than 2 columns, has negative values
            or has values greater than 2 after scaling.
            Also raised if the U5 output columns do not match the expected structure or if the U5
            output contains negative values or values exceeding 1.0.

        Parameters:
            outputs (Mapping[str, pd.DataFrame]): A dictionary mapping output names (strings)
            to corresponding Pandas DataFrame objects.

        Returns:
            None: This method performs validation and does not return any value.
        """
        super().validate(outputs)

        # ASFR plausibility
        asfr = outputs["ResourceFile_ASFR_DHS.csv"]
        if asfr.shape[1] < 2:
            raise AssertionError("ASFR must have at least 2 columns (label + values).")

        asfr_vals = pd.to_numeric(asfr.iloc[:, 1:].stack(), errors="coerce").dropna()
        if not asfr_vals.empty:
            if (asfr_vals < 0).any():
                raise AssertionError("ASFR contains negative values after scaling.")
            if (asfr_vals > 2).any():
                raise AssertionError("ASFR contains values > 2 per-woman. Check units/scaling.")

        # U5 checks
        u5 = outputs["ResourceFile_Under_Five_Mortality_DHS.csv"]
        if list(u5.columns) != ["Year", "Est", "Lo", "Hi"]:
            raise AssertionError(
                f"U5 columns must be ['Year','Est','Lo','Hi'], got {list(u5.columns)}"
            )

        for c in ["Est", "Lo", "Hi"]:
            vals = pd.to_numeric(u5[c], errors="coerce").dropna()
            if not vals.empty:
                if (vals < 0).any():
                    raise AssertionError(f"U5 {c} contains negative values.")
                if (vals > 1.0).any():
                    raise AssertionError(f"U5 {c} contains values > 1.0; likely still per-1000.")


# ---------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------
def _coerce_year_series(s: pd.Series) -> pd.Series:
    """
    Converts a pandas Series containing year information into a standardized and validated
    format by coercing the values, adjusting scaling for single-digit years, rounding,
    and filtering valid year ranges.

    Parameters:
    s (pd.Series): Input series to process, expected to contain numerical or textual
                   representations of year values.

    Returns:
    pd.Series: Series of years coerced into a standardized integer format with non-valid
               values masked as NaN, and restricted to the range 1900–2100.

    Raises:
    None
    """
    x = pd.to_numeric(s, errors="coerce")

    if x.notna().any():
        x_non = x.dropna()
        if x_non.max() < 10 and x_non.min() > 0:
            x = x * 1000

    x = x.round()
    x = x.where((x >= 1900) & (x <= 2100))
    return x.astype("Int64")


def _maybe_scale_probs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Scales the probabilistic values in specific columns of a DataFrame if necessary.

    This function takes a DataFrame and a list of column names. It evaluates the
    numerical values within the specified columns and determines whether the values
    exceed a certain magnitude. If the median value of the numerical entries across
    the provided columns is greater than 1.0, all values within these columns are
    scaled down by dividing them by 1000.0. If no numerical values are present, the
    original DataFrame is returned unchanged.

    Parameters:
    df: pd.DataFrame
        The input DataFrame containing the data to be processed.
    cols: list[str]
        List of column names in the DataFrame that need to be evaluated
        and possibly modified.

    Returns:
    pd.DataFrame
        A copy of the original DataFrame with the specified columns potentially
        scaled down if the median of their values exceeds 1.0.
    """
    out = df.copy()
    vals = pd.to_numeric(out[cols].stack(), errors="coerce").dropna()
    if vals.empty:
        return out

    if vals.median() > 1.0:
        out[cols] = out[cols] / 1000.0
    return out


def read_dhs_u5_table(dhs_file: str | Path, sheet: str, header: int = 1) -> pd.DataFrame:
    """
    Reads a DHS U5 table from an Excel file and processes it into a standardized DataFrame.

    This function reads an Excel file using a given sheet and header row, identifies
    columns relevant to the DHS U5 data (e.g., Year, Estimate/Lower/Upper values),
    cleans and standardizes the data, and returns the processed result. It also handles
    situations where the expected columns might not be explicitly named, falling back
    on inferred numeric columns. The processed DataFrame includes the columns "Year",
    "Est" (Estimate), "Lo" (Lower), and "Hi" (Higher).

    Arguments:
        dhs_file (str | pathlib.Path): The path to the Excel file containing the DHS U5 table.
        sheet (str): The name of the sheet within the Excel file to be read.
        header (int): The row number (0-indexed) to use as the header row. Defaults to 1.

    Returns:
        pandas.DataFrame: A DataFrame with the columns "Year", "Est", "Lo", and "Hi" (after
        processing and cleaning). "Year" is coerced to integers; "Est", "Lo", and "Hi"
        are numeric values with missing rows dropped.

    Raises:
        ValueError: If the function cannot identify "Est", "Lo", or "Hi" columns in the data.
    """
    raw = pd.read_excel(dhs_file, sheet_name=sheet, header=header)
    raw.columns = [str(c).strip() for c in raw.columns]

    year_col: str | None = None
    for c in raw.columns:
        if str(c).strip().lower() == "year":
            year_col = c
            break
    if year_col is None:
        unnamed = [c for c in raw.columns if str(c).lower().startswith("unnamed")]
        year_col = unnamed[0] if unnamed else raw.columns[0]

    def find_col(patterns: list[str]) -> str | None:
        for c in raw.columns:
            name = str(c).strip().lower()
            if any(name == p for p in patterns):
                return c
        return None

    est_col = find_col(["est", "estimate", "mean"])
    lo_col = find_col(["lo", "lower", "lci"])
    hi_col = find_col(["hi", "higher", "upper", "uci"])

    if est_col is None or lo_col is None or hi_col is None:
        numeric_cols = [
            c for c in raw.columns if pd.to_numeric(raw[c], errors="coerce").notna().any()
        ]
        numeric_cols_wo_year = [c for c in numeric_cols if c != year_col]
        if len(numeric_cols_wo_year) < 3:
            raise ValueError(
                "Could not identify Est/Lo/Hi columns in DHS U5 sheet. "
                f"Found columns: {list(raw.columns)}"
            )
        est_col, lo_col, hi_col = (
            numeric_cols_wo_year[-3],
            numeric_cols_wo_year[-2],
            numeric_cols_wo_year[-1],
        )

    out = raw[[year_col, est_col, lo_col, hi_col]].copy()
    out.columns = ["Year", "Est", "Lo", "Hi"]

    out["Year"] = _coerce_year_series(out["Year"])
    out = out.loc[out["Year"].notna()].copy()

    for c in ["Est", "Lo", "Hi"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Est", "Lo", "Hi"], how="all")
    out = _maybe_scale_probs(out, ["Est", "Lo", "Hi"])
    out["Year"] = out["Year"].astype(int)

    return out
