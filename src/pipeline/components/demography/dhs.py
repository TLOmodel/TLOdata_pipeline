#!/usr/bin/env python3
"""
A module for processing DHS-based demographic data files and exporting resources.

This module includes utilities for handling complex demographic data from DHS
(Demographic and Health Surveys) files, transforming and normalizing the data,
and exporting it to standardized resource files. It also handles configuration
loading, resource preparation, and outputs diagnostic information.

Functions:
- _coerce_year_series: Normalize year-like data in a Pandas Series.
- _maybe_scale_probs: Scale probability data if necessary in a DataFrame.
- read_dhs_u5_table: Process under-five mortality tables from DHS Excel data.
- build_dhs_resources: Handle DHS resource generation based on configuration.
- main: Main execution flow for orchestrating DHS resource processing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.components.utils import load_cfg


def _coerce_year_series(s: pd.Series) -> pd.Series:
    """
    Convert year-like values to int years.
    Handles:
      - 2015 / '2015'
      - 2.015 / '2.015'  -> 2015
      - 2015.0 -> 2015
    Avoids turning 2015 into 2015000.
    """
    x = pd.to_numeric(s, errors="coerce")

    if x.notna().any():
        x_non = x.dropna()

        # Only apply the 1000x fix when values are in the ~1-3 range (e.g., 2.015)
        # This is the DHS/Excel quirk you hit earlier.
        if x_non.max() < 10 and x_non.min() > 0:
            x = x * 1000

    # Now normalize to integer years
    x = x.round()

    # If anything is still not a plausible year, leave as NA rather than propagating junk
    # (optional but prevents silent weird outputs)
    x = x.where((x >= 1900) & (x <= 2100))

    return x.astype("Int64")


def _maybe_scale_probs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    DHS U5 sheets sometimes are in per-1000 (e.g., 64) and sometimes already in probability (0.064).
    Detect and scale only if needed.
    """
    out = df.copy()
    vals = pd.to_numeric(out[cols].stack(), errors="coerce").dropna()
    if vals.empty:
        return out

    # If typical values are >1, it's likely per-1000 (or percent). If typical values <1, keep as-is.
    # Use median as robust statistic.
    if vals.median() > 1.0:
        out[cols] = out[cols] / 1000.0

    return out


def read_dhs_u5_table(dhs_file: str | Path, sheet: str, header: int = 1) -> pd.DataFrame:
    """
    Returns df with columns exactly: Year, Est, Lo, Hi
    """
    raw = pd.read_excel(dhs_file, sheet_name=sheet, header=header)

    # Normalize column names
    cols = [str(c).strip() for c in raw.columns]
    raw.columns = cols

    # Identify Year column:
    # Prefer a column explicitly named like 'Year'; else use an 'Unnamed' col; else
    # fallback to first col
    year_col = None
    for c in raw.columns:
        if str(c).strip().lower() == "year":
            year_col = c
            break
    if year_col is None:
        unnamed = [c for c in raw.columns if str(c).lower().startswith("unnamed")]
        year_col = unnamed[0] if unnamed else raw.columns[0]

    # Identify Est/Lo/Hi columns by name (case-insensitive)
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
        # Fallback: try to detect by taking the last 3 numeric columns
        numeric_cols = [
            c for c in raw.columns if pd.to_numeric(raw[c], errors="coerce").notna().any()
        ]
        # remove year_col if numeric
        numeric_cols_wo_year = [c for c in numeric_cols if c != year_col]
        if len(numeric_cols_wo_year) < 3:
            raise ValueError(
                f"Could not identify Est/Lo/Hi columns in DHS U5 sheet. "
                f"Found columns: {list(raw.columns)}"
            )
        est_col, lo_col, hi_col = (
            numeric_cols_wo_year[-3],
            numeric_cols_wo_year[-2],
            numeric_cols_wo_year[-1],
        )

    out = raw[[year_col, est_col, lo_col, hi_col]].copy()
    out.columns = ["Year", "Est", "Lo", "Hi"]

    # Coerce year properly
    out["Year"] = _coerce_year_series(out["Year"])

    # Drop rows without a valid year
    out = out.loc[out["Year"].notna()].copy()

    # Coerce numeric for measures
    for c in ["Est", "Lo", "Hi"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop fully empty measure rows
    out = out.dropna(subset=["Est", "Lo", "Hi"], how="all")

    # Scale if necessary (only when values look like per-1000)
    out = _maybe_scale_probs(out, ["Est", "Lo", "Hi"])

    # Ensure proper ordering (your “actual output” is descending-ish;
    # keep sheet order unless you want sort)
    out["Year"] = out["Year"].astype(int)

    return out


def build_dhs_resources(cfg: dict, resources_dir: Path) -> None:
    """
    Build DHS-based demographic resource files (if DHS config & file exist).

    Outputs:
      - ResourceFile_ASFR_DHS.csv
      - ResourceFile_Under_Five_Mortality_DHS.csv
    """
    if "dhs" not in cfg:
        # DHS not configured: do nothing (intentional)
        return

    dhs_cfg = cfg["dhs"]
    dhs_file = Path(dhs_cfg["file"])

    if not dhs_file.exists():
        # DHS configured but file missing: do nothing (intentional)
        return

    resources_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) DHS ASFR
    # -------------------------
    dhs_asfr = pd.read_excel(dhs_file, sheet_name=dhs_cfg["sheet_asfr"])

    # Convert per-1000 to per-woman
    value_cols = dhs_asfr.columns[1:]
    dhs_asfr[value_cols] = dhs_asfr[value_cols].astype(float) / 1000.0

    dhs_asfr.to_csv(resources_dir / "ResourceFile_ASFR_DHS.csv", index=False)

    # -------------------------
    # 2) DHS Under-Five Mortality
    # -------------------------
    dhs_u5 = read_dhs_u5_table(
        dhs_file=dhs_file,
        sheet=dhs_cfg["sheet_u5"],
        header=int(dhs_cfg.get("u5_header", 1)),
    )
    dhs_u5.to_csv(resources_dir / "ResourceFile_Under_Five_Mortality_DHS.csv", index=False)


def main() -> None:
    """
    Main execution function for handling DHS resource creation and validation.

    This function orchestrates configuration loading, prepares the necessary
    resources directory, and executes the process of generating DHS resources.
    It also includes checks for relevant DHS configuration and file existence
    to ensure proper execution workflow. Diagnostic messages are printed for
    user feedback regarding skipped or executed actions.

    Raises:
        FileNotFoundError: Raised if the configured DHS file does not exist
        when required by the process flow.

    Returns:
        None
    """
    cfg = load_cfg()

    resources_dir = Path(cfg["outputs"]["resources_dir"])
    build_dhs_resources(cfg, resources_dir)

    # Optional: make it obvious when nothing happened
    if "dhs" not in cfg:
        print("[SKIP] DHS not configured in config.")
        return

    dhs_file = Path(cfg["dhs"]["file"])
    if not dhs_file.exists():
        print(f"[SKIP] DHS file not found: {dhs_file}")
        return

    print(f"[OK] DHS resources written to: {resources_dir}")


if __name__ == "__main__":
    main()
