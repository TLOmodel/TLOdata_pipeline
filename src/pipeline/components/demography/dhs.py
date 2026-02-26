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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from pipeline.components.resource_builder import BuildContext, ResourceBuilder, ResourceArtifact
from pipeline.components.utils import resolve_input_path


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DHSConfig:
    file: Path
    sheet_asfr: str
    sheet_u5: str
    u5_header: int = 1

    @staticmethod
    def from_ctx(ctx: BuildContext) -> Optional["DHSConfig"]:
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
    COMPONENT = "demography"

    EXPECTED_OUTPUTS = (
        "ResourceFile_ASFR_DHS.csv",
        "ResourceFile_Under_Five_Mortality_DHS.csv",
    )

    def run(self) -> list[ResourceArtifact]:
        # Optional builder: skip if not configured
        dhs = DHSConfig.from_ctx(self.ctx)
        if dhs is None:
            print("[SKIP] DHS not configured.")
            return []

        return super().run()

    def preflight(self) -> None:
        super().preflight()

        dhs = DHSConfig.from_ctx(self.ctx)
        assert dhs is not None  # for type checkers; run() guarantees this

        if not dhs.file.exists():
            raise FileNotFoundError(f"DHS Excel file not found: {dhs.file}")

    def load_data(self) -> Mapping[str, Any]:
        dhs = DHSConfig.from_ctx(self.ctx)
        assert dhs is not None

        asfr_raw = pd.read_excel(dhs.file, sheet_name=dhs.sheet_asfr)
        u5_raw = read_dhs_u5_table(dhs_file=dhs.file, sheet=dhs.sheet_u5, header=dhs.u5_header)

        return {"dhs": dhs, "asfr_raw": asfr_raw, "u5_raw": u5_raw}

    def build(self, raw: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
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
            raise AssertionError(f"U5 columns must be ['Year','Est','Lo','Hi'], got {list(u5.columns)}")

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
    x = pd.to_numeric(s, errors="coerce")

    if x.notna().any():
        x_non = x.dropna()
        if x_non.max() < 10 and x_non.min() > 0:
            x = x * 1000

    x = x.round()
    x = x.where((x >= 1900) & (x <= 2100))
    return x.astype("Int64")


def _maybe_scale_probs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    vals = pd.to_numeric(out[cols].stack(), errors="coerce").dropna()
    if vals.empty:
        return out

    if vals.median() > 1.0:
        out[cols] = out[cols] / 1000.0
    return out


def read_dhs_u5_table(dhs_file: str | Path, sheet: str, header: int = 1) -> pd.DataFrame:
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
        numeric_cols = [c for c in raw.columns if pd.to_numeric(raw[c], errors="coerce").notna().any()]
        numeric_cols_wo_year = [c for c in numeric_cols if c != year_col]
        if len(numeric_cols_wo_year) < 3:
            raise ValueError(
                "Could not identify Est/Lo/Hi columns in DHS U5 sheet. "
                f"Found columns: {list(raw.columns)}"
            )
        est_col, lo_col, hi_col = numeric_cols_wo_year[-3], numeric_cols_wo_year[-2], numeric_cols_wo_year[-1]

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