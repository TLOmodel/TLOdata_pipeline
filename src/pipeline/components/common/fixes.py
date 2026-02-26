"""
Utilities for working with WPP data, including data processing, reading from Excel sheets,
index mapping, and applying patches to DataFrames. These functions and classes provide
tools to standardize and process demographic data for analysis.

Functions:
  - reformat_date_period_for_wpp: Convert WPP period column values into inclusive year ranges.
  - _norm_name: Normalize strings for robust matching by trimming and collapsing whitespace.
  - load_name_mapping_csv: Build a mapping dictionary from specified DataFrame columns.
  - rename_index_from_file: Rename DataFrame index values using mapping defined in a CSV.
  - apply_cell_patches: Apply modifications from a DataFrame of patches to a target DataFrame.

Classes:
  - WPPReader: Reads and processes WPP data with specific configurations for country-level
    filtering, column removal, and data concatenation.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def reformat_date_period_for_wpp(df: pd.DataFrame, period_col: str = "Period") -> None:
    """
    Convert WPP Period like "2010-2015" into inclusive year range "2010-2014".
    Modifies df in-place.
    """
    t = df[period_col].astype(str).str.split("-", n=1, expand=True)
    lo = t[0].astype(int)
    hi = t[1].astype(int) - 1
    df[period_col] = lo.astype(str) + "-" + hi.astype(str)


@dataclass(frozen=True)
class WPPReader:
    """
    Reads and processes WPP data from Excel sheets.

    Parameters:
      - country_label: the exact WPP country/location label to match
      - header_row: Excel header row index passed to pandas.read_excel
      - country_col_index: fallback column index for country/location if name-based match fails
      - drop_col_positions: positional indices to drop as metadata after filtering
    """

    country_label: str
    header_row: int = 16
    country_col_index: int = 2
    drop_col_positions: Sequence[int] = (0, 2, 3, 4, 5, 6)

    # common WPP location column names
    _COUNTRY_COL_CANDIDATES: Sequence[str] = (
        "Location",
        "Country",
        "Area",
        "Region, subregion, country or area",
    )

    def read_concat(self, file_path: str | Path, sheets: Iterable[str], **read_excel_kwargs) -> pd.DataFrame:
        file_path = str(file_path)
        frames = [
            pd.read_excel(file_path, sheet_name=s, header=self.header_row, **read_excel_kwargs)
            for s in sheets
        ]
        return pd.concat(frames, sort=False, ignore_index=True)

    def _country_col(self, df: pd.DataFrame) -> str:
        for name in self._COUNTRY_COL_CANDIDATES:
            if name in df.columns:
                return name
        # fallback to index (original behavior)
        return df.columns[self.country_col_index]

    @staticmethod
    def _norm_str(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.replace("\u00a0", " ", regex=False)  # NBSP
            .str.strip()
        )

    def filter_country(self, df: pd.DataFrame) -> pd.DataFrame:
        ccol = self._country_col(df)

        series = self._norm_str(df[ccol])
        label = str(self.country_label).replace("\u00a0", " ").strip()

        # casefold makes matching robust to casing
        out = df.loc[series.str.casefold() == label.casefold()].copy()

        if out.empty:
            sample = series.dropna().unique()[:20]
            raise ValueError(
                "WPPReader.filter_country() produced empty result.\n"
                f"- country_label={label!r}\n"
                f"- country_col={ccol!r} (fallback_index={self.country_col_index})\n"
                f"- sample_values={list(sample)!r}\n"
                "Fix by adjusting wpp.country_label, header_row, or country_col_index in config."
            )

        return out.reset_index(drop=True)

    def drop_metadata_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        drop_cols = [cols[i] for i in self.drop_col_positions if 0 <= i < len(cols)]
        return df.drop(columns=drop_cols)

    def read_country_table(
        self,
        file_path: str | Path,
        sheets: Iterable[str],
        extra_cols: dict[str, object] | None = None,
        **read_excel_kwargs,
    ) -> pd.DataFrame:
        df = self.read_concat(file_path, sheets, **read_excel_kwargs)
        df = self.filter_country(df)
        df = self.drop_metadata_cols(df)

        if extra_cols:
            for k, v in extra_cols.items():
                df[k] = v

        return df


def _norm_name(x: object) -> str:
    """Normalize a label for robust matching: NBSP->space, trim, collapse whitespace, case-sensitive."""
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ")
    s = " ".join(s.strip().split())
    return s


def load_name_mapping_csv(df: pd.DataFrame) -> dict[str, str]:
    """
    Build a mapping dict from a DataFrame with columns: 'from', 'to'.
    - Normalizes whitespace
    - Drops blank rows
    """
    required = {"from", "to"}
    if not required.issubset(df.columns):
        raise ValueError(f"Mapping table must have columns {required}. Found: {list(df.columns)}")

    src = df.copy()

    src["from"] = src["from"].map(_norm_name)
    src["to"] = src["to"].map(_norm_name)

    src = src[(src["from"] != "") & (src["to"] != "")]
    # "last wins" on duplicates (common when analysts add overrides later)
    return dict(zip(src["from"], src["to"], strict=False))


def rename_index_from_file(
    df: pd.DataFrame,
    dist_names: pd.DataFrame,
    *,
    canonical_districts: Iterable[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Rename df.index using mapping in dist_names DataFrame (columns: from,to).

    If strict=True and canonical_districts is provided:
      - error if any renamed index values are not in canonical_districts (after normalization)

    Note: Returns renamed df only (no applied_map), to match your current usage.
    """
    mapping = load_name_mapping_csv(dist_names)

    out = df.copy()
    before = out.index.map(_norm_name)
    out.index = [mapping.get(b, b) for b in before]

    if strict and canonical_districts is not None:
        canonical = {_norm_name(x) for x in canonical_districts}
        observed = {_norm_name(x) for x in out.index}
        unmatched = sorted(observed - canonical)
        if unmatched:
            raise ValueError(
                f"Unmatched index labels after CSV renames: {unmatched[:10]}"
                + (f" (and {len(unmatched) - 10} more)" if len(unmatched) > 10 else "")
            )

    return out


def apply_cell_patches(
    df: pd.DataFrame,
    patches_df: pd.DataFrame,
    *,
    sheet: str,
    index_is_labels: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Apply cell patches to a DataFrame representing one sheet.

    patches_df required columns: sheet,row_label,col_label,value

    Matching:
      - row_label matched against df.index (normalized by _norm_name) when index_is_labels=True
      - col_label matched against df.columns (normalized by _norm_name)

    strict=True:
      - raises KeyError if row/col not found
      - raises KeyError if patch table is missing required columns

    strict=False:
      - silently skips patches whose row/col cannot be matched
    """
    required = {"sheet", "row_label", "col_label", "value"}
    if not required.issubset(patches_df.columns):
        raise KeyError(f"patches_df must have columns {required}. Found: {list(patches_df.columns)}")

    out = df.copy()

    patches = patches_df.loc[patches_df["sheet"].astype(str) == str(sheet)].copy()
    if patches.empty:
        return out

    if not index_is_labels:
        raise NotImplementedError("index_is_labels=False not implemented in this helper.")

    # Build normalized -> original key lookups (first wins)
    idx_norm_to_key: dict[str, object] = {}
    for k in out.index:
        nk = _norm_name(k)
        idx_norm_to_key.setdefault(nk, k)

    col_norm_to_key: dict[str, object] = {}
    for c in out.columns:
        nc = _norm_name(c)
        col_norm_to_key.setdefault(nc, c)

    for _, p in patches.iterrows():
        row_label = _norm_name(p["row_label"])
        col_label = _norm_name(p["col_label"])
        raw_val = p["value"]

        idx_key = idx_norm_to_key.get(row_label)
        if idx_key is None:
            if strict:
                raise KeyError(f"[{sheet}] Patch row_label not found in index: {row_label!r}")
            continue

        col_key = col_norm_to_key.get(col_label)
        if col_key is None:
            if strict:
                raise KeyError(f"[{sheet}] Patch col_label not found in columns: {col_label!r}")
            continue

        # Convert value: blank/NaN -> None
        if pd.isna(raw_val) or str(raw_val).strip() == "":
            val: Any = None
        else:
            val = raw_val

        out.loc[idx_key, col_key] = val

    return out