#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def reformat_date_period_for_wpp(df: pd.DataFrame, period_col: str = "Period") -> None:
    """
    Convert WPP Period like 2010-2015 into inclusive year range 2010-2014.
    Modifies df in-place.
    """
    t = df[period_col].astype(str).str.split("-", n=1, expand=True)
    lo = t[0].astype(int)
    hi = t[1].astype(int) - 1
    df[period_col] = lo.astype(str) + "-" + hi.astype(str)


@dataclass(frozen=True)
class WPPReader:
    """
    Centralized WPP reading/cleaning to avoid repetition.

    Assumptions match your current script:
      - Each sheet has a country label column at index 2 (3rd column) that we filter on.
      - We drop the first 7 columns except 'Variant' (your current drop pattern).
    """
    country_label: str
    header_row: int = 16
    country_col_index: int = 2
    # columns to drop by positional index (matches your current code)
    drop_col_positions: Sequence[int] = (0, 2, 3, 4, 5, 6)

    def read_concat(self, file_path: str | Path, sheets: Iterable[str], **read_excel_kwargs) -> pd.DataFrame:
        file_path = str(file_path)
        frames = [
            pd.read_excel(file_path, sheet_name=s, header=self.header_row, **read_excel_kwargs)
            for s in sheets
        ]
        return pd.concat(frames, sort=False, ignore_index=True)

    def filter_country(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter rows where the "country" column (by index) equals configured label
        ccol = df.columns[self.country_col_index]
        out = df.loc[df[ccol] == self.country_label].copy()
        return out.reset_index(drop=True)

    def drop_metadata_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        drop_cols = [cols[i] for i in self.drop_col_positions if i < len(cols)]
        return df.drop(columns=drop_cols)

    def read_country_table(
        self,
        file_path: str | Path,
        sheets: Iterable[str],
        extra_cols: Optional[dict[str, object]] = None,
        **read_excel_kwargs
    ) -> pd.DataFrame:
        """
        Read WPP excel tables (possibly multiple sheets), concatenate, filter to country, and drop metadata columns.
        """
        df = self.read_concat(file_path, sheets, **read_excel_kwargs)
        df = self.filter_country(df)
        df = self.drop_metadata_cols(df)
        if extra_cols:
            for k, v in extra_cols.items():
                df[k] = v
        return df


def melt_year_age_groups(
    df: pd.DataFrame,
    id_vars: list[str],
    value_name: str,
    var_name: str = "Age_Grp",
    multiplier: float = 1.0,
    value_cols_slice: Optional[slice] = None,
) -> pd.DataFrame:
    """
    Helper to scale numeric value columns and then melt.
    `value_cols_slice` lets you apply scaling only to a subset (to match WPP layouts).
    """
    out = df.copy()
    if value_cols_slice is not None:
        cols = out.columns[value_cols_slice]
        out[cols] = out[cols] * multiplier
    else:
        # scale all non-id columns
        cols = [c for c in out.columns if c not in id_vars]
        out[cols] = out[cols] * multiplier

    return out.melt(id_vars=id_vars, value_name=value_name, var_name=var_name)
