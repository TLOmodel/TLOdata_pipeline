"""
Utility functions to handle DataFrame transformations including name mapping,
index renaming, and applying cell patches from an external source.

This module offers a set of utility functions to streamline working with pandas
DataFrames. It includes functions to parse and apply mappings from CSVs,
rename DataFrame indices using external files, and apply data patches to
specific DataFrames. It is well-suited for use cases requiring structured data
manipulation and augmentation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def _norm_name(s: str) -> str:
    """Lightweight, safe normalisation: trim + collapse whitespace."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ")  # non-breaking space
    s = " ".join(s.strip().split())
    return s


def load_name_mapping_csv(df: pd.DataFrame) -> dict[str, str]:
    """
    Parses a DataFrame to generate a mapping dictionary.

    This function takes a pandas DataFrame with specific columns and processes it to
    create a mapping of keys to values. It verifies the presence of required
    columns, removes whitespace, and excludes rows with blank entries before
    creating the mapping dictionary.

    Arguments:
        df: pandas.DataFrame
            A DataFrame containing the mapping data. It must include the columns
            "from" and "to".

    Raises:
        ValueError: Raised if the required columns "from" and "to" are not present
        in the DataFrame.

    Returns:
        dict
            A dictionary mapping values from the "from" column to the "to" column.
    """
    required = {"from", "to"}
    if not required.issubset(df.columns):
        raise ValueError(f"Mapping CSV must have columns {required}. Found: {list(df.columns)}")
    # strip whitespace and drop blanks
    df["from"] = df["from"].astype(str).str.strip()
    df["to"] = df["to"].astype(str).str.strip()
    df = df[(df["from"] != "") & (df["to"] != "")]
    return dict(zip(df["from"], df["to"], strict=False))


def rename_index_from_file(
    df: pd.DataFrame,
    dist_names: pd.DataFrame,
    *,
    canonical_districts: Iterable[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Replace hard-coded df.rename(index={...}) with a CSV-driven mapping.

    - Applies mapping to df.index (exactly like your original code)
    - Returns (df_renamed, applied_map)
    - If strict=True and canonical_districts provided, errors if any index values
      (after rename) are not in canonical_districts.
    """
    mapping = load_name_mapping_csv(dist_names)

    out = df.copy()
    before = out.index.astype(str)

    out.index = before.map(lambda x: mapping.get(x, x))

    if strict and canonical_districts is not None:
        canonical = set(str(x).strip() for x in canonical_districts)
        observed = set(str(x).strip() for x in out.index)
        unmatched = sorted(observed - canonical)
        if unmatched:
            raise ValueError(
                f"Unmatched index labels after CSV renames: {unmatched[:10]}"
                + (f" (and {len(unmatched)-10} more)" if len(unmatched) > 10 else "")
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

    patches_df columns: sheet,row_label,col_label,value
    - row_label targets df.index if index_is_labels else targets a
      'District'/label column (not implemented here).
    - value: empty/NaN -> None

    strict=True => error if row/col not found.
    """
    out = df.copy()
    patches = patches_df.loc[patches_df["sheet"].astype(str) == str(sheet)]

    for _, p in patches.iterrows():
        row_label = _norm_name(p["row_label"])
        col_label = _norm_name(p["col_label"])
        raw_val = p["value"]

        if index_is_labels:
            if row_label not in map(_norm_name, out.index.astype(str)):
                if strict:
                    raise KeyError(f"[{sheet}] Patch row_label not found in index: '{row_label}'")
            # Find exact index key (preserve original)
            idx_key = next(ix for ix in out.index if _norm_name(ix) == row_label)
        else:
            raise NotImplementedError("index_is_labels=False not implemented in this helper.")

        if col_label not in map(_norm_name, out.columns.astype(str)):
            if strict:
                raise KeyError(f"[{sheet}] Patch col_label not found in columns: '{col_label}'")

            # Find exact col key
        col_key = next(cx for cx in out.columns if _norm_name(cx) == col_label)

        # Convert value
        if pd.isna(raw_val) or str(raw_val).strip() == "":
            val: Any = None
        else:
            # keep as string; downstream can cast. If you want numeric casting, add it here.
            val = raw_val

        out.loc[idx_key, col_key] = val

    return out
