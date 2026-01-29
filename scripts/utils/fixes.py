from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np


def _norm_name(s: str) -> str:
    """Lightweight, safe normalisation: trim + collapse whitespace."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00A0", " ")  # non-breaking space
    s = " ".join(s.strip().split())
    return s

def load_name_mapping_csv(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path)
    required = {"from", "to"}
    if not required.issubset(df.columns):
        raise ValueError(f"Mapping CSV must have columns {required}. Found: {list(df.columns)}")
    # strip whitespace and drop blanks
    df["from"] = df["from"].astype(str).str.strip()
    df["to"] = df["to"].astype(str).str.strip()
    df = df[(df["from"] != "") & (df["to"] != "")]
    return dict(zip(df["from"], df["to"]))


def rename_index_from_csv(
    df: pd.DataFrame,
    mapping_csv: Path,
    *,
    canonical_districts: Optional[Iterable[str]] = None,
    strict: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Replace hard-coded df.rename(index={...}) with a CSV-driven mapping.

    - Applies mapping to df.index (exactly like your original code)
    - Returns (df_renamed, applied_map)
    - If strict=True and canonical_districts provided, errors if any index values
      (after rename) are not in canonical_districts.
    """
    mapping = load_name_mapping_csv(mapping_csv)

    out = df.copy()
    before = out.index.astype(str)

    out.index = before.map(lambda x: mapping.get(x, x))
    applied = {k: v for k, v in mapping.items() if k in set(before)}

    if strict and canonical_districts is not None:
        canonical = set(str(x).strip() for x in canonical_districts)
        observed = set(str(x).strip() for x in out.index)
        unmatched = sorted(observed - canonical)
        if unmatched:
            raise ValueError(f"Unmatched index labels after CSV renames: {unmatched[:10]}"
                             + (f" (and {len(unmatched)-10} more)" if len(unmatched) > 10 else ""))

    return out, applied

def _read_cell_patches_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["sheet", "row_label", "col_label", "value"])

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    required = {"sheet", "row_label", "col_label", "value"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Cell patches CSV must have columns {required}. Found: {list(df.columns)}")
    return df


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
    - row_label targets df.index if index_is_labels else targets a 'District'/label column (not implemented here).
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
                else:
                    continue
            # Find exact index key (preserve original)
            idx_key = next(ix for ix in out.index if _norm_name(ix) == row_label)
        else:
            raise NotImplementedError("index_is_labels=False not implemented in this helper.")

        if col_label not in map(_norm_name, out.columns.astype(str)):
            if strict:
                raise KeyError(f"[{sheet}] Patch col_label not found in columns: '{col_label}'")
            else:
                continue
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


def load_cell_patches(cfg: Dict[str, Any]) -> pd.DataFrame:
    cell_fixes = cfg.get("cell_fixes", {}) if isinstance(cfg.get("cell_fixes", {}), dict) else {}
    patches_file = cell_fixes.get("patches_file")
    if not patches_file:
        return pd.DataFrame(columns=["sheet", "row_label", "col_label", "value"])
    return _read_cell_patches_csv(Path(str(patches_file)))
