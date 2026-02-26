from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.components.common.fixes import (
    _norm_name,
    apply_cell_patches,
    load_name_mapping_csv,
    rename_index_from_file,
)


# ---------------------------------------------------------------------
# _norm_name
# ---------------------------------------------------------------------
def test_norm_name_trims_collapses_whitespace_and_nbsp() -> None:
    nbsp = "\u00a0"
    assert _norm_name(f"  A{nbsp}  B   C ") == "A B C"
    assert _norm_name(None) == ""
    assert _norm_name("") == ""


# ---------------------------------------------------------------------
# load_name_mapping_csv
# ---------------------------------------------------------------------
def test_load_name_mapping_csv_requires_from_to_columns() -> None:
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    with pytest.raises(ValueError, match="must have columns"):
        load_name_mapping_csv(df)


def test_load_name_mapping_csv_drops_blanks_and_normalizes() -> None:
    df = pd.DataFrame(
        {
            "from": ["  A ", " ", None, "B\u00a0C"],
            "to": [" X  ", "Y", "Z", "  D  E "],
        }
    )
    out = load_name_mapping_csv(df)
    # blank/None rows dropped
    assert out == {"A": "X", "B C": "D E"}


def test_load_name_mapping_csv_last_wins_on_duplicates() -> None:
    df = pd.DataFrame({"from": ["A", "A", "A"], "to": ["X", "Y", "Z"]})
    out = load_name_mapping_csv(df)
    assert out["A"] == "Z"


# ---------------------------------------------------------------------
# rename_index_from_file
# ---------------------------------------------------------------------
def test_rename_index_from_file_applies_mapping_and_preserves_unmapped() -> None:
    df = pd.DataFrame({"v": [1, 2, 3]}, index=["A", "B", "C"])
    mapping = pd.DataFrame({"from": ["A", "C"], "to": ["AA", "CC"]})
    out = rename_index_from_file(df, mapping)
    assert list(out.index) == ["AA", "B", "CC"]
    # values preserved
    assert out.loc["AA", "v"] == 1
    assert out.loc["B", "v"] == 2


def test_rename_index_from_file_normalizes_index_before_mapping() -> None:
    df = pd.DataFrame({"v": [1, 2]}, index=[" Blantyre\u00a0Rural ", "Lilongwe Rural"])
    mapping = pd.DataFrame(
        {
            "from": ["Blantyre Rural", "Lilongwe Rural"],
            "to": ["Blantyre", "Lilongwe"],
        }
    )
    out = rename_index_from_file(df, mapping)
    assert list(out.index) == ["Blantyre", "Lilongwe"]


def test_rename_index_from_file_strict_with_canonical_raises() -> None:
    df = pd.DataFrame({"v": [1]}, index=["A"])
    mapping = pd.DataFrame({"from": ["A"], "to": ["AA"]})

    with pytest.raises(ValueError, match="Unmatched index labels"):
        rename_index_from_file(df, mapping, canonical_districts=["A"], strict=True)


def test_rename_index_from_file_strict_passes_when_in_canonical() -> None:
    df = pd.DataFrame({"v": [1]}, index=["A"])
    mapping = pd.DataFrame({"from": ["A"], "to": ["AA"]})

    out = rename_index_from_file(df, mapping, canonical_districts=["AA"], strict=True)
    assert list(out.index) == ["AA"]


# ---------------------------------------------------------------------
# apply_cell_patches
# ---------------------------------------------------------------------
def test_apply_cell_patches_requires_patch_columns() -> None:
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame({"sheet": ["s"], "row_label": ["A"]})  # missing col_label,value

    with pytest.raises(KeyError, match="patches_df must have columns"):
        apply_cell_patches(df, patches, sheet="s")


def test_apply_cell_patches_no_patches_for_sheet_is_noop() -> None:
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame(
        {"sheet": ["other"], "row_label": ["A"], "col_label": ["x"], "value": [99]}
    )
    out = apply_cell_patches(df, patches, sheet="s")
    pd.testing.assert_frame_equal(out, df)


def test_apply_cell_patches_applies_value_with_normalized_row_and_col() -> None:
    df = pd.DataFrame({" Col\u00a0A ": [1], "B": [2]}, index=[" Row\u00a01 "])
    patches = pd.DataFrame(
        {
            "sheet": ["S"],
            "row_label": ["Row 1"],     # normalized match
            "col_label": ["Col A"],     # normalized match
            "value": [123],
        }
    )
    out = apply_cell_patches(df, patches, sheet="S")
    assert out.loc[" Row\u00a01 ", " Col\u00a0A "] == 123


def test_apply_cell_patches_blank_or_nan_becomes_none() -> None:
    df = pd.DataFrame({"x": [1, 2]}, index=["A", "B"])
    patches = pd.DataFrame(
        {
            "sheet": ["S", "S"],
            "row_label": ["A", "B"],
            "col_label": ["x", "x"],
            "value": ["", np.nan],
        }
    )
    out = apply_cell_patches(df, patches, sheet="S")
    assert out.loc["A", "x"] is None
    assert out.loc["B", "x"] is None


def test_apply_cell_patches_strict_raises_on_missing_row() -> None:
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame(
        {"sheet": ["S"], "row_label": ["NOPE"], "col_label": ["x"], "value": [9]}
    )
    with pytest.raises(KeyError, match="row_label not found"):
        apply_cell_patches(df, patches, sheet="S", strict=True)


def test_apply_cell_patches_non_strict_skips_missing_row_or_col() -> None:
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame(
        {
            "sheet": ["S", "S"],
            "row_label": ["NOPE", "A"],
            "col_label": ["x", "NOPECOL"],
            "value": [9, 9],
        }
    )
    out = apply_cell_patches(df, patches, sheet="S", strict=False)
    # unchanged
    pd.testing.assert_frame_equal(out, df)


def test_apply_cell_patches_index_is_labels_false_not_implemented() -> None:
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame(
        {"sheet": ["S"], "row_label": ["A"], "col_label": ["x"], "value": [9]}
    )
    with pytest.raises(NotImplementedError):
        apply_cell_patches(df, patches, sheet="S", index_is_labels=False)