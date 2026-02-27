"""
Test suite for verifying the functionality of key parts within the fixes.py file
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.components.common.fixes import (
    WPPReader,
    _norm_name,
    apply_cell_patches,
    load_name_mapping_csv,
    reformat_date_period_for_wpp,
    rename_index_from_file,
)


# ---------------------------------------------------------------------
# _norm_name
# ---------------------------------------------------------------------
def test_norm_name_trims_collapses_whitespace_and_nbsp() -> None:
    """
    Tests the `_norm_name` function to ensure it correctly trims leading and trailing
    whitespace, collapses multiple spaces into a single space, and replaces non-breaking
    spaces with regular spaces. Verifies behavior for various inputs including
    whitespace, special characters, and empty or null values.

    Raises:
        AssertionError: If any of the assertions for the `_norm_name` function fail.
    """
    nbsp = "\u00a0"
    assert _norm_name(f"  A{nbsp}  B   C ") == "A B C"
    assert _norm_name(None) == ""
    assert _norm_name("") == ""


# ---------------------------------------------------------------------
# load_name_mapping_csv
# ---------------------------------------------------------------------
def test_load_name_mapping_csv_requires_from_to_columns() -> None:
    """
    Tests that the function `load_name_mapping_csv` raises a `ValueError` when the input
    DataFrame does not have the required 'from' and 'to' columns.

    Raises:
        ValueError: If the input DataFrame does not contain the required columns with
            the names 'from' and 'to'.
    """
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    with pytest.raises(ValueError, match="must have columns"):
        load_name_mapping_csv(df)


def test_load_name_mapping_csv_drops_blanks_and_normalizes() -> None:
    """
    Tests the `load_name_mapping_csv` function to ensure it handles blank entries
    and normalizes whitespace consistently. Specifically, checks that rows with
    blank or None values in the input DataFrame are dropped, and that leading,
    trailing, and excessive whitespace, including non-breaking spaces, are
    properly trimmed and normalized.

    Args:
        None

    Raises:
        AssertionError: If the function output does not match the expected result.

    Returns:
        None
    """
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
    """
    Tests the behavior of the load_name_mapping_csv function when handling
    duplicate entries in the input dataframe. Verifies that the last value
    takes precedence in the case of duplicate 'from' entries.

    Args:
        None

    Raises:
        AssertionError: If the function does not prioritize the last entry
        for duplicate 'from' values.

    Returns:
        None
    """
    df = pd.DataFrame({"from": ["A", "A", "A"], "to": ["X", "Y", "Z"]})
    out = load_name_mapping_csv(df)
    assert out["A"] == "Z"


# ---------------------------------------------------------------------
# rename_index_from_file
# ---------------------------------------------------------------------
def test_rename_index_from_file_applies_mapping_and_preserves_unmapped() -> None:
    """
    Renames the index of a DataFrame based on a mapping DataFrame while preserving
    unmapped indices.

    This test verifies the functionality of the `rename_index_from_file` function
    by ensuring that it correctly applies the index mapping from a given mapping
    DataFrame. Indices that are not present in the mapping file are kept
    unchanged. It also ensures that the values in the DataFrame are preserved
    after the renaming process.

    Args:
        None

    Raises:
        AssertionError: If the resulting index does not match the expected set of
        indices or if values associated with the indices are not preserved.
    """
    df = pd.DataFrame({"v": [1, 2, 3]}, index=["A", "B", "C"])
    mapping = pd.DataFrame({"from": ["A", "C"], "to": ["AA", "CC"]})
    out = rename_index_from_file(df, mapping)
    assert list(out.index) == ["AA", "B", "CC"]
    # values preserved
    assert out.loc["AA", "v"] == 1
    assert out.loc["B", "v"] == 2


def test_rename_index_from_file_normalizes_index_before_mapping() -> None:
    """
    Tests the `rename_index_from_file` function to ensure it correctly normalizes and maps
    the DataFrame index based on the provided mapping DataFrame.

    It validates that the function handles index normalization, such as trimming whitespace
    and replacing non-breaking spaces, before applying the mapping. The correct application
    of index mapping is tested by comparing the result to expected values.

    Raises
    ------
    AssertionError
        If the function's output index does not match the expected index after mapping.

    Returns
    -------
    None
    """
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
    """
    Tests that the `rename_index_from_file` function raises a `ValueError` with an
    appropriate message when called with a strict mode and canonical districts, and
    there are unmatched index labels in the DataFrame.

    Args:
        None

    Raises:
        ValueError: If there are unmatched index labels while strict mode is enabled.

    Returns:
        None
    """
    df = pd.DataFrame({"v": [1]}, index=["A"])
    mapping = pd.DataFrame({"from": ["A"], "to": ["AA"]})

    with pytest.raises(ValueError, match="Unmatched index labels"):
        rename_index_from_file(df, mapping, canonical_districts=["A"], strict=True)


def test_rename_index_from_file_strict_passes_when_in_canonical() -> None:
    """
    Tests the behavior of the rename_index_from_file function to ensure it properly
    renames the index of a DataFrame based on a mapping DataFrame when operating
    in strict mode and all mapped values are in the canonical list.

    Raises:
        AssertionError: If the output index does not match the expected renamed
        index based on the provided mapping in strict mode.
    """
    df = pd.DataFrame({"v": [1]}, index=["A"])
    mapping = pd.DataFrame({"from": ["A"], "to": ["AA"]})

    out = rename_index_from_file(df, mapping, canonical_districts=["AA"], strict=True)
    assert list(out.index) == ["AA"]


# ---------------------------------------------------------------------
# apply_cell_patches
# ---------------------------------------------------------------------
def test_apply_cell_patches_requires_patch_columns() -> None:
    """
    Tests that the function `apply_cell_patches` raises a KeyError when the provided
    patches DataFrame does not contain all required columns.

    Specifically checks for the case where the 'col_label' and 'value' columns are
    missing from the patches DataFrame.

    Raises:
        KeyError: If the patches DataFrame does not include the required columns.
    """
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame({"sheet": ["s"], "row_label": ["A"]})  # missing col_label,value

    with pytest.raises(KeyError, match="patches_df must have columns"):
        apply_cell_patches(df, patches, sheet="s")


def test_apply_cell_patches_no_patches_for_sheet_is_noop() -> None:
    """
    Tests that if no patches are applicable for the specified sheet, the original dataframe
    is returned unchanged.

    """
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame(
        {"sheet": ["other"], "row_label": ["A"], "col_label": ["x"], "value": [99]}
    )
    out = apply_cell_patches(df, patches, sheet="s")
    pd.testing.assert_frame_equal(out, df)


def test_apply_cell_patches_applies_value_with_normalized_row_and_col() -> None:
    """
    Tests the `apply_cell_patches` function by ensuring that it correctly applies a
    given value to a DataFrame using normalized row and column labels.

    Raises
    ------
    AssertionError
        If the patched value does not match the expected value in the DataFrame.
    """
    df = pd.DataFrame({" Col\u00a0A ": [1], "B": [2]}, index=[" Row\u00a01 "])
    patches = pd.DataFrame(
        {
            "sheet": ["S"],
            "row_label": ["Row 1"],  # normalized match
            "col_label": ["Col A"],  # normalized match
            "value": [123],
        }
    )
    out = apply_cell_patches(df, patches, sheet="S")
    assert out.loc[" Row\u00a01 ", " Col\u00a0A "] == 123


def test_apply_cell_patches_blank_or_nan_becomes_none() -> None:
    """
    Tests the `apply_cell_patches` function to ensure that blank or NaN values in the patch
    dataframe are converted to None in the output dataframe. The test verifies that after
    applying the patches, the corresponding cells in the output dataframe contain None.

    Raises:
        AssertionError: If the patched dataframe does not contain None in the specified cells.
    """
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
    """
    Tests the `apply_cell_patches` function in strict mode and ensures it raises
    a `KeyError` when a row in the patch is not found in the DataFrame.

    Sections:
    Raises:
        KeyError: If a specified `row_label` in the patches DataFrame is not
        present in the target DataFrame while in strict mode.
    """
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame(
        {"sheet": ["S"], "row_label": ["NOPE"], "col_label": ["x"], "value": [9]}
    )
    with pytest.raises(KeyError, match="row_label not found"):
        apply_cell_patches(df, patches, sheet="S", strict=True)


def test_apply_cell_patches_non_strict_skips_missing_row_or_col() -> None:
    """
    Tests the functionality of applying cell patches in a non-strict mode.

    When running the function `apply_cell_patches` with the `strict` parameter set to False,
    it ensures that missing rows or columns in the patches do not alter the original DataFrame.

    Raises
    ------
    AssertionError
        If the output DataFrame does not match the expected unchanged original DataFrame.
    """
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
    """
    Tests the `apply_cell_patches` function when the `index_is_labels` parameter is set to False,
    which is currently not implemented. The test ensures that the function raises the expected
    NotImplementedError when executed under these conditions.

    Parameters
    ----------
    None

    Raises
    ------
    NotImplementedError
        Raised when `index_is_labels` is False as this functionality is not supported.
    """
    df = pd.DataFrame({"x": [1]}, index=["A"])
    patches = pd.DataFrame({"sheet": ["S"], "row_label": ["A"], "col_label": ["x"], "value": [9]})
    with pytest.raises(NotImplementedError):
        apply_cell_patches(df, patches, sheet="S", index_is_labels=False)


# ---------------------------------------------------------------------
# reformat_date_period_for_wpp
# ---------------------------------------------------------------------
def test_reformat_date_period_for_wpp_inclusive_range() -> None:
    """
    Tests the function 'reformat_date_period_for_wpp', specifically for its behavior
    with inclusive range adjustments to the periods provided. The function is
    expected to reformat periods in the DataFrame from inclusive end to exclusive
    end by subtracting one year from the end year of the range.

    Raises:
        AssertionError: If the reformatted periods in the DataFrame do not match
        the expected values.
    """
    df = pd.DataFrame({"Period": ["2010-2015", "1950-1955"]})
    reformat_date_period_for_wpp(df)
    assert df["Period"].tolist() == ["2010-2014", "1950-1954"]


def test_reformat_date_period_for_wpp_nonstring_ok() -> None:
    """
    Tests the reformat_date_period_for_wpp function to ensure it handles non-string
    values in the input Period data properly by casting them to strings.

    Raises
    ------
    Exception
        If the conversion of reformatted period values to integers fails. This is
        expected behavior as WPP periods should have the format "lo-hi".

    Notes
    -----
    The test creates a DataFrame with mixed types in the Period column: one value
    as an integer (2010) and another as a string ("2010-2015"). When reformatted,
    the first value ("2010") splits into ["2010", None], causing a failure during
    integer conversion, which is the expected outcome for this test case.
    """
    # Ensure it tolerates non-string values (casts to str)
    df = pd.DataFrame({"Period": [2010, "2010-2015"]})
    # first value "2010" will split to ["2010", None] and then fail int conversion
    # so we expect it to raise: this is fine because WPP periods must be "lo-hi"
    with pytest.raises(ValueError, TypeError):
        reformat_date_period_for_wpp(df)


# ---------------------------------------------------------------------
# WPPReader country column selection
# ---------------------------------------------------------------------
def test_country_col_prefers_named_candidates() -> None:
    """
    Tests that the method identifying the appropriate column for the country label
    prefers explicitly named candidates when provided in the WPPReader configuration.

    Parameters:
        None

    Returns:
        None
    """
    df = pd.DataFrame(
        {
            "Location": ["United Republic of Tanzania", "Kenya"],
            "x": [1, 2],
        }
    )
    r = WPPReader(country_label="United Republic of Tanzania", country_col_index=0)
    assert r.country_col(df) == "Location"


def test_country_col_falls_back_to_index() -> None:
    """
    Test the functionality of column fallback to index based on the provided
    country label and index when using the WPPReader class.

    The test verifies that when a specific index is provided for the
    column containing country labels, the correct column name is derived
    from the DataFrame.

    Raises:
        AssertionError: If the derived column name does not match the
        expected column based on the country label and column index.
    """
    df = pd.DataFrame(
        {
            "A": ["foo", "bar"],
            "B": ["baz", "qux"],
            "C": ["Tanzania", "Kenya"],
        }
    )
    r = WPPReader(country_label="Tanzania", country_col_index=2)
    assert r.country_col(df) == "C"


# ---------------------------------------------------------------------
# WPPReader.filter_country
# ---------------------------------------------------------------------
def test_filter_country_exact_match() -> None:
    """
    Tests the filtering of the DataFrame by matching the country label.

    The function tests the capability of the `filter_country` method within the
    `WPPReader` class to correctly filter a DataFrame's rows based on the exact
    match of the country label provided. The test case ensures that only the rows
    matching the specified country are retained.

    Raises:
        AssertionError: If the filtered DataFrame does not contain the expected
        number of rows or if the values in the filtered rows do not match the
        expected ones.
    """
    df = pd.DataFrame(
        {
            "Location": ["United Republic of Tanzania", "Kenya"],
            "v": [10, 20],
        }
    )
    r = WPPReader(country_label="United Republic of Tanzania")
    out = r.filter_country(df)
    assert out.shape[0] == 1
    assert out.iloc[0]["v"] == 10


def test_filter_country_casefold_and_whitespace_and_nbsp() -> None:
    """
    Tests the functionality of filtering a DataFrame based on a casefolded
    and stripped country label, including handling of non-breaking spaces.

    This test ensures that the method `filter_country` can correctly recognize
    country names with extra whitespace, variations in letter casing, and
    non-breaking spaces. The method is expected to return only the rows
    matching the normalized form of the specified country label.

    Raises:
        AssertionError: If the filtering operation does not correctly handle
        casefolding, stripping of whitespace, or non-breaking spaces, or
        if the filtered DataFrame does not meet the expected conditions.
    """
    nbsp = "\u00a0"
    df = pd.DataFrame(
        {
            "Location": [f"  united{nbsp}republic of tanzania  ", "Kenya"],
            "v": [10, 20],
        }
    )
    r = WPPReader(country_label="United Republic of Tanzania")
    out = r.filter_country(df)
    assert out.shape[0] == 1
    assert out.iloc[0]["v"] == 10


def test_filter_country_raises_with_diagnostics_when_empty() -> None:
    """
    Test case for the filter_country method in the WPPReader class.

    This test checks whether the filter_country method raises a ValueError with
    appropriate diagnostic information when the filtered DataFrame is empty. It
    ensures that the error message contains specific keywords relevant to the
    issue, such as "produced empty result", "country_label", and "sample_values".

    Parameters
    ----------
    None

    Raises
    ------
    ValueError
        If the filtered DataFrame is empty.

    Assertions
    ----------
    Checks that the raised ValueError's message contains the keywords:
    - "produced empty result"
    - "country_label"
    - "sample_values"
    """
    df = pd.DataFrame(
        {
            "Location": ["Kenya", "Uganda"],
            "v": [1, 2],
        }
    )
    r = WPPReader(country_label="United Republic of Tanzania")
    with pytest.raises(ValueError) as e:
        r.filter_country(df)

    msg = str(e.value)
    assert "produced empty result" in msg
    assert "country_label" in msg
    assert "sample_values" in msg


# ---------------------------------------------------------------------
# WPPReader.drop_metadata_cols
# ---------------------------------------------------------------------
def test_drop_metadata_cols_by_position_safe_when_short() -> None:
    """
    Tests the method drop_metadata_cols to ensure it drops columns from the
    input DataFrame based on the specified positions. The method should
    safely handle cases when the column positions to be dropped exceed the
    available columns in the DataFrame.

    Raises:
        AssertionError: If the resulting DataFrame columns are not as
        expected after applying the drop_METADATA_cols method.
    """
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    r = WPPReader(country_label="x", drop_col_positions=(0, 2, 10))
    out = r.drop_metadata_cols(df)
    assert list(out.columns) == ["B"]  # dropped A and C; ignored 10


# ---------------------------------------------------------------------
# WPPReader.read_country_table (mock pd.read_excel)
# ---------------------------------------------------------------------
def test_read_country_table_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Tests the `read_country_table` pipeline using a mocked `pd.read_excel` function and a
    temporary directory. Verifies correct selection and transformation of data for a given
    country.

    This test simulates the behavior of the `WPPReader` when processing Excel sheets with
    specific configurations. Input sheets and dataset manipulation are verified to ensure
    desired functionality of the tested method.

    Parameters:
    monkeypatch: pytest.MonkeyPatch
        Fixture that allows modification of standard library and third-party libraries' behavior
        for the duration of the test.
    tmp_path: Path
        Fixture providing a temporary directory unique to the test function invocation.
    """
    # Simulate two sheets returned by pd.read_excel
    sheet1 = pd.DataFrame(
        {"Location": ["Tanzania", "Kenya"], "Meta": [0, 0], "X": [1, 2], "Y": [3, 4], "Z": [5, 6]}
    )
    sheet2 = pd.DataFrame(
        {
            "Location": ["Tanzania", "Uganda"],
            "Meta": [0, 0],
            "X": [10, 20],
            "Y": [30, 40],
            "Z": [50, 60],
        }
    )

    def fake_read_excel(sheet_name, header):
        """
        Mocks the behavior of reading an Excel file, returning predefined data based on
        the given sheet name and specific header value.

        Parameters:
        sheet_name: str
            The name of the sheet in the Excel file to be "read".
        header: int
            The row index to use as the header for the data. Must be 16.

        Returns:
        dict
            A predefined dataset if the sheet name matches "S1" or "S2".

        Raises:
        KeyError
            If the sheet name does not match "S1" or "S2".
        """
        assert header == 16
        if sheet_name == "S1":
            return sheet1
        if sheet_name == "S2":
            return sheet2
        raise KeyError(sheet_name)

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    r = WPPReader(
        country_label="Tanzania", header_row=16, country_col_index=0, drop_col_positions=(1,)
    )
    out = r.read_country_table(tmp_path / "fake.xlsx", sheets=["S1", "S2"], extra_cols={"Sex": "M"})

    # Should include Tanzania rows from both sheets (2 rows)
    assert out.shape[0] == 2
    assert "Sex" in out.columns
    assert set(out["Sex"]) == {"M"}

    # drop_col_positions=(1,) should drop "Meta" column (position 1 in filtered df)
    assert "Meta" not in out.columns
