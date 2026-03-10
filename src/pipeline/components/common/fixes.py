"""
Utility functions and classes for processing and reading WPP (World Population
Prospects) data files.

This module contains functions and a dataclass for handling and processing
WPP data, including reading Excel sheets, formatting WPP date periods,
filtering data by country, and managing metadata columns.
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
    _country_col_candidates: Sequence[str] = (
        "Location",
        "Country",
        "Area",
        "Region, subregion, country or area",
    )

    def read_concat(
        self, file_path: str | Path, sheets: Iterable[str], **read_excel_kwargs
    ) -> pd.DataFrame:
        """
        Reads and concatenates data from multiple sheets in an Excel file into a single
        Pandas DataFrame. Each sheet specified by the user is read into a separate
        DataFrame and then concatenated along the row axis. Allows additional settings
        to be passed for reading the Excel sheets.

        Parameters:
            file_path: str | Path
                The path to the Excel file to read the sheets from.
            sheets: Iterable[str]
                A collection containing the names of the sheets to read from the Excel file.
            **read_excel_kwargs
                Additional keyword arguments passed to the pandas.read_excel function.

        Returns:
            pd.DataFrame
                A concatenated DataFrame containing data from all the specified sheets.
        """
        file_path = str(file_path)
        frames = [
            pd.read_excel(file_path, sheet_name=s, header=self.header_row, **read_excel_kwargs)
            for s in sheets
        ]
        return pd.concat(frames, sort=False, ignore_index=True)

    def country_col(self, df: pd.DataFrame) -> str:
        """
        Finds the name of the column that represents the country in the given DataFrame.

        The method checks for the presence of specific column names (as specified in
        the internal list of country column candidates) in the DataFrame. If a match is
        found, it returns the name of the matched column. If no match is found, it
        falls back to selecting a column using an index value.

        Parameters:
        df (pd.DataFrame): The input DataFrame from which the country column is to be
        determined.

        Returns:
        str: The name of the country column if found; otherwise, the name of the
        column determined by the fallback index.
        """
        for name in self._country_col_candidates:
            if name in df.columns:
                return name
        # fallback to index (original behavior)
        return df.columns[self.country_col_index]

    @staticmethod
    def _norm_str(s: pd.Series) -> pd.Series:
        """
        Normalizes a Pandas Series by converting to strings, replacing non-breaking
        spaces with regular spaces, and stripping leading and trailing whitespace.

        Parameters:
        s (pd.Series): The Pandas Series to normalize.

        Returns:
        pd.Series: The normalized Pandas Series.
        """
        return s.astype(str).str.replace("\u00a0", " ", regex=False).str.strip()  # NBSP

    def filter_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a DataFrame to include only rows matching a specified country label.

        The method identifies the column containing country information and normalizes
        its string values for comparison with the provided country label. If no rows
        match the country label, an error is raised, providing diagnostic information
        for corrective actions.

        Parameters:
        df: pd.DataFrame
            The input DataFrame to filter.

        Returns:
        pd.DataFrame
            A new DataFrame containing only rows where the country label matches.

        Raises:
        ValueError
            If the resulting DataFrame is empty, indicating no matches were found for
            the provided country label.
        """
        ccol = self.country_col(df)

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
        """
        Drops specified columns from a pandas DataFrame based on predefined column positions.

        This method identifies column names based on their positions in the DataFrame
        and removes those columns, if the positions fall within the valid range.

        Parameters:
        df : pd.DataFrame
            The pandas DataFrame from which columns are to be dropped.

        Returns:
        pd.DataFrame
            A new DataFrame with the specified metadata columns removed.
        """
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
        """
        Reads and processes data from an Excel file containing country information. This function
        handles concatenation of sheets, filtering of country-specific data, and dropping of
        metadata columns. Additionally, it allows for extra columns to be added to the resulting
        DataFrame, based on provided key-value pairs.

        Args:
            file_path: The path to the Excel file containing the data.
                       Can be a str or a Path object.
            sheets: An iterable containing the names of the sheets to be read and concatenated.
            extra_cols: A dictionary of additional column names and their corresponding values
                to be added to the resulting DataFrame. Defaults to None.
            **read_excel_kwargs: Additional keyword arguments to be passed to the function that
                reads Excel data.

        Returns:
            A pandas DataFrame containing the processed country data.
        """
        df = self.read_concat(file_path, sheets, **read_excel_kwargs)
        df = self.filter_country(df)
        df = self.drop_metadata_cols(df)

        if extra_cols:
            for k, v in extra_cols.items():
                df[k] = v

        return df


def _norm_name(x: object) -> str:
    """Normalize a label for robust matching: NBSP->space, trim,
    collapse whitespace, case-sensitive."""
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


def validate_patches_df(patches_df: pd.DataFrame) -> None:
    """
    Validates the structure of the patches_df DataFrame to ensure it contains the
    expected columns.

    Ensures that the DataFrame has all required columns, and if any are missing,
    an error is raised indicating the missing columns.

    Parameters
    ----------
    patches_df : pd.DataFrame
        The DataFrame to validate. It must include the columns 'sheet', 'row_label',
        'col_label', and 'value'.

    Raises
    ------
    KeyError
        If any of the required columns ('sheet', 'row_label', 'col_label', 'value')
        are missing in the provided DataFrame.
    """
    required = {"sheet", "row_label", "col_label", "value"}
    missing = required.difference(patches_df.columns)
    if missing:
        raise KeyError(
            f"patches_df must have columns {required}. Missing: {sorted(missing)}. "
            f"Found: {list(patches_df.columns)}"
        )


def filter_patches_for_sheet(patches_df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """
    Filters a DataFrame of patches to include only rows matching a specific sheet identifier.

    This function ensures that both the "sheet" column in the DataFrame and the given
    sheet identifier are treated as strings for comparison purposes, avoiding unexpected
    behaviors with numeric sheet IDs often encountered in Excel files.

    Parameters:
        patches_df (pd.DataFrame): A DataFrame containing patch data, including a
                                   column named "sheet" for sheet identifiers.
        sheet (str): The identifier of the sheet to filter patches for.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows where the "sheet" column
                      matches the given sheet identifier.
    """
    # normalize both sides to string (avoids Excel numeric sheet IDs surprises)
    return patches_df.loc[patches_df["sheet"].astype(str) == str(sheet)].copy()


def build_norm_lookup(keys) -> dict[str, object]:
    """
    Creates a normalized lookup dictionary mapping normalized names to original keys.

    This function uses a normalization function on the given keys and keeps the
    first occurrence of each normalized name as the associated key in the resulting
    dictionary.

    Parameters:
        keys (Iterable): The collection of keys to process.

    Returns:
        dict[str, object]: A dictionary where the keys are normalized names
        and the values are the original keys.
    """
    # first occurrence wins
    norm_to_key: dict[str, object] = {}
    for k in keys:
        norm_to_key.setdefault(_norm_name(k), k)
    return norm_to_key


def resolve_key(
    lookup: dict[str, object],
    label_norm: str,
    *,
    axis_name: str,
    sheet: str,
    strict: bool,
) -> object | None:
    """
    Resolves a key from a given lookup dictionary based on a normalized label. If the
    key is not found and strict mode is enabled, a KeyError is raised.

    Parameters:
        lookup (dict[str, object]): A dictionary where keys are normalized labels and
            values are the associated objects.
        label_norm (str): The normalized label used to search for the key in the
            lookup dictionary.
        axis_name (str): The name of the axis being searched, used for error messages.
        sheet (str): The name of the sheet or context for the key resolution, used
            in error messages.
        strict (bool): Indicates if a KeyError should be raised when no match is found.

    Returns:
        object | None: The object associated with the given normalized label if found.
            Returns None if no match is found and strict mode is disabled.

    Raises:
        KeyError: If strict mode is enabled and the key is not found.
    """
    key = lookup.get(label_norm)
    if key is None and strict:
        raise KeyError(f"[{sheet}] Patch {axis_name} not found: {label_norm!r}")
    return key


def coerce_patch_value(raw_val: Any) -> Any:
    """
    Coerces a raw value into a patched value suitable for processing.
    Converts blank, empty or NaN values to None. Non-empty values
    are returned unmodified. This ensures uniform representation of
    missing or empty input data.

    Parameters:
    raw_val : Any
        The raw value to be processed.

    Returns:
    Any
        Returns None for blank/NaN values, otherwise returns the
        original value without modification.
    """
    # blank/NaN -> None, else passthrough
    print(raw_val)
    if pd.isna(raw_val) or str(raw_val).strip() == "":
        return None
    return raw_val


def apply_cell_patches(
    df: pd.DataFrame,
    patches_df: pd.DataFrame,
    *,
    sheet: str,
    index_is_labels: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Applies a series of cell patches to the specified DataFrame, modifying its content.

    Patches are defined in the `patches_df` DataFrame, which contains instructions for altering
    specific cells in the `df` DataFrame. This function requires the patches to be associated
    with a specific sheet name and supports optional constraints indicated through the `strict`
    parameter.

    Parameters:
        df (pd.DataFrame): The DataFrame to which the patches will be applied.
        patches_df (pd.DataFrame): The DataFrame containing patching instructions with row and
        column labels and corresponding new values.
        sheet (str): The name of the sheet to which the patches should apply. Only patches
        for this specific sheet will be processed.
        index_is_labels (bool): A boolean determining whether to use labels for indexing rows.
        Currently, only True is supported. Defaults to True.
        strict (bool): A boolean indicating whether to enforce strict matching when resolving
        row and column keys from patches. Defaults to True.

    Returns:
        pd.DataFrame: A new DataFrame with the specified patches applied.

    Raises:
        NotImplementedError: If `index_is_labels` is set to False. This feature is not implemented
        in the current helper.
    """
    validate_patches_df(patches_df)

    if not index_is_labels:
        raise NotImplementedError("index_is_labels=False not implemented in this helper.")

    patches = filter_patches_for_sheet(patches_df, sheet)
    if patches.empty:
        return df.copy()

    out = df.copy()
    idx_lookup = build_norm_lookup(out.index)
    col_lookup = build_norm_lookup(out.columns)

    for _, patch in patches.iterrows():
        row_norm = _norm_name(patch["row_label"])
        col_norm = _norm_name(patch["col_label"])

        idx_key = resolve_key(
            idx_lookup, row_norm, axis_name="row_label", sheet=sheet, strict=strict
        )
        if idx_key is None:
            continue

        col_key = resolve_key(
            col_lookup, col_norm, axis_name="col_label", sheet=sheet, strict=strict
        )
        if col_key is None:
            continue

        out.loc[idx_key, col_key] = coerce_patch_value(patch["value"])

    return out


def return_wpp_columns() -> list[str]:
    """
    Returns the list of column names used in the WPP dataset.

    This function provides a predefined list representing the column headers for
    the World Population Prospects (WPP) dataset. It ensures that any data
    processed or analyzed corresponds to the expected structure.

    Returns:
        list[str]: A list of column names as strings.
    """
    return [
        "Variant",
        "District",
        "District_Num",
        "Region",
        "Year",
        "Period",
        "Age_Grp",
        "Sex",
        "Count",
    ]
