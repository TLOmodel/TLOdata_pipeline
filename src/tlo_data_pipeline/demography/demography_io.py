#!/usr/bin/env python3
"""
Transforms a wide-format DataFrame into a long-format DataFrame while optionally scaling
selected columns by a multiplier. This function is designed for datasets with age group
columns or similar categorical breakdowns that need reformatting or scaling.

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def ensure_dir(p: Path) -> None:
    """
    Creates a directory and any necessary intermediate directories if they do not
    already exist.

    This function ensures that the directory specified by the Path object is
    created. If the directory or any required parent directories already exist,
    the function does not raise any errors.

    Parameters:
        p (Path): The path of the directory to create.

    Returns:
        None
    """
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
    Reads and processes World Population Prospects (WPP) data from Excel sheets.

    This class is designed to handle operations related to reading and processing WPP data.
    It provides functionality to read and concatenate data from multiple Excel sheets, filter
    records based on a specific country label, and remove metadata columns. Additionally, it
    supports adding extra columns during processing.

    Attributes:
        country_label: The label of the country used for filtering rows in the dataset.
        header_row: The row number to be used as the header while reading Excel files.
        country_col_index: The column index where the country information is located.
        drop_col_positions: A sequence of column indices that need to be dropped from the data.
    """

    country_label: str
    header_row: int = 16
    country_col_index: int = 2
    # columns to drop by positional index (matches your current code)
    drop_col_positions: Sequence[int] = (0, 2, 3, 4, 5, 6)

    def read_concat(
        self, file_path: str | Path, sheets: Iterable[str], **read_excel_kwargs
    ) -> pd.DataFrame:
        """
        Reads multiple sheets from an Excel file and concatenates them into a single DataFrame.

        This method takes the file path of an Excel file and a list of sheet names to read.
        It applies the `pandas.read_excel` function to each specified sheet and combines
        the resulting DataFrames vertically into a single DataFrame using `pandas.concat`.
        Additional keyword arguments for `pandas.read_excel` can be passed
        via `**read_excel_kwargs`.

        Parameters:
            file_path (str | Path): The path to the Excel file to read.
            sheets (Iterable[str]): A collection of sheet names to read from the file.
            **read_excel_kwargs: Additional keyword arguments to pass to `pandas.read_excel`.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated data from the specified sheets.
        """
        file_path = str(file_path)
        frames = [
            pd.read_excel(file_path, sheet_name=s, header=self.header_row, **read_excel_kwargs)
            for s in sheets
        ]
        return pd.concat(frames, sort=False, ignore_index=True)

    def filter_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters rows in the provided DataFrame based on a specific country value from
        a configured column index. Returns a new DataFrame containing only the rows
        where the country column matches the specified label.

        Parameters:
        df: pd.DataFrame
            The input DataFrame from which rows will be filtered.

        Returns:
        pd.DataFrame
            A new DataFrame containing rows where the country column matches the
            specified label. The index of the returned DataFrame is reset.
        """
        # Filter rows where the "country" column (by index) equals configured label
        ccol = df.columns[self.country_col_index]
        out = df.loc[df[ccol] == self.country_label].copy()
        return out.reset_index(drop=True)

    def drop_metadata_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns from the given DataFrame based on pre-configured column positions to drop.

        The method iterates through the pre-defined column positions that need to be dropped,
        validates that the position is within the bounds of the DataFrame's column indices, and
        drops the corresponding columns.

        Parameters:
        df : pd.DataFrame
            The input DataFrame from which specified columns will be removed.

        Returns:
        pd.DataFrame
            A new DataFrame after removing the specified columns.
        """
        cols = list(df.columns)
        drop_cols = [cols[i] for i in self.drop_col_positions if i < len(cols)]
        return df.drop(columns=drop_cols)

    def read_country_table(
        self,
        file_path: str | Path,
        sheets: Iterable[str],
        extra_cols: dict[str, object] | None = None,
        **read_excel_kwargs,
    ) -> pd.DataFrame:
        """
        Reads and processes country-related data from an Excel file with specified sheets,
        adding any extra columns if provided.

        This function consolidates data from multiple sheets of an Excel file into a single
        DataFrame. It filters the data to include only relevant country information, removes
        metadata columns, and appends additional columns if specified.

        Arguments:
            file_path: str | Path
                Path to the Excel file to be read.
            sheets: Iterable[str]
                Collection of sheet names to be processed from the Excel file.
            extra_cols: Optional[dict[str, object]]
                Key-value pairs of column names and their values to be added to the resulting
                DataFrame, if provided.
            **read_excel_kwargs:
                Additional keyword arguments to be passed to the pandas `read_excel` function.

        Returns:
            pd.DataFrame:
                Processed DataFrame containing country-related data.
        """
        df = self.read_concat(file_path, sheets, **read_excel_kwargs)
        df = self.filter_country(df)
        df = self.drop_metadata_cols(df)
        if extra_cols:
            for k, v in extra_cols.items():
                df[k] = v
        return df
