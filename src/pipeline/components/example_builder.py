"""
Skeleton Resourcefile Builder (Contributor Template)
==============================================

Purpose
-------
This file demonstrates the minimum structure required to implement
one formatting module in the pipeline framework.

Every formatting module should normally contain the following methods:

1. __init__()   : initialize builder
2. preflight()  : check files/config before processing
3. load_data()  : read raw input files
4. build()      :    format data and create outputs
5. validate()   :  verify outputs are correct

The framework will handle writing of CSV files

This template produces one dummy CSV so new contributors can run
it immediately and understand the workflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.components.framework.builder import BuildContext, ResourceBuilder


class ExampleBuilder(ResourceBuilder):
    """
    Minimal example builder implementing the standard framework methods.

    Contributors should copy this template when creating new
    data formatting modules.
    """

    outputs = [
        "ResourceFile_Dummy_Example.csv",
    ]

    def __init__(self, ctx: BuildContext, *, dry_run: bool = False):
        """
        Initializes a class instance with the given context and optional dry-run mode.

        Parameters:
            ctx (BuildContext): The build context associated with the initialization.
            dry_run (bool): Indicates whether to run in dry-run mode. Defaults to False.
        """

        super().__init__(ctx, dry_run=dry_run)

        self.raw_df: pd.DataFrame = pd.DataFrame()

    def preflight(self) -> None:
        """
        Run quick checks before processing begins.

        Typical checks:

        • required files exist
        • configuration values present
        • directories accessible
        """

        input_file = Path(self.ctx.input_dir) / "dummy_input.csv"

        if not input_file.exists():
            print("dummy_input.csv not found — using fallback example data")

    def load_data(self) -> None:
        """
        Load raw input data.

        This method should read raw files from disk and store them
        as attributes on the builder instance.

        Example sources:
        • CSV
        • Excel
        """

        input_file = Path(self.ctx.input_dir) / "dummy_input.csv"

        if input_file.exists():
            self.raw_df = pd.read_csv(input_file)

        else:
            # fallback example data
            self.raw_df = pd.DataFrame(
                {
                    "region_name": ["Northern", "Central", "Southern"],
                    "year_value": [2020, 2020, 2020],
                    "raw_count": [1000, 1500, 1200],
                }
            )

    def build(self, raw: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
        """
        Transform raw data into pipeline resource files.

        Steps typically include:

        • cleaning columns
        • reshaping tables
        • renaming fields
        • converting units
        • calculating derived variables
        """

        df = self.raw_df.rename(
            columns={
                "region_name": "Region",
                "year_value": "Year",
                "raw_count": "Count",
            }
        ).copy()

        df = df[["Region", "Year", "Count"]]

        return {
            "ResourceFile_Dummy_Example.csv": df,
        }

    def validate(self, outputs: dict[str, pd.DataFrame]) -> None:
        """
        Validate output tables.

        Common validation checks:

        • table is not empty
        • required columns exist
        • no negative counts
        • valid categorical values
        """

        for name, df in outputs.items():

            if df.empty:
                raise AssertionError(f"{name} is empty")

            required_columns = {"Region", "Year", "Count"}

            if not required_columns.issubset(df.columns):
                raise AssertionError(f"{name} missing columns {required_columns - set(df.columns)}")
