"""
Unit test for testing the DHSBuilder functionality and its outputs.

The function tests whether the DHSBuilder reads the provided input(s),
produces the expected outputs (e.g., CSV files), and generates the related
resource manifest. It validates key integration aspects like Excel file reading,
correct handling of data inputs, and outputs being written to disk.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.components.demography.dhs import DHSBuilder
from pipeline.components.resource_builder import BuildContext


def test_dhs_builder_writes_expected_outputs_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    This function tests the DHSBuilder functionality, particularly its capability to
    process specific input configurations, generate the expected artifacts, and write
    the required files to the output directories. The test simulates the input
    environment, mocks Excel file reading behavior, and validates the outputs.

    This test includes the creation of input directories and files, patching the
    behavior of pandas' `read_excel` function to provide mock DataFrames, and asserting
    that the generated output matches the expected structure and file names.

    Args:
        tmp_path (Path): Temporary directory provided by pytest for test file I/O.
        monkeypatch (pytest.MonkeyPatch): pytest fixture used to dynamically modify
        attributes such as replacing `pd.read_excel`.

    Raises:
        AssertionError: Raised if preconditions are not met (e.g., unexpected paths or
        sheet names), or if the generated outputs deviate from the test expectations.

    Returns:
        None
    """
    raw_dir = tmp_path / "inputs" / "demography"
    raw_dir.mkdir(parents=True)
    dhs_path = raw_dir / "dhs" / "dhs_data.xlsx"
    dhs_path.parent.mkdir(parents=True)
    dhs_path.write_bytes(b"fake-xlsx")

    resources_dir = tmp_path / "outputs" / "resources"

    cfg: dict[str, Any] = {
        "country_code": "tz",
        "paths": {"input_dir": str(raw_dir), "resources_dir": str(resources_dir)},
        "dhs": {
            "file": str(dhs_path),
            "sheet_asfr": "ASFR",
            "sheet_u5": "UNDER_5_MORT",
            "u5_header": 1,
        },
    }

    # Patch pd.read_excel:
    # - ASFR sheet: first col is label, rest numeric-ish per 1000
    asfr_df = pd.DataFrame(
        {
            "Age_Grp": ["15-19", "20-24"],
            "2010-2014": [120, 200],
            "2015-2019": [110, 190],
        }
    )

    # The builder calls read_dhs_u5_table() which calls pd.read_excel(sheet_name=UNDER_5_MORT)
    # We return a raw table with Year/Est/Lo/Hi already parseable.
    u5_raw_df = pd.DataFrame(
        {
            "Year": [2010, 2011, 2012],
            "Est": [60, 58, 55],
            "Lo": [50, 49, 47],
            "Hi": [70, 68, 64],
        }
    )

    def fake_read_excel(path, sheet_name=None, header=None):
        """
        Simulates reading an Excel file and returning a DataFrame based on the sheet name.

        This function is primarily used for testing scenarios where Excel files
        are expected to have certain predefined sheets. Depending on the sheet name
        provided, it will return a mocked DataFrame that mirrors the expected data.
        If the sheet name is not recognized, an assertion error is raised.

        Args:
            path: The file path to the Excel file as a string or Path-like object.
            sheet_name: Name of the sheet to be read from the Excel file, or None.
            header: Row number(s) to use as the column names, or None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If the file path does not match a predefined condition
                            or an unexpected sheet name is provided.

        Returns:
            A copy of the mocked DataFrame corresponding to the specified sheet name.
        """
        assert str(path) == str(dhs_path)
        if sheet_name == "ASFR":
            assert header is None
            return asfr_df.copy()
        if sheet_name == "UNDER_5_MORT":
            # header argument is allowed; ignore for this test
            assert header == 1
            return u5_raw_df.copy()
        raise AssertionError(f"Unexpected sheet_name={sheet_name}")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    ctx = BuildContext(
        cfg=cfg,
        country="tz",
        raw_dir=raw_dir,
        resources_dir=resources_dir,
        component="demography",
    )

    artifacts = DHSBuilder(ctx).run()

    names = {a.name for a in artifacts}
    assert "ResourceFile_ASFR_DHS.csv" in names
    assert "ResourceFile_Under_Five_Mortality_DHS.csv" in names

    # files exist
    assert (ctx.output_dir / "ResourceFile_ASFR_DHS.csv").exists()
    assert (ctx.output_dir / "ResourceFile_Under_Five_Mortality_DHS.csv").exists()

    # quick sanity: ASFR scaled per-woman (<= 2 check is in validate)
    asfr_out = pd.read_csv(ctx.output_dir / "ResourceFile_ASFR_DHS.csv")
    assert not asfr_out.empty

    u5_out = pd.read_csv(ctx.output_dir / "ResourceFile_Under_Five_Mortality_DHS.csv")
    assert list(u5_out.columns) == ["Year", "Est", "Lo", "Hi"]
    assert not u5_out.empty
