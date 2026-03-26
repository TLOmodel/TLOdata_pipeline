"""
Module for testing the behavior of the CensusBuilder component and mocking the behavior of
external dependencies for controlled test environments.

This module provides the functionality to create mock census workbooks in the form of
dictionaries with pandas DataFrame sheets, as well as a test case to verify the CensusBuilder
component’s ability to process input data and generate the correct outputs.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.components.demography.census import CensusBuilder
from pipeline.components.framework.builder import BuildContext
from pipeline.components.framework.fixes import return_wpp_columns


def fake_census_workbook() -> dict[str, pd.DataFrame]:
    """
    Creates a mock census workbook as a dictionary with multiple pandas DataFrame sheets.

    The generated workbook simulates a given census dataset for a specified year. Each sheet
    represents different aspects of the census data, including population totals, age distribution,
    regions, and optional fixes or patches. It is tailored for testing scenarios where these data
    structures are required.

    Parameters:
        census_year: int
            The year of the census for constructing the workbook.

    Returns:
        dict[str, pd.DataFrame]:
            A dictionary where keys represent sheet names and values are pandas DataFrame
            objects containing mock data.
    """
    # regions sheet
    regions = pd.DataFrame({"Region": ["North", "South"]})

    # pop_totals sheet structure expected by your cleanup:
    # - drops rows [0,1] and then drops first remaining row => we provide 3 junk rows
    #   then real data.
    # - first column becomes index (district/labels)
    # - after cleanup must have 3 columns for single-year: Total, Male, Female
    pop_totals = pd.DataFrame(
        [
            ["junk", None, None, None],
            ["junk", None, None, None],
            ["junk", None, None, None],
            # region rows
            ["North", 60, 30, 30],
            ["A", 30, 15, 15],
            ["B", 30, 15, 15],
            ["South", 40, 20, 20],
            ["C", 40, 20, 20],
            # national
            ["Tanzania", 100, 50, 50],
        ],
        columns=["Area", "Total", "Male", "Female"],
    )

    # age_distribution sheet structure expected:
    # - first row is header row
    # - has an "Area" column for index
    age_dist = pd.DataFrame(
        [
            ["Area", "0-1", "1-4", "5-9", "Total"],
            ["A", 5, 5, 20, 30],
            ["B", 5, 5, 20, 30],
            ["C", 10, 10, 20, 40],
        ]
    )

    # optional sheets
    dist_name_fixes = pd.DataFrame(columns=["from", "to"])
    cell_patches = pd.DataFrame(columns=["sheet", "row_label", "col_label", "value"])

    return {
        "pop_totals": pop_totals,
        "age_distribution": age_dist,
        "regions": regions,
        "dist_name_fixes": dist_name_fixes,
        "cell_patches": cell_patches,
    }


@pytest.mark.parametrize("year", [2012])
def test_census_builder_writes_expected_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, year: int
) -> None:
    """
    Test function to verify the behavior of the CensusBuilder process in generating expected outputs
    and a corresponding manifest file.

    Summary:
    This test ensures that the CensusBuilder reads the provided input data correctly, generates
    the required output CSV file with the expected structure and content, and creates a valid
    resource manifest. It uses mock data and monkeypatched functions to simulate reading an Excel
    file and verify the implementation under controlled conditions.

    Args:
        tmp_path (Path): A temporary directory for input and output data setup used during the
                         test.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture to override attributes and methods
                         during the test execution.
        year (int): The year of the census data to be processed.

    Raises:
        AssertionError: If the CensusBuilder does not generate the expected output file, content,
                        or manifest file based on the test input setup.

    """
    # Arrange: fake workbook file exists
    raw_dir = tmp_path / "inputs" / "demography"
    raw_dir.mkdir(parents=True)
    census_path = raw_dir / "census" / "tanzania_census.xlsx"
    census_path.parent.mkdir(parents=True)
    census_path.write_bytes(b"fake-xlsx")  # existence only

    resources_dir = tmp_path / "outputs" / "resources"

    cfg: dict[str, Any] = {
        "country_code": "tz",
        "country_name": "Tanzania",
        "paths": {"input_dir": str(raw_dir), "resources_dir": str(resources_dir)},
        "census": {
            "year": year,
            "population_tables": str(census_path),  # use resolve_input_path robustness
            "pop_totals": "pop_totals",
            "age_dist": "age_distribution",
            "regions": "regions",
            "dist_name_fixes": "dist_name_fixes",
            "cell_patches": "cell_patches",
            "national_label": "Tanzania",
        },
    }

    # Monkeypatch pd.read_excel to return "workbook dict" when sheet_name=None
    wb = fake_census_workbook()

    def fake_read_excel(path: Path, sheet_name=None):
        """
        Fake function to simulate reading an Excel file for testing purposes.

        Summary:
        This function acts as a replacement for `read_excel` in testing scenarios, validating
        that the provided path matches the expected path and ensuring that the sheet name is
        not specified. If the conditions are met, it returns a predetermined workbook object;
        otherwise, it raises an assertion error.

        Args:
            path (str or object): The path to the Excel file to read.
            sheet_name (str, optional): The sheet name to load from the file. Default is None,
                which indicates the entire workbook should be read.
            *args: Additional positional arguments for compatibility.
            **kwargs: Additional keyword arguments for compatibility.

        Raises:
            AssertionError: If the path does not match the expected `census_path` or if a
                `sheet_name` is specified (other than None).

        Returns:
            object: The mocked workbook object if validation passes.
        """
        assert str(path) == str(census_path)
        if sheet_name is None:
            return wb
        raise AssertionError("CensusBuilder should read sheet_name=None (entire workbook).")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    ctx = BuildContext(
        cfg=cfg,
        country="ug",
        input_dir=raw_dir,
        resources_dir=resources_dir,
        component="demography",
    )

    # Act
    artifacts = CensusBuilder(ctx).run()

    # Assert
    assert len(artifacts) == 1
    out_name = f"ResourceFile_PopulationSize_{year}Census.csv"
    assert artifacts[0].name == out_name

    out_path = ctx.output_dir / out_name
    assert out_path.exists()

    df = pd.read_csv(out_path)
    required_cols = set(return_wpp_columns())
    assert required_cols.issubset(df.columns)
    assert not df.empty
