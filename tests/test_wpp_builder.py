"""
This module provides utilities for generating demographic DataFrames and testing the
functionality of the World Population Prospects (WPP) Builder. It includes functions
that create mock DataFrames to simulate demographic data such as population age groups,
births, deaths, sex ratios, and age-specific fertility rates (ASFR). Additionally,
it provides a test to validate the output and behavior of the WPPBuilder.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.components.demography import wpp as wpp_mod
from pipeline.components.demography.wpp import WPPBuilder
from pipeline.components.resource_builder import BuildContext


def _make_pop_agegrp_df() -> pd.DataFrame:
    """
    Creates a DataFrame with demographic population data for age groups.

    This utility function generates a mock DataFrame with predetermined
    columns needed for population estimations grouped by age. The DataFrame
    contains a single row with specific default values.

    Returns:
        pd.DataFrame: A DataFrame with 23 columns, including demographic
        variants, year, and 21 age group columns filled with example data.
    """
    # Must have columns such that build() does: ests.columns[2:23] scaling
    # => need at least 23 columns: Variant, Year, then 21 age group cols.
    cols = ["Variant", "Year"] + [f"age{i}" for i in range(21)]
    row = ["Estimates", 2010] + [1] * 21
    return pd.DataFrame([row], columns=cols)


def _make_births_df() -> pd.DataFrame:
    """
    Creates a pandas DataFrame containing birth data in a "wide" table format.

    This function generates a pandas DataFrame with pre-defined data in a wide format,
    which includes a "Variant" column and a range-based period column. It provides a
    starting structure for handling births data.

    Returns:
        pd.DataFrame: A pandas DataFrame structured with Variant and period columns.
    """
    # read_country_table returns a "wide" table with Variant + period columns
    return pd.DataFrame(
        {
            "Variant": ["Estimates"],
            "2010-2015": [10.0],
        }
    )


def _make_sex_ratio_df() -> pd.DataFrame:
    """
    Creates a DataFrame containing sex ratio data for a specific variant and time period.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with columns 'Variant' and '2010-2015'. The 'Variant'
        column contains the description of the demographic projection variant,
        and the '2010-2015' column contains the corresponding sex ratio value.
    """
    return pd.DataFrame(
        {
            "Variant": ["Medium variant"],
            "2010-2015": [1.05],
        }
    )


def _make_asfr_df() -> pd.DataFrame:
    """
    Creates a DataFrame representing age-specific fertility rates (ASFR) with predefined
    columns and data. Intended for internal use.

    Returns:
        pd.DataFrame: A DataFrame with predefined ASFR data, including columns for
        variant, period, and age-specific fertility rates (a0 to a6).
    """
    # build() does asfr2.columns[2:9] /1000 => needs at least 9 columns
    cols = ["Variant", "Period"] + [f"a{i}" for i in range(7)]
    row = ["Estimates", "2010-2015"] + [100] * 7
    return pd.DataFrame([row], columns=cols)


def _make_deaths_df(extra_sex: str) -> pd.DataFrame:
    """
    Generates a DataFrame for death statistics with predefined columns filled with estimates.

    The function creates a DataFrame with a fixed structure of 22 columns: two identifying
    columns followed by a sequence of 20 additional columns with numeric placeholder values.
    An additional "Sex" column is appended to indicate the gender associated with the data.
    This is primarily used for building a foundational structure for death statistics data.

    Args:
        extra_sex: A string representing the gender to be added in the "Sex" column.

    Returns:
        pd.DataFrame: A DataFrame containing predefined columns filled with placeholder values
        and an appended "Sex" column.
    """
    # build() scales d.columns[2:22] => need 22 columns.
    cols = ["Variant", "Period"] + [f"d{i}" for i in range(20)]
    df = pd.DataFrame([["Estimates", "2010-2015"] + [1] * 20], columns=cols)
    df["Sex"] = extra_sex
    return df


def test_wpp_builder_writes_expected_outputs_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Tests the WPPBuilder to ensure it writes expected outputs and a manifest file.

    This test validates the functionality of the WPPBuilder by setting up a simulated
    environment with temporary directories and files, mocking necessary methods,
    and verifying the generated outputs.

    Arguments:
    tmp_path : Path
        A pytest fixture providing a temporary directory unique for the test invocation.
    monkeypatch : pytest.MonkeyPatch
        A pytest fixture that allows for dynamic modification or replacement of attributes
        and functions during a test.

    Raises:
    AssertionError
        If any of the expected outputs or manifest file is missing from the generated
        artifacts, or if specific schemas are not properly included.
    """
    raw_dir = tmp_path / "inputs" / "demography"
    raw_dir.mkdir(parents=True)
    resources_dir = tmp_path / "outputs" / "resources"

    # Create fake input files just so preflight() can pass existence checks
    def touch(rel: str) -> str:
        """
        Creates a file with predefined content while ensuring the directory structure exists.

        This function takes a relative path, creates any necessary parent directories,
        and writes a small placeholder content into the specified file.

        Arguments:
            rel: A string representing the relative path where the file will be created.

        Returns:
            str: The absolute path of the newly created file.
        """
        p = raw_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    cfg: dict[str, Any] = {
        "country_code": "tz",
        "paths": {"input_dir": str(raw_dir), "resources_dir": str(resources_dir)},
        "wpp": {
            "country_label": "United Republic of Tanzania",
            "header_row": 16,
            "country_col_index": 2,
            "enable_annual_pop": False,  # avoid census dependency in this unit test
            "pop_agegrp_male": touch("wpp/pop_agegrp_male.xlsx"),
            "pop_agegrp_female": touch("wpp/pop_agegrp_female.xlsx"),
            "pop_agegrp_sheets": ["ESTIMATES"],
            "pop_agegrp_multiplier": 1000,
            "total_births_file": touch("wpp/total_births.xlsx"),
            "sex_ratio_file": touch("wpp/sex_ratio.xlsx"),
            "asfr_file": touch("wpp/asfr.xlsx"),
            "fert_sheets_all": ["ESTIMATES"],
            "fert_sheets_est_med": ["ESTIMATES"],
            "deaths_male_file": touch("wpp/deaths_m.xlsx"),
            "deaths_female_file": touch("wpp/deaths_f.xlsx"),
            "deaths_sheets": ["ESTIMATES"],
            "deaths_multiplier": 1000,
            "lifetable_male_file": touch("wpp/life_m.xlsx"),
            "lifetable_female_file": touch("wpp/life_f.xlsx"),
            "lifetable_sheets": ["ESTIMATES"],
            "lifetable_usecols": "B,C,H,I,J,K",
        },
        "model": {"max_age": 5},
        "census": {"year": 2012},
    }

    # Monkeypatch WPPReader.read_country_table to return our tiny frames by file_path key
    def fake_read_country_table(file_path, extra_cols=None):
        """
        Reads data from a simulated country-specific table based on the given file path.

        This function returns a DataFrame constructed by helper functions depending on the
        filename. It simulates reading structured data relevant to population, births,
        deaths, and ratios. If additional columns are specified, they are added to the
        resultant DataFrame. An error is raised if the provided file path does not match
        any expected filenames.

        Parameters:
        file_path : str
            The file path string used to determine what type of data to simulate.
        sheets : Any
            Placeholder parameter for sheet-related information.
        extra_cols : dict, optional
            A dictionary containing key-value pairs to add additional columns to the
            DataFrame, if specified.
        **kwargs : Any
            Additional keyword arguments, not used in the current implementation.

        Returns:
        pandas.DataFrame
            A simulated DataFrame specific to the identified file type determined by
            `file_path`.

        Raises:
        AssertionError
            If the `file_path` doesn't match any of the expected file patterns.
        """
        fp = str(file_path)
        if fp.endswith("pop_agegrp_male.xlsx") or fp.endswith("pop_agegrp_female.xlsx"):
            df = _make_pop_agegrp_df()
            if extra_cols:
                for k, v in extra_cols.items():
                    df[k] = v
            return df

        if fp.endswith("total_births.xlsx"):
            return _make_births_df()

        if fp.endswith("sex_ratio.xlsx"):
            return _make_sex_ratio_df()

        if fp.endswith("asfr.xlsx"):
            return _make_asfr_df()

        if fp.endswith("deaths_m.xlsx"):
            return _make_deaths_df("M")

        if fp.endswith("deaths_f.xlsx"):
            return _make_deaths_df("F")

        raise AssertionError(f"Unexpected file_path: {fp}")

    monkeypatch.setattr(wpp_mod.WPPReader, "read_country_table", fake_read_country_table)

    # Monkeypatch lifetable outputs to avoid excel parsing complexity
    lt_out = pd.DataFrame(
        {
            "Variant": ["WPP_Estimates"],
            "Period": ["2010-2014"],
            "Sex": ["M"],
            "Age_Grp": ["0-4"],
            "death_rate": [0.01],
        }
    )
    expanded = pd.DataFrame(
        {
            "fallbackyear": [2010],
            "sex": ["M"],
            "age_years": [0],
            "death_rate": [0.01],
        }
    )

    def fake_lifetable_to_death_rates():  # noqa: ARG001
        """Return a deterministic lifetable output for unit testing."""
        return lt_out.copy()

    def fake_expand_death_rates():  # noqa: ARG001
        """Return a deterministic expanded death-rate table for unit testing."""
        return expanded.copy()

    monkeypatch.setattr(wpp_mod, "_lifetable_to_death_rates", fake_lifetable_to_death_rates)
    monkeypatch.setattr(wpp_mod, "_expand_death_rates", fake_expand_death_rates)

    ctx = BuildContext(
        cfg=cfg,
        country="mw",
        raw_dir=raw_dir,
        resources_dir=resources_dir,
        component="demography",
    )

    artifacts = WPPBuilder(ctx).run()

    for expected in WPPBuilder.EXPECTED_OUTPUTS:
        assert expected in {a.name for a in artifacts}
        assert (ctx.output_dir / expected).exists()

    assert (ctx.output_dir / "resource_manifest.json").exists()

    # Schema sanity: Pop_WPP must include Count
    pop = pd.read_csv(ctx.output_dir / "ResourceFile_Pop_WPP.csv")
    assert "Count" in pop.columns
    assert not pop.empty
