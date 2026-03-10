# tests/test_wpp_builder.py
"""
Unit test suite for the World Population Projection (WPP) builder.

This module includes various test definitions and fixtures to validate the
functionality of the WPP components and configurations. Tests ensure proper
interaction between generated configurations, data structures, and defined
methods within the WPP pipeline. Fixtures provide reusable contexts for
setting up unit test environments.

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pipeline.components.demography.wpp as wpp
from pipeline.components import utils


# ------------------------------------------------------------------------------
# Minimal test context (matches what ResourceBuilder expects)
# ------------------------------------------------------------------------------
@dataclass
class DummyCtx:
    cfg: dict[str, Any]
    input_dir: Path
    raw_dir: Path
    output_dir: Path


def _minimal_cfg(
    *,
    country_label: str = "United Republic of Tanzania",
    max_age: int = 10,  # keep tests small
    census_year: int = 2012,
    enable_annual_pop: bool = True,
    init_population_year: int = 2010,
) -> dict[str, Any]:
    """
    Generates a minimal configuration dictionary for demographic model setup.

    This function creates and returns a configuration dictionary structured
    for use in demographic modeling. The configuration includes key information
    like census year, maximum age, population details, and various file paths
    for data inputs (e.g., fertility rates, mortality rates). The dictionary
    is tailored to support flexible control over settings like annual population
    data inclusion and initial population year. Defaults can be overridden
    through keyword parameters.

    Args:
        country_label (str, optional): Country label used in the configuration.
                                       Defaults to "United Republic of Tanzania".
        max_age (int, optional): Maximum age considered for modeling. Defaults to 10.
        census_year (int, optional): Year of the census, relevant for the model.
                                     Defaults to 2012.
        enable_annual_pop (bool, optional): Flag to enable or disable annual population.
                                            Defaults to True.
        init_population_year (int, optional): Initial year for model population data.
                                              Defaults to 2010.

    Returns:
        dict[str, Any]: Configuration dictionary populated with the provided parameters
                        and structured for demographic modeling.
    """
    return {
        "census": {"year": census_year},
        "model": {"max_age": max_age},
        "wpp": {
            "country_label": country_label,
            "header_row": 16,
            "country_col_index": 2,
            "enable_annual_pop": enable_annual_pop,
            "init_population_year": init_population_year,
            "pop_annual_male_file": "wpp/annual_m.xlsx",
            "pop_annual_female_file": "wpp/annual_f.xlsx",
            "pop_annual_sheets": ["ESTIMATES", "MEDIUM VARIANT"],
            "pop_annual_multiplier": 1000,
            "pop_annual_age_cols_slice_end": 103,
            "pop_agegrp_male": "wpp/agegrp_m.xlsx",
            "pop_agegrp_female": "wpp/agegrp_f.xlsx",
            "pop_agegrp_sheets": ["ESTIMATES", "LOW VARIANT", "MEDIUM VARIANT", "HIGH VARIANT"],
            "pop_agegrp_multiplier": 1000,
            "total_births_file": "wpp/births.xlsx",
            "sex_ratio_file": "wpp/sexratio.xlsx",
            "asfr_file": "wpp/asfr.xlsx",
            "fert_sheets_all": ["ESTIMATES", "LOW VARIANT", "MEDIUM VARIANT", "HIGH VARIANT"],
            "fert_sheets_est_med": ["ESTIMATES", "MEDIUM VARIANT"],
            "deaths_male_file": "wpp/deaths_m.xlsx",
            "deaths_female_file": "wpp/deaths_f.xlsx",
            "deaths_sheets": ["ESTIMATES", "LOW VARIANT", "MEDIUM VARIANT", "HIGH VARIANT"],
            "deaths_multiplier": 1000,
            "lifetable_male_file": "wpp/lt_m.xlsx",
            "lifetable_female_file": "wpp/lt_f.xlsx",
            "lifetable_sheets": ["ESTIMATES", "MEDIUM 2020-2050", "MEDIUM 2050-2100"],
            "lifetable_usecols": "B,C,H,I,J,K",
        },
    }


@pytest.fixture()
def ctx(tmp_path: Path) -> DummyCtx:
    """
    Fixture to provide a test context for unit tests.

    This fixture sets up a temporary file structure required for testing. It creates
    directories for inputs, raw data, and outputs within a given temporary path. These
    directories are prepared to simulate the testing environment. The function sets up
    a `DummyCtx` instance containing configuration data, as well as these paths,
    to be used by test cases.

    Parameters:
    tmp_path: Path
        A temporary directory path provided by pytest to be used in the test setup.

    Returns:
    DummyCtx
        A DummyCtx instance initialized with configuration and directory paths for a
        test context.
    """
    cfg = _minimal_cfg()
    input_dir = tmp_path / "inputs"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return DummyCtx(cfg=cfg, input_dir=input_dir, raw_dir=raw_dir, output_dir=output_dir)


@pytest.fixture()
def cfg(ctx: DummyCtx, monkeypatch: pytest.MonkeyPatch) -> wpp.WPPConfig:
    """
    Fixture for configuring the WPPConfig instance for testing purposes.

    This function sets up a WPPConfig object using a testing context and monkeypatching.
    It modifies the resolution of input file paths within the WPPConfig module by temporarily
    injecting a custom behavior for the `resolve_input_path` function. The function ensures an
    accurate testing configuration based on the provided context and the behavior.

    Parameters:
        ctx (DummyCtx): The mock testing context containing the required attributes for WPPConfig.
        monkeypatch (pytest.MonkeyPatch): A pytest utility that allows temporary patching of
                                          modules or objects for testing purposes.

    Returns:
        wpp.WPPConfig: A configured WPPConfig instance based on the provided testing context.
    """
    monkeypatch.setattr(utils, "resolve_input_path", lambda c, p: Path(c.input_dir) / str(p))
    return wpp.WPPConfig.from_ctx(ctx)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------
# Helpers to craft frames with the right column layout
# ------------------------------------------------------------------------------
def _age_cols(n: int) -> dict[str, list[float]]:
    """
    Generates a dictionary of numeric age-group columns.

    Each column is named using a prefix "A" followed by a zero-padded
    two-digit number, representing the age group index. The values in
    each column are lists containing floating-point numbers representing
    the range of an age group.

    Parameters:
    n: int
        The number of age-group columns to generate. This determines
        the range of indices for naming the columns.

    Returns:
    dict[str, list[float]]
        A dictionary where each key is a string in the format "Axx"
        (e.g., "A00", "A01", ..., "A(n-1)") and each value is a list of
        two floating-point numbers defining the age group range.
    """
    # return n numeric age-group columns named "A00".."A(n-1)"
    return {f"A{i:02d}": [float(i + 1), float(i + 2)] for i in range(n)}


def _mock_pop_agegrp_df() -> pd.DataFrame:
    """
    Mocks a population age group DataFrame for testing purposes.

    This function creates a DataFrame with a specific layout and data, designed for
    testing scenarios where a population distribution across age groups by year,
    variant, and sex is required. The layout includes specific numeric columns that
    are deliberately placed to test slice selection and data manipulation.

    Returns:
        pd.DataFrame: A DataFrame with the following layout:
            - A "Variant" column with population estimate or projection variants.
            - A "Year" column indicating the year of the data.
            - Numeric columns representing demographic data across 23 fields.
            - A "Sex" column indicating gender information.

    Notes:
        The numeric columns are generated dynamically to meet the required structure.
        The "Sex" column is positioned deliberately outside the numeric slice to
        ensure separation for testing purposes.
    """
    # Ensure Sex is NOT inside columns[2:23] => put >= 23 numeric cols before Sex.
    # Layout: [Variant, Year, 23 numeric cols..., Sex]
    df = pd.DataFrame(
        {
            "Variant": ["Estimates", "Medium variant"],
            "Year": [2010, 2010],
            **_age_cols(
                23
            ),  # columns 2..24 (inclusive) are numeric; slice 2:23 hits 21 of them, all numeric
            "Sex": ["M", "F"],  # index is 25, safely outside slice
        }
    )
    return df


def _mock_deaths_df() -> pd.DataFrame:
    """
    Creates a mock DataFrame for deaths data, containing specified columns and data
    to simulate a realistic dataset.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing mock data with the following columns:
        - 'Variant': Indicates the type of data, e.g., 'Estimates'.
        - 'Period': Specifies the time period, e.g., '2010-2015'.
        Additional numeric age columns and a 'Sex' column are included.
    """
    # Ensure Sex is NOT inside columns[2:22] => put >= 22 numeric cols before Sex.
    df = pd.DataFrame(
        {
            "Variant": ["Estimates", "Estimates"],
            "Period": ["2010-2015", "2010-2015"],
            **_age_cols(22),  # numeric age cols
            "Sex": ["M", "F"],  # outside slice
        }
    )
    return df


# ------------------------------------------------------------------------------
# Unit tests: normalization + country filter
# ------------------------------------------------------------------------------
def test_norm_text_series_casefold_and_nbsp() -> None:
    """
    Tests the normalization of a pandas Series containing text with mixed cases,
    leading/trailing whitespace, and non-breaking spaces.

    This function verifies that normalization applied to the text series
    results in uniformly lowercased strings with stripped whitespace,
    including handling of non-breaking spaces.

    Raises
    ------
    AssertionError
        If the normalized output does not match the expected list of strings.
    """
    s = pd.Series(["  Malawi", "MALAWI\u00a0", "malawi "])
    out = wpp._norm_text_series(s)
    assert out.tolist() == ["malawi", "malawi", "malawi"]


def test_filter_country_by_col_matches_case_and_nbsp() -> None:
    """
    Filters a DataFrame by a specific column and matches the given country name, considering
    case sensitivity and non-breaking spaces.

    This function ensures that a row is selected from the DataFrame if its value in the
    specified column matches the given target country name. It accounts for case differences
    and trailing/embedded non-breaking spaces.

    Parameters:
    df : DataFrame
        The input pandas DataFrame to filter.
    col : str
        The name of the column in the DataFrame to check for matching values.
    country_name : str
        The target country name to match against the values in the specified column.

    Raises:
    None

    Returns:
    None
    """
    df = pd.DataFrame({"Loc": ["United Republic of Tanzania\u00a0", "Malawi"], "x": [1, 2]})
    out = wpp._filter_country_by_col(df, "Loc", "United Republic of Tanzania")
    assert len(out) == 1
    assert out.loc[0, "x"] == 1


def test_filter_country_by_col_raises_with_samples() -> None:
    """
    Tests the behavior of the '_filter_country_by_col' function when it encounters
    an empty result based on the given column and value for filtering.

    Ensures that the function raises a ValueError and verifies the
    error message contains the expected substrings.

    Raises:
        ValueError: If the filter operation results in no matching rows based
        on the specified column and value.
    """
    df = pd.DataFrame({"Loc": ["A", "B"], "x": [1, 2]})
    with pytest.raises(ValueError) as e:
        _ = wpp._filter_country_by_col(df, "Loc", "Z")
    msg = str(e.value)
    assert "Country filter returned empty" in msg
    assert "sample_values" in msg


# ------------------------------------------------------------------------------
# Unit tests: births fraction expansion
# ------------------------------------------------------------------------------
def test_expand_frac_births_male_per_year_uses_estimate_over_medium() -> None:
    """
    Tests the function `expand_frac_births_male_per_year` to ensure it selects the
    'Estimate' variant over 'Medium variant' and correctly calculates the male
    birth fraction for the provided years.

    Raises:
        AssertionError: If the function does not correctly calculate the expected
            male birth fraction or does not output the expected years.
    """
    births_df = pd.DataFrame(
        {
            "Variant": ["Estimates", "Medium variant"],
            "Period": ["1950-1954", "1950-1954"],
            "M_to_F_Sex_Ratio": [1.20, 1.10],
        }
    )
    out = wpp.expand_frac_births_male_per_year(births_df, year_lo=1950, year_hi=1952)
    expected = 1.20 / (1.0 + 1.20)
    assert out["Year"].tolist() == [1950, 1951]
    assert np.allclose(out["frac_births_male"].to_numpy(), expected)


def test_expand_frac_births_male_per_year_raises_if_year_not_covered() -> None:
    """
    Tests the behavior of `expand_frac_births_male_per_year` when the specified year
    range is not covered by the provided data.

    Raises
    ------
    ValueError
        If the specified year range is not covered by the `births_df` DataFrame.
    """
    births_df = pd.DataFrame(
        {"Variant": ["Estimates"], "Period": ["1950-1950"], "M_to_F_Sex_Ratio": [1.2]}
    )
    with pytest.raises(ValueError) as e:
        _ = wpp.expand_frac_births_male_per_year(births_df, year_lo=1949, year_hi=1951)
    assert "year not covered" in str(e.value)


# ------------------------------------------------------------------------------
# Unit tests: builders (pop, births, asfr, deaths)
# ------------------------------------------------------------------------------
def test_build_pop_wpp_multiplies_and_melts(cfg: wpp.WPPConfig) -> None:
    """
    Tests the functionality of _build_pop_wpp in the WPP module.

    This test function verifies that the resulting DataFrame from the
    _build_pop_wpp function contains the expected columns and ensures that the
    multiplication factor (1000) is correctly applied when reshaping the data. It
    also checks a specific condition for a melted row to confirm correctness.

    Parameters:
    cfg (wpp.WPPConfig): Configuration object for WPP.

    Raises:
    AssertionError: If the resulting DataFrame does not contain the required
    columns or if the multiplication factor is not correctly applied.
    """
    pop_agegrp = _mock_pop_agegrp_df()
    out = wpp._build_pop_wpp(cfg=cfg, pop_agegrp=pop_agegrp)
    assert set(out.columns) == {"Variant", "Year", "Sex", "Age_Grp", "Count"}

    # pick one melted row and check multiplier (1000)
    row = out[(out["Variant"] == "WPP_Estimates") & (out["Sex"] == "M")]
    assert not row.empty
    assert float(row["Count"].iloc[0]) == pytest.approx(1000.0)


def test_build_births_tables_merges_and_clones_medium_to_high_low(monkeypatch) -> None:
    """
    Tests the functionality of merged and cloned computations for birth tables between different variants.

    This function verifies the correctness of merging total births and sex ratios across multiple
    birth variants (e.g., 'Low', 'Medium', 'High') and using a mock method for expanding the fraction
    of male births over the years. It ensures that all variants and their corresponding total births
    are correctly handled and validates the fractional male births' range of years.

    Parameters:
    monkeypatch (pytest.MonkeyPatch): A pytest monkeypatch object used to override methods during testing.

    Raises:
    AssertionError: If the assertions on the resulting DataFrames (for variants, total births, or fraction
    of male births data) fail.

    """
    # Use 1950-1955 so that AFTER reformat (likely -1 on high) it covers 1954.
    tot_births = pd.DataFrame(
        {
            "Variant": ["Low variant", "Medium variant", "High variant", "Estimates"],
            "1950-1955": [1.0, 2.0, 3.0, 4.0],
        }
    )

    sex_ratio = pd.DataFrame(
        {"Variant": ["Estimates", "Medium variant"], "1950-1955": [1.05, 1.10]}
    )

    def fake_expand_frac_births_male_per_year(births_df, year_lo=1950, year_hi=2100):
        """
        Expands the fraction of male births for each year within a given range.

        This function generates a DataFrame containing the specified range of years
        and assigns a constant fraction of male births for each year. The function
        does not directly use input data but instead creates a static dataset.

        Parameters:
        births_df (pandas.DataFrame): Input DataFrame containing birth data. The
            DataFrame is not utilized in the current implementation.
        year_lo (int): Lower bound of the year range (inclusive). Defaults to 1950.
        year_hi (int): Upper bound of the year range (exclusive). Defaults to 2100.

        Returns:
        pandas.DataFrame: A DataFrame with two columns: "Year" containing the years
            from `year_lo` to `year_hi`, and "frac_births_male" containing the
            fraction of male births, fixed at 0.51 for all years.
        """
        return pd.DataFrame(
            {
                "Year": [1950, 1951, 1952, 1953, 1954],
                "frac_births_male": [0.51, 0.51, 0.51, 0.51, 0.51],
            }
        )

    monkeypatch.setattr(
        wpp,
        "expand_frac_births_male_per_year",
        fake_expand_frac_births_male_per_year,
    )

    births, frac = wpp._build_births_tables(tot_births=tot_births, sex_ratio=sex_ratio)

    assert {"Variant", "Period", "Total_Births", "M_to_F_Sex_Ratio"} <= set(births.columns)
    assert float(births.loc[births["Variant"] == "Estimates", "Total_Births"].iloc[0]) == 4000.0

    # cloned Medium -> High/Low should make merge succeed for all variants present in tot_births
    assert set(births["Variant"].unique()) >= {
        "Low variant",
        "Medium variant",
        "High variant",
        "Estimates",
    }

    assert set(frac.columns) == {"Year", "frac_births_male"}
    assert frac["Year"].min() == 1950
    assert frac["Year"].max() == 1954


def test_build_asfr_scales_per_1000_and_melts() -> None:
    """
    Test function for verifying `wpp._build_asfr`.

    This function tests whether the `_build_asfr` method of the `wpp` module properly scales the
    age-specific fertility rates (ASFR) per 1000 population and reshapes the data into a long format.
    It ensures that the output columns are as expected, the ASFR values are scaled correctly, and
    the "Variant" column values are prefixed appropriately.

    Raises
    ------
    AssertionError
        If the columns in the output DataFrame are not as expected, if any values in the "asfr" column
        are not scaled correctly, or if the "Variant" column does not have the required prefix.
    """
    asfr = pd.DataFrame(
        {
            "Variant": ["Estimates"],
            "Period": ["1950-1955"],
            "15-19": [50.0],
            "20-24": [100.0],
            "25-29": [0.0],
            "30-34": [10.0],
            "35-39": [5.0],
            "40-44": [1.0],
            "45-49": [0.0],
        }
    )
    out = wpp._build_asfr(asfr=asfr)
    assert set(out.columns) == {"Variant", "Period", "Age_Grp", "asfr"}
    v = out.loc[out["Age_Grp"] == "15-19", "asfr"].iloc[0]
    assert float(v) == pytest.approx(0.05)
    assert out["Variant"].iloc[0].startswith("WPP_")


def test_build_deaths_multiplies_and_melts(cfg: wpp.WPPConfig) -> None:
    """
    Tests the `_build_deaths` function to ensure that it correctly processes the deaths DataFrame,
    applies necessary scaling, and outputs the expected melted structure.

    Arguments:
        cfg (wpp.WPPConfig): Configuration object used for processing. Contains
                             necessary settings for scaling and transformation.

    Raises:
        AssertionError: If the output DataFrame does not have the expected columns,
                        or the scaling does not result in the expected values.
    """
    deaths = _mock_deaths_df()
    out = wpp._build_deaths(cfg=cfg, deaths=deaths)
    assert set(out.columns) == {"Variant", "Period", "Sex", "Age_Grp", "Count"}

    # ensure scaling happened
    v = out.loc[out["Age_Grp"] == "A00", "Count"].iloc[0]
    assert float(v) == pytest.approx(1000.0)
    assert out["Variant"].iloc[0].startswith("WPP_")


# ------------------------------------------------------------------------------
# Unit tests: death-rate expansion expects full coverage
# ------------------------------------------------------------------------------
def test_expand_death_rates_creates_row_per_age_sex_period(cfg: wpp.WPPConfig) -> None:
    """
    Tests the function `_expand_death_rates` to verify that it correctly expands death rates
    into separate rows for each age, sex, and period combination. Specifically validates
    that the output contains the required columns, conforms to the expected length, and
    covers the expected range of ages based on the configured maximum age.

    Parameters
    ----------
    cfg : wpp.WPPConfig
        Configuration instance containing the maximum age and other parameters.

    Raises
    ------
    AssertionError
        If the output does not have the expected columns, the expected length, or the
        age range does not conform to the configuration.
    """
    # cfg max_age is 10, so provide "0-4" and "5-9" for BOTH sexes.
    lt_out = pd.DataFrame(
        {
            "Variant": ["WPP_Medium"] * 4,
            "Period": ["2020-2024"] * 4,
            "Sex": ["M", "M", "F", "F"],
            "Age_Grp": ["0-4", "5-9", "0-4", "5-9"],
            "death_rate": [0.1, 0.11, 0.2, 0.21],
        }
    )
    expanded = wpp._expand_death_rates(cfg=cfg, lt_out=lt_out)
    assert set(expanded.columns) == {"fallbackyear", "sex", "age_years", "death_rate"}
    assert len(expanded) == cfg.extras_dict["max_age"] * 2
    assert expanded["age_years"].min() == 0
    assert expanded["age_years"].max() == cfg.extras_dict["max_age"] - 1


# ------------------------------------------------------------------------------
# Init population distribution tests
# ------------------------------------------------------------------------------
def test_build_init_population_by_district_sums_match() -> None:
    """
    Tests that an initial population distribution across districts is constructed correctly.

    This test verifies if the population distribution calculated for the districts
    matches the provided annual population data and respects the district breakdown ratios.
    It ensures that:
    1. The output columns contain the expected fields, including district-related and
       demographic categories.
    2. The total population count across the districts aligns with the input population
       sums, maintaining consistency.

    Raises
    ------
    AssertionError
        If the calculated output does not meet the expected column configuration or
        if the sum of counts in the output deviates from the total input count.
    """
    pop_annual = pd.DataFrame(
        {
            "Year": [2010, 2010, 2010, 2010],
            "Sex": ["M", "M", "F", "F"],
            "Age": [0, 1, 0, 1],
            "Count": [100.0, 50.0, 120.0, 30.0],
        }
    )

    district_breakdown = pd.Series({"A": 0.25, "B": 0.75}, name="Count")
    district_breakdown.index.name = "District"  # IMPORTANT

    district_nums = pd.DataFrame(
        {"District_Num": [0, 1], "Region": ["R1", "R2"]},
        index=pd.Index(["A", "B"], name="District"),
    )

    out = wpp._build_init_population_by_district(
        pop_annual=pop_annual,
        district_breakdown=district_breakdown,
        district_nums=district_nums,
        init_year=2010,
    )
    assert set(out.columns) == {"District", "District_Num", "Region", "Sex", "Age", "Count"}
    assert float(out["Count"].sum()) == pytest.approx(float(pop_annual["Count"].sum()))


# ------------------------------------------------------------------------------
# validate() tests: avoid ResourceBuilder expectations by using real builder but proper ctx
# ------------------------------------------------------------------------------
def _minimal_outputs_for_validate() -> dict[str, pd.DataFrame]:
    return {
        "ResourceFile_Pop_WPP.csv": pd.DataFrame(
            {
                "Variant": ["WPP_Estimates"],
                "Year": [2010],
                "Sex": ["M"],
                "Age_Grp": ["0-4"],
                "Count": [1.0],
            }
        ),
        "ResourceFile_TotalBirths_WPP.csv": pd.DataFrame(
            {
                "Variant": ["Estimates"],
                "Period": ["2010-2014"],
                "Total_Births": [100.0],
                "M_to_F_Sex_Ratio": [1.05],
            }
        ),
        "ResourceFile_Pop_Frac_Births_Male.csv": pd.DataFrame(
            {"Year": [2010], "frac_births_male": [0.51]}
        ),
        "ResourceFile_ASFR_WPP.csv": pd.DataFrame(
            {
                "Variant": ["WPP_Estimates"],
                "Period": ["2010-2014"],
                "Age_Grp": ["15-19"],
                "asfr": [0.05],
            }
        ),
        "ResourceFile_TotalDeaths_WPP.csv": pd.DataFrame(
            {
                "Variant": ["WPP_Estimates"],
                "Period": ["2010-2014"],
                "Sex": ["M"],
                "Age_Grp": ["0-4"],
                "Count": [10.0],
            }
        ),
        "ResourceFile_Pop_DeathRates_WPP.csv": pd.DataFrame(
            {
                "Variant": ["WPP_Medium"],
                "Period": ["2010-2014"],
                "Sex": ["M"],
                "Age_Grp": ["0-4"],
                "death_rate": [0.1],
            }
        ),
        "ResourceFile_Pop_DeathRates_Expanded_WPP.csv": pd.DataFrame(
            {"fallbackyear": [2010], "sex": ["M"], "age_years": [0], "death_rate": [0.1]}
        ),
    }


def test_validate_passes_on_minimal_outputs(ctx: DummyCtx) -> None:
    """
    Validates the outputs using the WPPBuilder instance with minimal outputs provided.

    Parameters
    ----------
    ctx : DummyCtx
        The context object providing necessary attributes or methods for validation.

    Raises
    ------
    Exception
        If validation fails for the given outputs.

    """
    builder = wpp.WPPBuilder(ctx)  # type: ignore[arg-type]
    outputs = _minimal_outputs_for_validate()
    builder.validate(outputs)


def test_validate_raises_on_empty_core_output(ctx: DummyCtx) -> None:
    """
    Tests the validation process for WPPBuilder to ensure it raises an error
    when given an output dataset with an empty core file. This test verifies
    that an AssertionError is triggered with the appropriate error message.

    Parameters:
        ctx (DummyCtx): The context object required to initialize WPPBuilder.

    Raises:
        AssertionError: If validation fails due to ResourceFile_TotalBirths_WPP.csv
        being empty.
    """
    builder = wpp.WPPBuilder(ctx)  # type: ignore[arg-type]
    outputs = _minimal_outputs_for_validate()
    outputs["ResourceFile_TotalBirths_WPP.csv"] = outputs["ResourceFile_TotalBirths_WPP.csv"].iloc[
        0:0
    ]
    with pytest.raises(AssertionError, match="ResourceFile_TotalBirths_WPP.csv is empty"):
        builder.validate(outputs)


def test_validate_raises_on_negative_population(ctx: DummyCtx) -> None:
    """
    Tests if the validation method of the WPPBuilder class raises an AssertionError
    when encountering a negative population count in the generated outputs.

    Raises
    ------
    AssertionError
        If a negative population count is detected in the output data.
    """
    builder = wpp.WPPBuilder(ctx)  # type: ignore[arg-type]
    outputs = _minimal_outputs_for_validate()
    outputs["ResourceFile_Pop_WPP.csv"].loc[0, "Count"] = -1
    with pytest.raises(AssertionError, match="negative counts"):
        builder.validate(outputs)


def test_validate_raises_on_asfr_out_of_bounds(ctx: DummyCtx) -> None:
    """
    Test function to validate that an exception is raised when ASFR
    (Age-Specific Fertility Rate) values in the input data are out of bounds.
    This test specifically tests for an `AssertionError` if an ASFR value
    is set to an implausibly large number, such as 2.5. The test ensures
    that the validation step in the `WPPBuilder` produces the expected
    behavior when invalid data is encountered.

    Parameters:
    ctx (DummyCtx): Context object to pass into the `WPPBuilder`.

    Raises:
    AssertionError: If the ASFR value in the input data is implausibly large.
    """
    builder = wpp.WPPBuilder(ctx)  # type: ignore[arg-type]
    outputs = _minimal_outputs_for_validate()
    outputs["ResourceFile_ASFR_WPP.csv"].loc[0, "asfr"] = 2.5
    with pytest.raises(AssertionError, match="implausibly large"):
        builder.validate(outputs)


def test_validate_raises_on_negative_death_rate(ctx: DummyCtx) -> None:
    """
    Tests the validation method within the WPPBuilder class to check if it raises
    an assertion error when a negative death rate is encountered.

    Args:
        ctx (DummyCtx): Context used for setting up the test environment.

    Raises:
        AssertionError: If the death rate is negative, validation will
        raise this error with the specified error message.
    """
    builder = wpp.WPPBuilder(ctx)  # type: ignore[arg-type]
    outputs = _minimal_outputs_for_validate()
    outputs["ResourceFile_Pop_DeathRates_WPP.csv"].loc[0, "death_rate"] = -0.01
    with pytest.raises(AssertionError, match="negative death_rate"):
        builder.validate(outputs)


def test_validate_checks_init_population_sum_matches_pop_annual(ctx: DummyCtx) -> None:
    """
    Tests the `validate` method of `WPPBuilder` to ensure that an assertion error is raised
    when there is a mismatch between the initial population sum from annual population data
    and the initial population file for the year 2010.

    Arguments:
        ctx: DummyCtx
            A mock or dummy context object used for testing.

    Raises:
        AssertionError: Raised if the initial population sum calculated from the annual
            WPP data does not match with the initial population file for the year 2010.
    """
    builder = wpp.WPPBuilder(ctx)  # type: ignore[arg-type]
    outputs = _minimal_outputs_for_validate()

    outputs["ResourceFile_Pop_Annual_WPP.csv"] = pd.DataFrame(
        {
            "Variant": ["WPP_Estimates"],
            "Year": [2010],
            "Sex": ["M"],
            "Age": [0],
            "Count": [100.0],
            "Period": ["2010-2014"],
            "Age_Grp": ["0-4"],
        }
    )
    outputs["ResourceFile_Population_2010.csv"] = pd.DataFrame(
        {
            "District": ["A"],
            "District_Num": [0],
            "Region": ["R"],
            "Sex": ["M"],
            "Age": [0],
            "Count": [99.0],
        }
    )

    with pytest.raises(AssertionError, match="Init population sum mismatch"):
        builder.validate(outputs)
