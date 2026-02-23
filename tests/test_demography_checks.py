"""
Module for testing data validation checks in the demographic pipeline.

This module includes a suite of tests to verify the expected behavior of various
data checks such as validating numerical counts, age ranges, and gender values.
Each test ensures that specific edge cases are handled correctly, providing adequate
coverage for the reliability and correctness of the checks implemented in the pipeline.
"""

import pandas as pd

from pipeline.components.common.checks import (
    check_age_range,
    check_no_negative_counts,
    check_sex_values,
    run_all_checks,
)


def _as_map(df: pd.DataFrame, name: str = "table") -> dict[str, pd.DataFrame]:
    """
    Converts a pandas DataFrame into a dictionary representation.

    The dictionary will consist of a single key-value pair where the key is the provided
    name and the value is the DataFrame itself.

    Parameters:
    df: pd.DataFrame
        The DataFrame to be converted into a dictionary.
    name: str, optional
        The desired key name in the resulting dictionary. Defaults to "table".

    Returns:
    dict[str, pd.DataFrame]
        A dictionary containing the provided DataFrame with the specified key as its
        identifier.
    """
    return {name: df}


def test_check_no_negative_counts_pass():
    """
    Tests that the `check_no_negative_counts` function correctly identifies and validates
    data where there are no negative counts.

    Raises
    ------
    AssertionError
        If the results from `check_no_negative_counts` do not match the expected output
        or if the test case fails.
    """
    df = pd.DataFrame({"Count": [0, 1.5, 2]})
    results = check_no_negative_counts(_as_map(df, "pop"))
    assert len(results) == 1
    r = results[0]
    assert r.name == "pop:no_negative_counts"
    assert r.status == "PASS"


def test_check_no_negative_counts_fail():
    """
    Tests the functionality of the check_no_negative_counts function to ensure it fails when
    negative counts are present in the dataset.

    Checks the result of applying the check_no_negative_counts function on a DataFrame
    containing both positive and negative counts. The function is expected to return a
    result with a FAIL status and a message indicating the number of negative values found.

    Raises:
        AssertionError: If the test conditions are not met.
    """
    df = pd.DataFrame({"Count": [1, -0.1, 2, -3]})
    results = check_no_negative_counts(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "FAIL"
    assert "Found 2 negative" in r.message


def test_check_no_negative_counts_skip_when_missing_count():
    """
    Tests the `check_no_negative_counts` function to ensure it properly handles the
    case when the `Count` column is missing from the provided DataFrame. Verifies
    that the check status is set to "SKIP" and the appropriate message is included
    in the results.

    Raises:
        AssertionError: If the check status or message does not match the expected
        values.
    """
    df = pd.DataFrame({"Age": [0, 1]})
    results = check_no_negative_counts(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "SKIP"
    assert "No 'Count' column" in r.message


def test_check_age_range_pass_default_max_age_120():
    """
    Validates the functionality of checking the age range within a dataframe using
     the default maximum allowable age (120). Ensures that all age values fall within
    the acceptable range and confirms that the status is marked as "PASS".

    """
    df = pd.DataFrame({"Age": [0, 1, 50, 120]})
    results = check_age_range(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "PASS"


def test_check_age_range_fail_out_of_bounds():
    """
    Tests the check_age_range function for handling out-of-bounds age values in
    a DataFrame. Verifies that the function correctly identifies and reports
    ages that are outside the allowed range of [0, max_age].

    Raises:
        AssertionError: If the test asserts fail during execution.
    """
    df = pd.DataFrame({"Age": [-1, 0, 121]})
    results = check_age_range(_as_map(df, "pop"), max_age=120)
    r = results[0]
    assert r.status == "FAIL"
    assert "outside [0,120]" in r.message


def test_check_age_range_skip_when_missing_age():
    """
    Tests the behavior of the `check_age_range` function when the input DataFrame
    does not contain an 'Age' column. Ensures that the function properly skips
    validation in this scenario and provides an appropriate status and message
    in the results.

    Raises:
        AssertionError: If the returned status is not "SKIP" or the result message
                        does not indicate the absence of the 'Age' column.
    """
    df = pd.DataFrame({"Count": [1, 2]})
    results = check_age_range(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "SKIP"
    assert "No 'Age' column" in r.message


def test_check_sex_values_pass_only_m_f():
    """
    Tests the `check_sex_values` function to check that only valid "M" and "F" values
    are present in the "Sex" column of the input data.

    Raises
    ------
    AssertionError
        If the test results do not indicate a "PASS" status when only valid "M"
        and "F" values are provided in the "Sex" column.

    """
    df = pd.DataFrame({"Sex": ["M", "F", "M"]})
    results = check_sex_values(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "PASS"


def test_check_sex_values_warn_on_unexpected_values():
    """
    Tests the functionality of `check_sex_values` to ensure it identifies unexpected
    values within a given DataFrame's 'Sex' column and correctly issues a warning.

    Checks if the function `check_sex_values`, when applied to a DataFrame containing
    a mix of expected and unexpected 'Sex' values, returns a warning result highlighting
    the presence of unexpected values.

    Parameters:
    df : pd.DataFrame
        Input DataFrame with a column 'Sex' containing values to check.

    Raises:
    AssertionError
        If the test fails due to mismatched expected and actual results.
    """
    df = pd.DataFrame({"Sex": ["M", "F", "U"]})
    results = check_sex_values(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "WARN"
    assert "Unexpected Sex values" in r.message
    assert "U" in r.message


def test_check_sex_values_skip_when_missing_sex():
    """
    Tests the behavior of `check_sex_values` when the input dataframe is
    missing the 'Sex' column. Verifies that the function correctly skips
    the check and provides an appropriate status and message in the result.

    Raises
    ------
    AssertionError
        If the test assertions fail.
    """
    df = pd.DataFrame({"Age": [0, 1]})
    results = check_sex_values(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "SKIP"
    assert "No 'Sex' column" in r.message


def test_run_all_checks_ok_to_continue_true_when_no_failures():
    """
    Tests the `run_all_checks` function to ensure all checks pass when there are no failures.

    This test verifies the proper functionality of the `run_all_checks` function by evaluating
    sample input data against predefined checks. It ensures that when all checks pass, the
    function returns a boolean `True` for the "ok" flag and the expected statuses for each
    defined check in the result.

    Raises:
        AssertionError: If the "ok" flag is not `True` or any of the defined checks do not
        return the expected "PASS" status.
    """
    df = pd.DataFrame({"Count": [1, 2], "Age": [0, 1], "Sex": ["M", "F"]})
    results, ok = run_all_checks(_as_map(df, "pop"))
    assert ok is True
    assert any(r.name == "pop:no_negative_counts" and r.status == "PASS" for r in results)
    assert any(r.name == "pop:age_range" and r.status == "PASS" for r in results)
    assert any(r.name == "pop:sex_values" and r.status == "PASS" for r in results)


def test_run_all_checks_ok_to_continue_false_when_any_fail():
    """
    Verifies that when running all checks, the procedure correctly identifies a failure
    and sets the state to not ok (`ok is False`) when any check fails. This test ensures
    that the function `run_all_checks` handles invalid scenarios by processing the
    results properly and returning a failure indication.

    Raises:
        AssertionError: If the checks do not fail as expected or if the result
        state (`ok` flag) does not reflect a failure condition.
    """
    df = pd.DataFrame({"Count": [1, -1], "Age": [0, 1], "Sex": ["M", "F"]})
    results, ok = run_all_checks(_as_map(df, "pop"))
    assert ok is False
    # The negative count should drive a FAIL
    assert any(r.status == "FAIL" for r in results)
