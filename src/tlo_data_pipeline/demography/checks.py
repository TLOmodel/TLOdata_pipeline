"""
Provides checks and validations for pandas DataFrames.

This module defines functionality to perform various data integrity
checks on pandas DataFrames. These include validations for negative
values in specific columns, age range compliance, and acceptable
values in categorical columns. It provides a general mechanism for
returning detailed results and determining whether further processing
can proceed based on the findings.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    """
    Represents the outcome of a specific check or validation process.

    This class is used to encapsulate the result details of a check, including
    its name, the status it concludes with, and an associated message providing
    additional context or explanations. Being immutable, instances of this
    class cannot be modified after creation.

    Attributes:
        name: str
            The unique identifier or descriptive name of the check performed.
        status: str
            Indicates the result of the check. Possible values are "PASS",
            "WARN", "FAIL", and "SKIP".
        message: str
            Provides further details or context about the result of the check.
            It can describe reasons for warnings or failures or other
            information relevant to the outcome.
    """

    name: str
    status: str  # "PASS" | "WARN" | "FAIL" | "SKIP"
    message: str


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def check_no_negative_counts(df_map: dict[str, pd.DataFrame]) -> list[CheckResult]:
    """
    Checks for the presence of negative values in the "Count" column across multiple DataFrames and
    returns a report of the results. This includes identifying DataFrames without a "Count" column
    and summarizing the check status for each DataFrame.

    Raises:
        None

    Args:
        df_map (Dict[str, pd.DataFrame]): A dictionary where keys represent dataset names and values
                                          are the corresponding DataFrames to be validated.

    Returns:
        List[CheckResult]: A list of CheckResult objects summarizing the validation results for
                           each DataFrame. For each dataset:
                           - A "SKIP" result is included if no "Count" column exists.
                           - A "FAIL" result if negative values are found in the "Count" column,
                             reporting the count of such values.
                           - A "PASS" result if no issues are detected.
    """
    results: list[CheckResult] = []
    for name, df in df_map.items():
        if "Count" not in df.columns:
            results.append(CheckResult(f"{name}:no_negative_counts", "SKIP", "No 'Count' column"))
            continue
        if (df["Count"] < 0).any():
            n = int((df["Count"] < 0).sum())
            results.append(
                CheckResult(
                    f"{name}:no_negative_counts", "FAIL", f"Found {n} negative Count values"
                )
            )
        else:
            results.append(CheckResult(f"{name}:no_negative_counts", "PASS", "OK"))
    return results


def check_age_range(df_map: dict[str, pd.DataFrame], max_age: int = 120) -> list[CheckResult]:
    """
    Checks the age ranges in the dataframes of the provided mapping.

    This function iterates through a dictionary of DataFrames, inspects the "Age"
    columns, and validates that all ages fall within the range [0, max_age].
    It will return a list of results that indicate whether the check for each
    DataFrame passed, failed, or was skipped (e.g., missing "Age" column).

    Parameters:
    df_map : Dict[str, pd.DataFrame]
        Mapping of DataFrame names to their corresponding pandas DataFrame
        objects for validation.
    max_age : int, default 120
        The upper limit for valid age values. Ages higher than this will cause
        a DataFrame to fail the check.

    Returns:
    List[CheckResult]
        A list of CheckResult objects containing the validation results for each
        DataFrame in the input mapping.
    """
    results: list[CheckResult] = []
    for name, df in df_map.items():
        if "Age" not in df.columns:
            results.append(CheckResult(f"{name}:age_range", "SKIP", "No 'Age' column"))
            continue
        bad = df[(df["Age"] < 0) | (df["Age"] > max_age)]
        if not bad.empty:
            results.append(
                CheckResult(
                    f"{name}:age_range", "FAIL", f"Found ages outside [0,{max_age}] (n={len(bad)})"
                )
            )
        else:
            results.append(CheckResult(f"{name}:age_range", "PASS", "OK"))
    return results


def check_sex_values(df_map: dict[str, pd.DataFrame]) -> list[CheckResult]:
    """
    Validates the 'Sex' column values in the provided dataframes and checks if any unexpected
    values are present.

    The function iterates through the provided dictionary of dataframes, where the keys are names,
    and the values are pandas DataFrames. It checks whether each DataFrame contains a column
    named 'Sex'. If the column is present, the function compares its unique, non-null values
    against the allowed set {'M', 'F'}. Any values outside this range are flagged as
    unexpected, and a warning result is appended to the output list. If no unexpected values
    are present or the column is missing, appropriate results are returned as well.

    Arguments:
        df_map (Dict[str, pd.DataFrame]): A dictionary where keys are names and values
                                          are pandas DataFrames for validation.

    Returns:
        List[CheckResult]: A list of results indicating the validation status for each
        DataFrame. Each result specifies the DataFrame name, the validation
        status ('PASS', 'WARN', or 'SKIP'), and a message providing additional details.
    """
    results: list[CheckResult] = []
    for name, df in df_map.items():
        if "Sex" not in df.columns:
            results.append(CheckResult(f"{name}:sex_values", "SKIP", "No 'Sex' column"))
            continue
        allowed = {"M", "F"}
        vals = set(df["Sex"].dropna().astype(str).unique().tolist())
        extra = vals - allowed
        if extra:
            results.append(
                CheckResult(f"{name}:sex_values", "WARN", f"Unexpected Sex values: {sorted(extra)}")
            )
        else:
            results.append(CheckResult(f"{name}:sex_values", "PASS", "OK"))
    return results


def run_all_checks(df_map: dict[str, pd.DataFrame]) -> tuple[list[CheckResult], bool]:
    """Returns (results, ok_to_continue)."""
    results: list[CheckResult] = []
    results += check_no_negative_counts(df_map)
    results += check_age_range(df_map)
    results += check_sex_values(df_map)

    hard_fail = any(r.status == "FAIL" for r in results)
    return results, (not hard_fail)
