from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL" | "SKIP"
    message: str


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def check_no_negative_counts(df_map: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for name, df in df_map.items():
        if "Count" not in df.columns:
            results.append(CheckResult(f"{name}:no_negative_counts", "SKIP", "No 'Count' column"))
            continue
        if (df["Count"] < 0).any():
            n = int((df["Count"] < 0).sum())
            results.append(CheckResult(f"{name}:no_negative_counts", "FAIL", f"Found {n} negative Count values"))
        else:
            results.append(CheckResult(f"{name}:no_negative_counts", "PASS", "OK"))
    return results


def check_age_range(df_map: Dict[str, pd.DataFrame], max_age: int = 120) -> List[CheckResult]:
    results: List[CheckResult] = []
    for name, df in df_map.items():
        if "Age" not in df.columns:
            results.append(CheckResult(f"{name}:age_range", "SKIP", "No 'Age' column"))
            continue
        bad = df[(df["Age"] < 0) | (df["Age"] > max_age)]
        if not bad.empty:
            results.append(CheckResult(
                f"{name}:age_range",
                "FAIL",
                f"Found ages outside [0,{max_age}] (n={len(bad)})"
            ))
        else:
            results.append(CheckResult(f"{name}:age_range", "PASS", "OK"))
    return results


def check_sex_values(df_map: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for name, df in df_map.items():
        if "Sex" not in df.columns:
            results.append(CheckResult(f"{name}:sex_values", "SKIP", "No 'Sex' column"))
            continue
        allowed = {"M", "F"}
        vals = set(df["Sex"].dropna().astype(str).unique().tolist())
        extra = vals - allowed
        if extra:
            results.append(CheckResult(f"{name}:sex_values", "WARN", f"Unexpected Sex values: {sorted(extra)}"))
        else:
            results.append(CheckResult(f"{name}:sex_values", "PASS", "OK"))
    return results


def run_all_checks(df_map: Dict[str, pd.DataFrame]) -> Tuple[List[CheckResult], bool]:
    """Returns (results, ok_to_continue)."""
    results: List[CheckResult] = []
    results += check_no_negative_counts(df_map)
    results += check_age_range(df_map)
    results += check_sex_values(df_map)

    hard_fail = any(r.status == "FAIL" for r in results)
    return results, (not hard_fail)
