import pandas as pd

from tlo_data_pipeline.demography.checks import (
    check_no_negative_counts,
    check_age_range,
    check_sex_values,
    run_all_checks,
)


def _as_map(df: pd.DataFrame, name: str = "table") -> dict[str, pd.DataFrame]:
    return {name: df}


def test_check_no_negative_counts_pass():
    df = pd.DataFrame({"Count": [0, 1.5, 2]})
    results = check_no_negative_counts(_as_map(df, "pop"))
    assert len(results) == 1
    r = results[0]
    assert r.name == "pop:no_negative_counts"
    assert r.status == "PASS"


def test_check_no_negative_counts_fail():
    df = pd.DataFrame({"Count": [1, -0.1, 2, -3]})
    results = check_no_negative_counts(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "FAIL"
    assert "Found 2 negative" in r.message


def test_check_no_negative_counts_skip_when_missing_count():
    df = pd.DataFrame({"Age": [0, 1]})
    results = check_no_negative_counts(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "SKIP"
    assert "No 'Count' column" in r.message


def test_check_age_range_pass_default_max_age_120():
    df = pd.DataFrame({"Age": [0, 1, 50, 120]})
    results = check_age_range(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "PASS"


def test_check_age_range_fail_out_of_bounds():
    df = pd.DataFrame({"Age": [-1, 0, 121]})
    results = check_age_range(_as_map(df, "pop"), max_age=120)
    r = results[0]
    assert r.status == "FAIL"
    assert "outside [0,120]" in r.message


def test_check_age_range_skip_when_missing_age():
    df = pd.DataFrame({"Count": [1, 2]})
    results = check_age_range(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "SKIP"
    assert "No 'Age' column" in r.message


def test_check_sex_values_pass_only_m_f():
    df = pd.DataFrame({"Sex": ["M", "F", "M"]})
    results = check_sex_values(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "PASS"


def test_check_sex_values_warn_on_unexpected_values():
    df = pd.DataFrame({"Sex": ["M", "F", "U"]})
    results = check_sex_values(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "WARN"
    assert "Unexpected Sex values" in r.message
    assert "U" in r.message


def test_check_sex_values_skip_when_missing_sex():
    df = pd.DataFrame({"Age": [0, 1]})
    results = check_sex_values(_as_map(df, "pop"))
    r = results[0]
    assert r.status == "SKIP"
    assert "No 'Sex' column" in r.message


def test_run_all_checks_ok_to_continue_true_when_no_failures():
    df = pd.DataFrame({"Count": [1, 2], "Age": [0, 1], "Sex": ["M", "F"]})
    results, ok = run_all_checks(_as_map(df, "pop"))
    assert ok is True
    assert any(r.name == "pop:no_negative_counts" and r.status == "PASS" for r in results)
    assert any(r.name == "pop:age_range" and r.status == "PASS" for r in results)
    assert any(r.name == "pop:sex_values" and r.status == "PASS" for r in results)


def test_run_all_checks_ok_to_continue_false_when_any_fail():
    df = pd.DataFrame({"Count": [1, -1], "Age": [0, 1], "Sex": ["M", "F"]})
    results, ok = run_all_checks(_as_map(df, "pop"))
    assert ok is False
    # The negative count should drive a FAIL
    assert any(r.status == "FAIL" for r in results)
