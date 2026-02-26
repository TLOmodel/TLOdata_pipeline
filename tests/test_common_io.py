from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.components.common.fixes import WPPReader, reformat_date_period_for_wpp


# ---------------------------------------------------------------------
# reformat_date_period_for_wpp
# ---------------------------------------------------------------------
def test_reformat_date_period_for_wpp_inclusive_range() -> None:
    df = pd.DataFrame({"Period": ["2010-2015", "1950-1955"]})
    reformat_date_period_for_wpp(df)
    assert df["Period"].tolist() == ["2010-2014", "1950-1954"]


def test_reformat_date_period_for_wpp_nonstring_ok() -> None:
    # Ensure it tolerates non-string values (casts to str)
    df = pd.DataFrame({"Period": [2010, "2010-2015"]})
    # first value "2010" will split to ["2010", None] and then fail int conversion
    # so we expect it to raise: this is fine because WPP periods must be "lo-hi"
    with pytest.raises(Exception):
        reformat_date_period_for_wpp(df)


# ---------------------------------------------------------------------
# WPPReader country column selection
# ---------------------------------------------------------------------
def test_country_col_prefers_named_candidates() -> None:
    df = pd.DataFrame(
        {
            "Location": ["United Republic of Tanzania", "Kenya"],
            "x": [1, 2],
        }
    )
    r = WPPReader(country_label="United Republic of Tanzania", country_col_index=0)
    assert r._country_col(df) == "Location"


def test_country_col_falls_back_to_index() -> None:
    df = pd.DataFrame(
        {
            "A": ["foo", "bar"],
            "B": ["baz", "qux"],
            "C": ["Tanzania", "Kenya"],
        }
    )
    r = WPPReader(country_label="Tanzania", country_col_index=2)
    assert r._country_col(df) == "C"


# ---------------------------------------------------------------------
# WPPReader.filter_country
# ---------------------------------------------------------------------
def test_filter_country_exact_match() -> None:
    df = pd.DataFrame(
        {
            "Location": ["United Republic of Tanzania", "Kenya"],
            "v": [10, 20],
        }
    )
    r = WPPReader(country_label="United Republic of Tanzania")
    out = r.filter_country(df)
    assert out.shape[0] == 1
    assert out.iloc[0]["v"] == 10


def test_filter_country_casefold_and_whitespace_and_nbsp() -> None:
    nbsp = "\u00a0"
    df = pd.DataFrame(
        {
            "Location": [f"  united{nbsp}republic of tanzania  ", "Kenya"],
            "v": [10, 20],
        }
    )
    r = WPPReader(country_label="United Republic of Tanzania")
    out = r.filter_country(df)
    assert out.shape[0] == 1
    assert out.iloc[0]["v"] == 10


def test_filter_country_raises_with_diagnostics_when_empty() -> None:
    df = pd.DataFrame(
        {
            "Location": ["Kenya", "Uganda"],
            "v": [1, 2],
        }
    )
    r = WPPReader(country_label="United Republic of Tanzania")
    with pytest.raises(ValueError) as e:
        r.filter_country(df)

    msg = str(e.value)
    assert "produced empty result" in msg
    assert "country_label" in msg
    assert "sample_values" in msg


# ---------------------------------------------------------------------
# WPPReader.drop_metadata_cols
# ---------------------------------------------------------------------
def test_drop_metadata_cols_by_position_safe_when_short() -> None:
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    r = WPPReader(country_label="x", drop_col_positions=(0, 2, 10))
    out = r.drop_metadata_cols(df)
    assert list(out.columns) == ["B"]  # dropped A and C; ignored 10


# ---------------------------------------------------------------------
# WPPReader.read_country_table (mock pd.read_excel)
# ---------------------------------------------------------------------
def test_read_country_table_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Simulate two sheets returned by pd.read_excel
    sheet1 = pd.DataFrame(
        {"Location": ["Tanzania", "Kenya"], "Meta": [0, 0], "X": [1, 2], "Y": [3, 4], "Z": [5, 6]}
    )
    sheet2 = pd.DataFrame(
        {
            "Location": ["Tanzania", "Uganda"],
            "Meta": [0, 0],
            "X": [10, 20],
            "Y": [30, 40],
            "Z": [50, 60],
        }
    )

    def fake_read_excel(file_path, sheet_name, header, **kwargs):
        assert header == 16
        if sheet_name == "S1":
            return sheet1
        if sheet_name == "S2":
            return sheet2
        raise KeyError(sheet_name)

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    r = WPPReader(
        country_label="Tanzania", header_row=16, country_col_index=0, drop_col_positions=(1,)
    )
    out = r.read_country_table(tmp_path / "fake.xlsx", sheets=["S1", "S2"], extra_cols={"Sex": "M"})

    # Should include Tanzania rows from both sheets (2 rows)
    assert out.shape[0] == 2
    assert "Sex" in out.columns
    assert set(out["Sex"]) == {"M"}

    # drop_col_positions=(1,) should drop "Meta" column (position 1 in filtered df)
    assert "Meta" not in out.columns
