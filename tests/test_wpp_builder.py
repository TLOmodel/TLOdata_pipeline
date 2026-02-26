from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.components.resource_builder import BuildContext
from pipeline.components.demography import wpp as wpp_mod
from pipeline.components.demography.wpp import WPPBuilder


def _make_pop_agegrp_df() -> pd.DataFrame:
    # Must have columns such that build() does: ests.columns[2:23] scaling
    # => need at least 23 columns: Variant, Year, then 21 age group cols.
    cols = ["Variant", "Year"] + [f"age{i}" for i in range(21)]
    row = ["Estimates", 2010] + [1] * 21
    return pd.DataFrame([row], columns=cols)


def _make_births_df() -> pd.DataFrame:
    # read_country_table returns a "wide" table with Variant + period columns
    return pd.DataFrame(
        {
            "Variant": ["Estimates"],
            "2010-2015": [10.0],
        }
    )


def _make_sex_ratio_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Variant": ["Medium variant"],
            "2010-2015": [1.05],
        }
    )


def _make_asfr_df() -> pd.DataFrame:
    # build() does asfr2.columns[2:9] /1000 => needs at least 9 columns
    cols = ["Variant", "Period"] + [f"a{i}" for i in range(7)]
    row = ["Estimates", "2010-2015"] + [100] * 7
    return pd.DataFrame([row], columns=cols)


def _make_deaths_df(extra_sex: str) -> pd.DataFrame:
    # build() scales d.columns[2:22] => need 22 columns.
    cols = ["Variant", "Period"] + [f"d{i}" for i in range(20)]
    df = pd.DataFrame([["Estimates", "2010-2015"] + [1] * 20], columns=cols)
    df["Sex"] = extra_sex
    return df


def test_wpp_builder_writes_expected_outputs_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "inputs" / "demography"
    raw_dir.mkdir(parents=True)
    resources_dir = tmp_path / "outputs" / "resources"

    # Create fake input files just so preflight() can pass existence checks
    def touch(rel: str) -> str:
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
    def fake_read_country_table(self, file_path, sheets, extra_cols=None, **kwargs):
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

    monkeypatch.setattr(wpp_mod, "_lifetable_to_death_rates", lambda *, cfg: lt_out.copy())
    monkeypatch.setattr(wpp_mod, "_expand_death_rates", lambda *, cfg, lt_out: expanded.copy())

    ctx = BuildContext(
        cfg=cfg,
        country="tz",
        raw_dir=raw_dir,
        resources_dir=resources_dir,
        component="demography",
    )

    artifacts = WPPBuilder(ctx).run()
    names = {a.name for a in artifacts}

    for expected in WPPBuilder.EXPECTED_OUTPUTS:
        assert expected in names
        assert (ctx.output_dir / expected).exists()

    assert (ctx.output_dir / "resource_manifest.json").exists()

    # Schema sanity: Pop_WPP must include Count
    pop = pd.read_csv(ctx.output_dir / "ResourceFile_Pop_WPP.csv")
    assert "Count" in pop.columns
    assert not pop.empty