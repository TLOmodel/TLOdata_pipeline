from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.components.resource_builder import BuildContext
from pipeline.components.demography.dhs import DHSBuilder


def test_dhs_builder_writes_expected_outputs_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def fake_read_excel(path, sheet_name=None, header=None, *args, **kwargs):
        assert str(path) == str(dhs_path)
        if sheet_name == "ASFR":
            return asfr_df.copy()
        if sheet_name == "UNDER_5_MORT":
            # header argument is allowed; ignore for this test
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
    assert (ctx.output_dir / "resource_manifest.json").exists()

    # quick sanity: ASFR scaled per-woman (<= 2 check is in validate)
    asfr_out = pd.read_csv(ctx.output_dir / "ResourceFile_ASFR_DHS.csv")
    assert not asfr_out.empty

    u5_out = pd.read_csv(ctx.output_dir / "ResourceFile_Under_Five_Mortality_DHS.csv")
    assert list(u5_out.columns) == ["Year", "Est", "Lo", "Hi"]
    assert not u5_out.empty