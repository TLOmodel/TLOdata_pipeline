from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.components.demography.census import CensusBuilder
from pipeline.components.resource_builder import BuildContext


def _fake_census_workbook(census_year: int) -> dict[str, pd.DataFrame]:
    # regions sheet
    regions = pd.DataFrame({"Region": ["North", "South"]})

    # pop_totals sheet structure expected by your cleanup:
    # - drops rows [0,1] and then drops first remaining row => we provide 3 junk rows then real data.
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
def test_census_builder_writes_expected_output_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, year: int
) -> None:
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
    wb = _fake_census_workbook(year)

    def fake_read_excel(path, sheet_name=None, *args, **kwargs):
        assert str(path) == str(census_path)
        if sheet_name is None:
            return wb
        raise AssertionError("CensusBuilder should read sheet_name=None (entire workbook).")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    ctx = BuildContext(
        cfg=cfg,
        country="tz",
        raw_dir=raw_dir,
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

    manifest = ctx.output_dir / "resource_manifest.json"
    assert manifest.exists()

    df = pd.read_csv(out_path)
    required_cols = {
        "Variant",
        "District",
        "District_Num",
        "Region",
        "Year",
        "Period",
        "Age_Grp",
        "Sex",
        "Count",
    }
    assert required_cols.issubset(df.columns)
    assert not df.empty
