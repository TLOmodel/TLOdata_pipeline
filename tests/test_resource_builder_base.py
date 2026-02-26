from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.components.resource_builder import BuildContext, ResourceBuilder


class _DummyBuilder(ResourceBuilder):
    COMPONENT = "demography"
    REQUIRED_INPUTS = ("needed.txt",)
    EXPECTED_OUTPUTS = ("a.csv", "b.csv")

    def load_data(self) -> Mapping[str, Any]:
        return {"x": 1}

    def build(self, raw: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        # intentionally return unsorted keys to ensure ResourceBuilder.write sorts keys
        return {
            "b.csv": pd.DataFrame({"b": [2, 3]}),
            "a.csv": pd.DataFrame({"a": [1]}),
        }


def _ctx(tmp_path: Path) -> BuildContext:
    raw_dir = tmp_path / "inputs"
    res_dir = tmp_path / "outputs" / "resources"
    raw_dir.mkdir(parents=True, exist_ok=True)

    return BuildContext(
        cfg={},
        country="tz",
        raw_dir=raw_dir,
        resources_dir=res_dir,
        component="demography",
    )


# ---------------------------------------------------------------------
# preflight / required inputs
# ---------------------------------------------------------------------
def test_preflight_creates_output_dir_and_checks_required_inputs(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    b = _DummyBuilder(ctx)

    # missing required input should raise
    with pytest.raises(FileNotFoundError) as e:
        b.preflight()

    assert "Missing required raw inputs" in str(e.value)

    # create required input
    (ctx.raw_dir / "needed.txt").write_text("ok", encoding="utf-8")

    # now preflight passes and output dir exists
    b.preflight()
    assert ctx.output_dir.exists()
    assert ctx.output_dir.is_dir()


# ---------------------------------------------------------------------
# run() end-to-end (write + manifest)
# ---------------------------------------------------------------------
def test_run_writes_csvs_and_manifest(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    (ctx.raw_dir / "needed.txt").write_text("ok", encoding="utf-8")

    b = _DummyBuilder(ctx, dry_run=False)
    artifacts = b.run()

    # artifacts should be in sorted name order due to write() sorting keys
    assert [a.name for a in artifacts] == ["a.csv", "b.csv"]

    # csv files exist
    assert (ctx.output_dir / "a.csv").exists()
    assert (ctx.output_dir / "b.csv").exists()

    # manifest exists and is valid json with expected schema
    manifest_path = ctx.output_dir / "resource_manifest__DummyBuilder.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["component"] == "demography"
    assert manifest["builder"] == "_DummyBuilder"
    assert "artifacts" in manifest
    assert [a["name"] for a in manifest["artifacts"]] == ["a.csv", "b.csv"]


def test_run_dry_run_does_not_write_files(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    (ctx.raw_dir / "needed.txt").write_text("ok", encoding="utf-8")

    b = _DummyBuilder(ctx, dry_run=True)
    artifacts = b.run()

    # artifacts still returned
    assert [a.name for a in artifacts] == ["a.csv", "b.csv"]

    # but no files written
    assert not (ctx.output_dir / "a.csv").exists()
    assert not (ctx.output_dir / "b.csv").exists()
    assert not (ctx.output_dir / "resource_manifest__DummyBuilder.json").exists()


# ---------------------------------------------------------------------
# validate() default enforcement of EXPECTED_OUTPUTS
# ---------------------------------------------------------------------
def test_validate_enforces_expected_outputs(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    b = _DummyBuilder(ctx, dry_run=True)

    # missing "b.csv"
    outputs = {"a.csv": pd.DataFrame({"a": [1]})}
    with pytest.raises(AssertionError, match="Expected outputs missing"):
        b.validate(outputs)


# ---------------------------------------------------------------------
# _normalize_outputs type checks
# ---------------------------------------------------------------------
def test_normalize_outputs_requires_mapping(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    b = _DummyBuilder(ctx)

    with pytest.raises(TypeError, match="build\\(\\) must return"):
        b._normalize_outputs(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_normalize_outputs_requires_string_keys(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    b = _DummyBuilder(ctx)

    bad = {1: pd.DataFrame()}  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="Output keys must be strings"):
        b._normalize_outputs(bad)  # type: ignore[arg-type]


def test_normalize_outputs_requires_dataframe_values(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    b = _DummyBuilder(ctx)

    bad = {"a.csv": "not df"}  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="Output values must be pandas DataFrames"):
        b._normalize_outputs(bad)  # type: ignore[arg-type]