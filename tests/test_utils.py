from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.components.resource_builder import BuildContext
from pipeline.components.utils import (
    create_age_range_lookup,
    load_cfg,
    make_calendar_period_lookup,
    resolve_input_path,
    resolve_templates,
)


# ---------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------
def test_resolve_templates_simple_nested() -> None:
    cfg: dict[str, Any] = {
        "country_name": "Tanzania",
        "paths": {"input_dir": "inputs/demography"},
        "wpp_name": "United Republic of Tanzania",
        "wpp": {
            "country_label": "{wpp_name}",
            "pop_file": "{paths.input_dir}/wpp/file.xlsx",
        },
    }

    out = resolve_templates(cfg)

    assert out["wpp"]["country_label"] == "United Republic of Tanzania"
    assert out["wpp"]["pop_file"] == "inputs/demography/wpp/file.xlsx"


def test_resolve_templates_multi_pass() -> None:
    cfg: dict[str, Any] = {
        "a": {"x": "hello"},
        "b": {"y": "{a.x} world"},
        "c": {"z": "{b.y}!!!"},
    }

    out = resolve_templates(cfg)
    assert out["c"]["z"] == "hello world!!!"


def test_resolve_templates_missing_key_raises_keyerror() -> None:
    cfg: dict[str, Any] = {"paths": {"input_dir": "inputs"}, "x": "{paths.missing}"}
    with pytest.raises(KeyError):
        resolve_templates(cfg)


def test_resolve_templates_unresolved_placeholders_raise_valueerror() -> None:
    # This simulates a circular or not-fully-resolvable template that survives passes
    cfg: dict[str, Any] = {"a": "{b}", "b": "{a}"}
    with pytest.raises(ValueError, match="Unresolved"):
        resolve_templates(cfg, max_passes=3)


# ---------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------
def test_load_cfg_reads_yaml_and_resolves_templates(tmp_path: Path) -> None:
    cfg_path = tmp_path / "tz.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "country_name: Tanzania",
                "paths:",
                "  input_dir: inputs/demography",
                "wpp_name: United Republic of Tanzania",
                "wpp:",
                "  country_label: '{wpp_name}'",
                "  file: '{paths.input_dir}/wpp/x.xlsx'",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_cfg(cfg_path)
    assert cfg["wpp"]["country_label"] == "United Republic of Tanzania"
    assert cfg["wpp"]["file"] == "inputs/demography/wpp/x.xlsx"


def test_load_cfg_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cfg(tmp_path / "does_not_exist.yaml")


def test_load_cfg_non_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")  # YAML list, not mapping
    with pytest.raises(ValueError, match="Config must be a YAML mapping"):
        load_cfg(p)


# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------
def test_resolve_input_path_relative_joins_raw_dir(tmp_path: Path) -> None:
    raw_dir = tmp_path / "inputs" / "demography"
    raw_dir.mkdir(parents=True)
    ctx = BuildContext(
        cfg={},
        country="tz",
        raw_dir=raw_dir,
        resources_dir=tmp_path / "outputs" / "resources",
        component="demography",
    )

    p = resolve_input_path(ctx, "wpp/file.xlsx")
    assert p == (raw_dir / "wpp/file.xlsx").resolve()


def test_resolve_input_path_absolute_passthrough(tmp_path: Path) -> None:
    raw_dir = tmp_path / "inputs" / "demography"
    raw_dir.mkdir(parents=True)
    ctx = BuildContext(
        cfg={},
        country="tz",
        raw_dir=raw_dir,
        resources_dir=tmp_path / "outputs" / "resources",
        component="demography",
    )

    abs_path = tmp_path / "somewhere" / "x.xlsx"
    # doesn't need to exist for this function
    p = resolve_input_path(ctx, abs_path)
    assert p == abs_path


def test_resolve_input_path_unresolved_template_raises(tmp_path: Path) -> None:
    raw_dir = tmp_path / "inputs" / "demography"
    raw_dir.mkdir(parents=True)
    ctx = BuildContext(
        cfg={},
        country="tz",
        raw_dir=raw_dir,
        resources_dir=tmp_path / "outputs" / "resources",
        component="demography",
    )

    with pytest.raises(ValueError, match="Unresolved template"):
        resolve_input_path(ctx, "{paths.input_dir}/wpp/file.xlsx")


# ---------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------
def test_create_age_range_lookup_basic() -> None:
    lookup = create_age_range_lookup(min_age=0, max_age=10, range_size=5)
    assert lookup[0] == "0-4"
    assert lookup[4] == "0-4"
    assert lookup[5] == "5-9"
    assert lookup[9] == "5-9"
    assert lookup[10] == "10+"  # default category


def test_create_age_range_lookup_with_under_min() -> None:
    lookup = create_age_range_lookup(min_age=2, max_age=10, range_size=5)
    assert lookup[0] == "0-2"
    assert lookup[1] == "0-2"
    assert lookup[2] == "2-6" or lookup[2] == "2-6"  # depends on range definition
    assert lookup[10] == "10+"


def test_make_calendar_period_lookup_edges() -> None:
    lookup = make_calendar_period_lookup()
    assert 1950 in lookup
    assert lookup[1950] == "1950-1954"
    assert lookup[1954] == "1950-1954"
    assert lookup[1955] == "1955-1959"
    assert 1949 not in lookup
