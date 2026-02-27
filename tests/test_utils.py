"""
Test all functions of the utils module.
"""

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
    """
    Tests the `resolve_templates` function with a configuration containing nested
    templated strings to verify that all templates are resolved correctly.

    This test ensures that placeholders in nested dictionaries are correctly
    interpolated using the provided configuration values.

    Raises:
        AssertionError: If the resolved templates do not match expected values.
    """
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
    """
    Tests the resolution of nested template strings within a multi-pass dictionary configuration.

    This test ensures that the `resolve_templates` function correctly resolves
    multi-level template variables in the dictionary configuration, where outer
    values depend on inner resolved template strings.

    Raises:
        AssertionError: If the resolved values do not match the expected outcome.
    """
    cfg: dict[str, Any] = {
        "a": {"x": "hello"},
        "b": {"y": "{a.x} world"},
        "c": {"z": "{b.y}!!!"},
    }

    out = resolve_templates(cfg)
    assert out["c"]["z"] == "hello world!!!"


def test_resolve_templates_missing_key_raises_keyerror() -> None:
    """
    Tests the resolve_templates function to ensure that it raises a KeyError when a
    template references a missing key in the configuration.

    Raises
    ------
    KeyError
        If a template references a missing key in the configuration.
    """
    cfg: dict[str, Any] = {"paths": {"input_dir": "inputs"}, "x": "{paths.missing}"}
    with pytest.raises(KeyError):
        resolve_templates(cfg)


def test_resolve_templates_unresolved_placeholders_raise_valueerror() -> None:
    """
    Tests that attempting to resolve templates with unresolved placeholders
    raises a ValueError. This ensures that circular or not-fully-resolvable
    templates are appropriately handled by raising an error.

    Raises:
        ValueError: If unresolved placeholders remain after the specified
            number of resolution passes.
    """
    # This simulates a circular or not-fully-resolvable template that survives passes
    cfg: dict[str, Any] = {"a": "{b}", "b": "{a}"}
    with pytest.raises(ValueError, match="Unresolved"):
        resolve_templates(cfg, max_passes=3)


# ---------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------
def test_load_cfg_reads_yaml_and_resolves_templates(tmp_path: Path) -> None:
    """
    Tests the `load_cfg` function to ensure loading of a YAML configuration file
    and resolving of templates within the file.

    Ensures the functionality of reading, interpreting, and substituting template
    placeholders using provided input in the YAML file.

    Parameters:
        tmp_path (Path): Temporary directory path for creating test files.

    Raises:
        AssertionError: If the loaded configuration does not resolve template
        placeholders correctly or does not match the expected values.
    """
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
    """
    Tests the `load_cfg` function for scenarios where the file is missing.

    This test ensures that the `load_cfg` function raises a `FileNotFoundError`
    when attempting to load a configuration file that does not exist.

    Parameters:
    tmp_path (Path): A temporary directory path provided by pytest.

    Raises:
    FileNotFoundError: If the specified configuration file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        load_cfg(tmp_path / "does_not_exist.yaml")


def test_load_cfg_non_mapping_raises(tmp_path: Path) -> None:
    """
    Tests that attempting to load a YAML configuration file that contains a list
    instead of a mapping raises a ValueError. The specific error message "Config
    must be a YAML mapping" is expected.

    Args:
        tmp_path (Path): Temporary directory path provided by pytest fixtures.

    Raises:
        ValueError: If the YAML configuration file does not contain a mapping.
    """
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")  # YAML list, not mapping
    with pytest.raises(ValueError, match="Config must be a YAML mapping"):
        load_cfg(p)


# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------
def test_resolve_input_path_relative_joins_raw_dir(tmp_path: Path) -> None:
    """
    Tests the resolution of input paths containing relative joins in the raw directory.

    This test verifies that the `resolve_input_path` function correctly resolves
    the paths based on the provided raw directory from the build context when
    given a relative input.

    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for testing
            purposes.

    Raises:
        AssertionError: If the resolved path does not match the expected result.
    """
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
    """
    Tests the `resolve_input_path` function for correctly returning an absolute path
    when it is already passed as an absolute path. This ensures that the function
    behaves as a passthrough for absolute paths without modification.

    Parameters:
    tmp_path (Path): A temporary directory provided by pytest for testing.

    """
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
    """
    Tests that resolving an input path with an unresolved template raises an exception.

    The function verifies the behavior of input path resolution when the template string
    is not properly replaced with actual values. It ensures that attempting to resolve
    an invalid template raises a `ValueError` with the expected error message.

    Parameters:
        tmp_path (Path): A temporary directory path provided by pytest to simulate
        file system interactions.

    Raises:
        ValueError: If the input path contains unresolved template strings.
    """
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
    """
    Tests the basic functionality of the create_age_range_lookup function.

    This test case verifies that the function correctly creates an age range lookup
    dictionary based on the provided parameters. It ensures that the generated lookup
    associates specific ages with the correct age range string and correctly assigns
    ages falling outside the defined range to a default category.

    Raises:
        AssertionError: If the generated lookup does not match the expected age range
        mappings or if the default category is not assigned correctly.

    """
    lookup = create_age_range_lookup(min_age=0, max_age=10, range_size=5)
    assert lookup[0] == "0-4"
    assert lookup[4] == "0-4"
    assert lookup[5] == "5-9"
    assert lookup[9] == "5-9"
    assert lookup[10] == "10+"  # default category


def test_create_age_range_lookup_with_under_min() -> None:
    """
    Tests the creation of an age range lookup table ensuring correct assignment
    of age group labels for given ages. The function specifically checks the
    behavior when ages provided are under the minimum specified value and
    whether the lookup table correctly categorizes these edge cases.

    Raises:
        AssertionError:
            If the constructed lookup table does not match the expected
            categories for the given ages.

    Returns:
        None
    """
    lookup = create_age_range_lookup(min_age=2, max_age=10, range_size=5)
    assert lookup[0] == "0-2"
    assert lookup[1] == "0-2"
    assert lookup[2] == "2-6" or lookup[2] == "2-6"  # depends on range definition
    assert lookup[10] == "10+"


def test_make_calendar_period_lookup_edges() -> None:
    """
    Tests the functionality of the make_calendar_period_lookup function to ensure it properly
    creates a lookup for calendar periods and handles edge cases.

    This test validates that:
    1. The generated lookup contains the correct mappings between years and calendar
       period ranges.
    2. The lookup correctly associates boundaries within a calendar period.
    3. Years outside of the defined calendar periods are appropriately excluded.

    Raises:
        AssertionError: If the lookup does not match the expected functionality.
    """
    lookup = make_calendar_period_lookup()
    assert 1950 in lookup
    assert lookup[1950] == "1950-1954"
    assert lookup[1954] == "1950-1954"
    assert lookup[1955] == "1955-1959"
    assert 1949 not in lookup
