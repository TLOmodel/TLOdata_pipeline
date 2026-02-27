"""
Testing framework for the `_DummyBuilder` class and its integration with
a mock pipeline.

This module contains unit tests to ensure that `_DummyBuilder`, a subclass
of `ResourceBuilder`, behaves as expected. It evaluates methods like
`preflight`, `run`, and `validate` under different conditions, including
dry-run modes and error scenarios. The module also verifies type
enforcement and functional correctness in end-to-end scenarios.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.components.resource_builder import BuildContext, ResourceBuilder


class _DummyBuilder(ResourceBuilder):
    """
    Represents a dummy implementation of a resource builder.

    This class inherits from ResourceBuilder and provides dummy methods for
    loading data and building resources. It is designed to simulate the
    behavior of a resource builder with predetermined inputs and outputs.
    The class uses predefined constants and contains logic to generate
    outputs in an unsorted manner to test sorting functionality.

    Attributes:
        COMPONENT: Constant string representing the component name.
        REQUIRED_INPUTS: Tuple of strings specifying the required input
            filenames.
        EXPECTED_OUTPUTS: Tuple of strings specifying the expected output
            filenames.
    """

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
    """
    Tests that the preflight method of the builder correctly handles required input
    files and the creation of an output directory.

    Ensures that the method raises an exception when required input files are
    missing, and verifies proper functionality when the inputs are provided.

    Parameters:
    tmp_path: Path
        An instance of pathlib.Path provided by pytest's tmp_path fixture, used
        as the base temporary directory for testing.

    Raises:
    FileNotFoundError
        If the required input files are missing during the preflight check.
    """
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
    """
    Tests the functionality of the `_DummyBuilder` class, ensuring that it writes
    the expected CSV files and a resource manifest in JSON format during its run.

    Summary:
    This unit test verifies that the `_DummyBuilder` class correctly generates
    the output artifacts (CSV files) sorted by name, creates a valid JSON manifest
    file with the correct schema, and ensures all files are written to the expected
    locations.

    Parameters:
    tmp_path: Path
        A temporary directory path provided by the test framework to store test
        files and outputs during the test run.

    Raises:
    AssertionError
        If the expected output artifacts or manifest file do not match the expected
        conditions, including naming, existence, or JSON schema validation.
    """
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
    """
    Tests that the `run` method in the `_DummyBuilder` class does not write any files when
    executed in dry run mode and always returns the expected artifacts.

    Parameters:
    tmp_path (Path): A temporary path provided as a fixture for file system operations
                    during the test.

    Raises:
    AssertionError: If the artifacts are not as expected or if any file is written to the output
    directory during the dry run process.
    """
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
    """
    Tests the validate method to ensure it enforces the expected outputs
    and raises appropriate errors for missing expected outputs.

    Parameters:
    tmp_path: Path
        Temporary directory path provided by pytest for creating test files
        or directories.

    Raises:
    AssertionError
        If the expected outputs are not found in the provided outputs.
    """
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
    """
    Tests that the `_normalize_outputs` method enforces the requirement for the
    build method to return a mapping. This is achieved by verifying that a
    `TypeError` is raised when an input list, instead of a mapping, is passed
    to `_normalize_outputs`.

    Parameters
    ----------
    tmp_path : Path
        A temporary filesystem path provided as a fixture by pytest.

    Raises
    ------
    TypeError
        If the build method does not return a mapping. In this test case,
        the error arises when a list is passed instead of a mapping.
    """
    ctx = _ctx(tmp_path)
    b = _DummyBuilder(ctx)

    with pytest.raises(TypeError, match="build\\(\\) must return"):
        b.normalize_outputs(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_normalize_outputs_requires_string_keys(tmp_path: Path) -> None:
    """
    Tests that the `_normalize_outputs` function requires dictionary keys to be strings.

    This test ensures that when passing a dictionary with non-string keys to the
    `_normalize_outputs` method, the method will raise a `TypeError` with the appropriate
    error message.

    Parameters:
    tmp_path (Path): A temporary directory path generated by pytest for testing.

    Raises:
    TypeError: If the dictionary keys provided to `_normalize_outputs` are not strings.
    """
    ctx = _ctx(tmp_path)
    b = _DummyBuilder(ctx)

    bad = {1: pd.DataFrame()}  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="Output keys must be strings"):
        b.normalize_outputs(bad)  # type: ignore[arg-type]


def test_normalize_outputs_requires_dataframe_values(tmp_path: Path) -> None:
    """
    Tests the `_normalize_outputs` method for its behavior when non-DataFrame outputs
    are provided. Ensures that a `TypeError` is raised with the appropriate message
    if the output values are not pandas DataFrames.

    Parameters
    ----------
    tmp_path : Path
        Temporary path fixture provided by pytest.

    Raises
    ------
    TypeError
        If the output values are not pandas DataFrames.
    """
    ctx = _ctx(tmp_path)
    b = _DummyBuilder(ctx)

    bad = {"a.csv": "not df"}  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="Output values must be pandas DataFrames"):
        b.normalize_outputs(bad)  # type: ignore[arg-type]
