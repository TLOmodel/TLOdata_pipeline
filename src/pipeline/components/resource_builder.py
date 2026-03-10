"""
A template for all classes that builds raw files into TLO resources.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BuildContext:
    """
    Context shared across all builders.

    - cfg: parsed YAML for country (e.g. config/tz.yaml)
    - country: ISO-like code ("tz", "mw", etc.) — informational only
    - raw_dir: root raw data directory
    - resources_dir: root resources directory
    - component: logical component name (e.g. "demography")
    - output_dir: computed as resources_dir / component
    """

    cfg: Mapping[str, Any]
    country: str
    raw_dir: Path
    resources_dir: Path
    component: str

    @property
    def output_dir(self) -> Path:
        """
        Returns the output directory associated with the current component.

        The output directory is derived by joining the resources directory and the
        component name. This property provides a convenient way of accessing the
        computed directory path.

        Returns:
            Path: The output directory path corresponding to the current component.
        """
        return self.resources_dir / self.component


@dataclass(frozen=True)
class ResourceArtifact:
    """
    Represents a single produced artifact (usually a CSV).
    """

    name: str
    path: Path
    rows: int | None = None
    cols: int | None = None


class ResourceBuilder:
    """
    Base class for scripts that transform raw data into TLO resource files.

    Lifecycle:
        preflight -> load_data -> build -> validate -> write -> write_manifest
    """

    COMPONENT: str = "unknown_component"

    # Relative to ctx.raw_dir
    REQUIRED_INPUTS: Sequence[str] = ()

    # Unconditional expected outputs
    EXPECTED_OUTPUTS: Sequence[str] = ()

    def __init__(self, ctx: BuildContext, *, dry_run: bool = False) -> None:
        self.ctx = ctx
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> list[ResourceArtifact]:
        """
        Runs the complete processing workflow including preflight checks, data loading,
        transformation, validation, and writing the final output along with its manifest.

        Returns
        -------
        list[ResourceArtifact]
            A list of ResourceArtifact objects representing the processed and written
            data resources.
        """

        self.preflight()

        raw = self.load_data()
        outputs = self.build(raw)

        df_map = self.normalize_outputs(outputs)
        self.validate(df_map)

        artifacts = self.write(df_map)
        self.write_manifest(artifacts)

        return artifacts

    def preflight(self) -> None:
        """Ensure output directory exists and required inputs are present."""
        self.ctx.output_dir.mkdir(parents=True, exist_ok=True)

        missing: list[str] = []
        for rel in self.REQUIRED_INPUTS:
            p = self.ctx.raw_dir / rel
            if not p.exists():
                missing.append(str(p))

        if missing:
            raise FileNotFoundError(
                "Missing required raw inputs:\n" + "\n".join(f"- {m}" for m in missing)
            )

    # ------------------------------------------------------------------
    # Hooks to implement in subclasses
    # ------------------------------------------------------------------

    def load_data(self) -> Mapping[str, Any]:
        """
        Loads data from a specified source.

        This method should be implemented by subclasses to define the logic
        for fetching or loading data. Calling it in its current state will raise
        a NotImplementedError.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Returns:
            Mapping[str, Any]: The loaded data.
        """
        raise NotImplementedError

    def build(self, raw: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        """
        Builds a structured data representation based on the provided raw data.

        This method is expected to take a raw input, process it, and return a
        mapping where the keys are strings, and the values are pandas DataFrame
        objects. Custom implementations must override this method and provide
        the specific logic for building the resulting data structure.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Parameters
        ----------
        raw : Mapping[str, Any]
            The input data to be processed, represented as a mapping where
            keys are strings and values can be of any type.

        Returns
        -------
        Mapping[str, pd.DataFrame]
            A mapping where keys are strings, and values are pandas DataFrame
            objects that represent the processed and structured data.
        """
        raise NotImplementedError

    def validate(self, outputs: Mapping[str, pd.DataFrame]) -> None:
        """
        Validates the provided outputs against the expected outputs specified by the class. If any
        expected output keys are missing from the actual outputs, an AssertionError is raised.

        Parameters
        ----------
        outputs : Mapping[str, pd.DataFrame]
            A mapping of output names to their corresponding data frames. The keys should
            match the expected outputs defined by the class.

        Raises
        ------
        AssertionError
            If any of the expected outputs are missing from the provided outputs.
        """
        if self.EXPECTED_OUTPUTS:
            missing = set(self.EXPECTED_OUTPUTS).difference(outputs.keys())
            if missing:
                raise AssertionError(f"Expected outputs missing: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Base implementations
    # ------------------------------------------------------------------

    def write(self, outputs: Mapping[str, pd.DataFrame]) -> list[ResourceArtifact]:
        """
        Writes the given dataframes to CSV files and generates associated artifacts.
        The dataframes are written under the output directory specified by the context.
        Artifacts, detailing information about the written files, such as file name,
        path, row count, and column count, are created for each dataframe and returned.

        Parameters:
            outputs: Mapping[str, pd.DataFrame]
                A mapping where the key represents the name of the output file and the
                value is a pandas DataFrame to be written.

        Returns:
            list[ResourceArtifact]
                A list of ResourceArtifact objects representing metadata about the
                written files.

        Raises:
            None
        """
        artifacts: list[ResourceArtifact] = []

        for name in sorted(outputs.keys()):
            df = outputs[name]
            path = self.ctx.output_dir / name

            if not self.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)

            artifacts.append(
                ResourceArtifact(
                    name=name,
                    path=path,
                    rows=len(df),
                    cols=len(df.columns),
                )
            )

        return artifacts

    def write_manifest(self, artifacts: Sequence[ResourceArtifact]) -> None:
        """
        Writes a manifest file containing metadata about provided artifacts.

        The method generates a JSON-structured manifest that represents metadata
        of the provided sequence of resource artifacts. This metadata includes
        name, path, rows, and columns of each artifact. The manifest is written
        to a file within the output directory specified in the context, unless
        the dry_run flag is set to True.

        Parameters:
            artifacts (Sequence[ResourceArtifact]): A sequence of resource artifacts
            containing metadata such as name, path, rows, and columns.

        Raises:
            None
        """
        manifest = {
            "component": self.ctx.component,
            "builder": type(self).__name__,
            "artifacts": [
                {
                    "name": a.name,
                    "path": str(a.path),
                    "rows": a.rows,
                    "cols": a.cols,
                }
                for a in artifacts
            ],
        }

        path = self.ctx.output_dir / f"resource_manifest_{type(self).__name__}.json"

        if not self.dry_run:
            path.write_text(json.dumps(manifest, indent=2))

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def normalize_outputs(self, outputs: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
        """
        Normalizes the outputs by validating their types and formats.

        Ensures the provided outputs are of type Mapping with string keys and pandas DataFrame
        values. Raises errors if the validations fail. Returns the original outputs if they
        meet the criteria.

        Args:
            outputs: A Mapping object containing string keys and pandas DataFrame values.

        Returns:
            A Mapping object containing string keys and pandas DataFrame values that pass the
            validation.

        Raises:
            TypeError: If the outputs are not a Mapping, if any keys are not strings, or
            if any values are not pandas DataFrames.
        """
        if not isinstance(outputs, Mapping):
            raise TypeError("build() must return dict[str, pd.DataFrame].")

        bad_keys = [k for k in outputs if not isinstance(k, str)]
        if bad_keys:
            raise TypeError(f"Output keys must be strings. Invalid keys: {bad_keys}")

        bad_vals = [k for k, v in outputs.items() if not isinstance(v, pd.DataFrame)]
        if bad_vals:
            raise TypeError(f"Output values must be pandas DataFrames. Invalid keys: {bad_vals}")

        return outputs
