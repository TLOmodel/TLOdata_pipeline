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
        self.preflight()

        raw = self.load_data()
        outputs = self.build(raw)

        df_map = self._normalize_outputs(outputs)
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
        raise NotImplementedError

    def build(self, raw: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        raise NotImplementedError

    def validate(self, outputs: Mapping[str, pd.DataFrame]) -> None:
        if self.EXPECTED_OUTPUTS:
            missing = set(self.EXPECTED_OUTPUTS).difference(outputs.keys())
            if missing:
                raise AssertionError(f"Expected outputs missing: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Base implementations
    # ------------------------------------------------------------------

    def write(self, outputs: Mapping[str, pd.DataFrame]) -> list[ResourceArtifact]:
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

    def _normalize_outputs(
        self, outputs: Mapping[str, pd.DataFrame]
    ) -> Mapping[str, pd.DataFrame]:

        if not isinstance(outputs, Mapping):
            raise TypeError("build() must return dict[str, pd.DataFrame].")

        bad_keys = [k for k in outputs if not isinstance(k, str)]
        if bad_keys:
            raise TypeError(f"Output keys must be strings. Invalid keys: {bad_keys}")

        bad_vals = [k for k, v in outputs.items() if not isinstance(v, pd.DataFrame)]
        if bad_vals:
            raise TypeError(
                f"Output values must be pandas DataFrames. Invalid keys: {bad_vals}"
            )

        return outputs