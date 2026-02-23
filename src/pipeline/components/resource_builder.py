from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import json
import pandas as pd


@dataclass(frozen=True)
class BuildContext:
    """
    Context shared across all builders.

    This keeps every pipeline stage consistent:
      - cfg: parsed YAML for country (e.g. config/mw.yaml)
      - country: ISO-ish code used in your folder layout ("mw", "tz", ...)
      - raw_dir: root raw data directory
      - resources_dir: root resources directory
      - component: logical component name (e.g. "demography")
      - output_dir: computed as resources_dir / country / component
    """
    cfg: Mapping[str, Any]
    country: str
    raw_dir: Path
    resources_dir: Path
    component: str

    @property
    def output_dir(self) -> Path:
        return self.resources_dir / self.country / self.component


@dataclass(frozen=True)
class ResourceArtifact:
    """
    Single-produced artifact (usually a CSV).

    name: file name, e.g. "ResourceFile_Population_2010.csv"
    path: absolute/relative path to artifact
    rows/cols: optional metadata for quick inspection
    """
    name: str
    path: Path
    rows: Optional[int] = None
    cols: Optional[int] = None


class ResourceBuilder:
    """
    Base class for any script that turns raw data into TLO resourcefiles.

    Lifecycle:
        preflight -> load_data -> build -> validate -> write -> write_manifest

    Subclasses implement:
      - load_data(): read raw sources / inputs into in-memory objects (DataFrames, dicts, etc.)
      - build(raw): transform loaded inputs into dict[str, pd.DataFrame]
      - validate(outputs): assert invariants (row sums, keys, coverage, etc.)

    The base class provides:
      - consistent output dirs
      - writing CSVs with stable conventions
      - manifest writing for downstream discovery (reporting/tests)
    """

    # Override in subclass
    COMPONENT: str = "unknown_component"

    # Optional: enforce required inputs existence before build()
    REQUIRED_INPUTS: Sequence[str] = ()

    # Optional: predictable output filenames (helps tests/reports)
    EXPECTED_OUTPUTS: Sequence[str] = ()

    def __init__(self, ctx: BuildContext, *, dry_run: bool = False) -> None:
        self.ctx = ctx
        self.dry_run = dry_run

    # ---------- lifecycle API ----------

    def run(self) -> List[ResourceArtifact]:
        """
        Orchestrates the full lifecycle:
          1) preflight checks
          2) load raw data
          3) build outputs
          4) validate
          5) write outputs + manifest
        """
        self.preflight()

        raw = self.load_data()
        outputs = self.build(raw)

        df_map = self._normalize_outputs(outputs)
        self.validate(df_map)

        artifacts = self.write(df_map)
        self.write_manifest(artifacts)

        return artifacts

    def preflight(self) -> None:
        """Lightweight checks before doing any heavy work."""
        out = self.ctx.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Optional required raw inputs
        missing: List[str] = []
        for rel in self.REQUIRED_INPUTS:
            p = self.ctx.raw_dir / rel
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                "Missing required raw inputs:\n" + "\n".join(f"- {m}" for m in missing)
            )

    # ---------- subclass hooks ----------

    def load_data(self) -> Mapping[str, Any]:
        """
        Read raw inputs from ctx.raw_dir (or elsewhere) and return them as a mapping.

        Recommended pattern:
            return {
                "a1": pd.read_excel(...),
                "wpp_births": births_df,
                "lookup": some_dict,
                ...
            }

        Why Mapping[str, Any]?
        - different builders will load different kinds of inputs
        - this keeps build(raw) explicit and testable

        Subclasses must implement this if they use raw inputs.
        If a builder does not require raw inputs, it can return {}.
        """
        raise NotImplementedError

    def build(self, raw: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        """
        Transform loaded inputs into a mapping of output filename -> DataFrame.

        Recommended: return dict[filename, dataframe].
        """
        raise NotImplementedError

    def validate(self, outputs: Mapping[str, pd.DataFrame]) -> None:
        """
        Assert invariants. Keep these strict. Fail fast.

        Examples:
          - Sum of districts equals national totals
          - Region groupby sums match reported region totals
          - Age fractions sum to 1.0 (within tolerance)
          - Key columns contain no nulls
          - expected outputs exist
        """
        # Default: just enforce EXPECTED_OUTPUTS if declared
        if self.EXPECTED_OUTPUTS:
            missing = set(self.EXPECTED_OUTPUTS).difference(outputs.keys())
            if missing:
                raise AssertionError(f"Expected outputs missing: {sorted(missing)}")

    # ---------- base implementations ----------

    def write(self, outputs: Mapping[str, pd.DataFrame]) -> List[ResourceArtifact]:
        """Write CSVs to ctx.output_dir with consistent conventions."""
        artifacts: List[ResourceArtifact] = []
        for name, df in outputs.items():
            path = self.ctx.output_dir / name
            if not self.dry_run:
                df.to_csv(path, index=False)
            artifacts.append(
                ResourceArtifact(name=name, path=path, rows=len(df), cols=len(df.columns))
            )
        return artifacts

    def write_manifest(self, artifacts: Sequence[ResourceArtifact]) -> None:
        """
        Write a machine-readable manifest that reporting/tests can consume.
        """
        manifest = {
            "country": self.ctx.country,
            "component": self.ctx.component,
            "builder": type(self).__name__,
            "artifacts": [
                {"name": a.name, "path": str(a.path), "rows": a.rows, "cols": a.cols}
                for a in artifacts
            ],
        }
        path = self.ctx.output_dir / "resource_manifest.json"
        if not self.dry_run:
            path.write_text(json.dumps(manifest, indent=2))

    # ---------- helpers ----------

    def _normalize_outputs(self, outputs: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
        if isinstance(outputs, Mapping):
            return outputs
        raise TypeError(
            "build() must return dict[str, pd.DataFrame]. "
            "If you need non-dataframe artifacts, override write()/write_manifest()."
        )