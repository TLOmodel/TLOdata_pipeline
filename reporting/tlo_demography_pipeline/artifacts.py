from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class ResourceArtifact:
    name: str
    path: Path


def _guess_country_demog_dir(resources_dir: Path, country: str) -> Path:
    # Adjust if your repo uses a different layout.
    # Common in your project: resources/demography/<country>/
    return resources_dir


def discover_resources(resources_dir: Path, country: str) -> list[ResourceArtifact]:
    base = _guess_country_demog_dir(resources_dir, country)
    if not base.exists():
        raise FileNotFoundError(f"Expected demography resources directory not found: {base}")

    artifacts: list[ResourceArtifact] = []
    for p in sorted(base.rglob("*.csv")):
        # name is relative path without extension (keeps uniqueness)
        rel = p.relative_to(base).as_posix()
        name = rel.removesuffix(".csv")
        artifacts.append(ResourceArtifact(name=name, path=p))
    return artifacts


def read_csv_safely(path: Path) -> pd.DataFrame:
    # Conservative defaults: keep strings, avoid dtype surprises
    df = pd.read_csv(path)
    return df


def load_all(artifacts: list[ResourceArtifact]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for a in artifacts:
        out[a.name] = read_csv_safely(a.path)
    return out
