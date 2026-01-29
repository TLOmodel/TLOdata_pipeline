# src/tlo_data_pipeline/paths.py
from __future__ import annotations

from pathlib import Path

def project_root() -> Path:
    """
    Resolve project root assuming standard layout:
    project_root/
      src/tlo_data_pipeline/paths.py
    """
    return Path(__file__).resolve().parents[2]


def inputs_dir() -> Path:
    return project_root() / "inputs"


def outputs_dir() -> Path:
    return project_root() / "outputs"


def demography_inputs(domain: str) -> Path:
    return inputs_dir() / "demography" / domain


def demography_resources(country: str) -> Path:
    return outputs_dir() / "resources" / country / "demography"


def demography_reports(country: str) -> Path:
    return outputs_dir() / "reports" / country
