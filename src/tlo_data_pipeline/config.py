# src/tlo_data_pipeline/config.py
from __future__ import annotations

from pathlib import Path
import yaml


def load_config(
    country: str,
    config_dir: Path | None = None,
) -> dict:
    """
    Load and merge pipeline configuration.

    Precedence:
      1. Global defaults (pipeline_setup.yaml)
      2. Country overrides (countries/<country>.yaml)
    """
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[2] / "config"

    base_cfg_path = config_dir / "pipeline_setup.yaml"
    country_cfg_path = config_dir / "countries" / f"{country}.yaml"

    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_cfg_path}")

    if not country_cfg_path.exists():
        raise FileNotFoundError(f"Missing country config: {country_cfg_path}")

    with base_cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    with country_cfg_path.open() as f:
        country_cfg = yaml.safe_load(f)

    # Simple shallow merge (country overrides base)
    cfg.update(country_cfg)

    return cfg
