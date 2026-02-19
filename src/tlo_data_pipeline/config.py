"""
Load and merge pipeline configuration based on country.

This function loads a global default configuration along with a country-specific
override configuration file, merges them, and returns the consolidated output.

Precedence order during merging:
1. Global defaults from the pipeline_setup.yaml file.
2. Country-specific overrides in the countries/<country>.yaml file.

Arguments:
country (str): The specific country code for which the configuration should be loaded.
config_dir (Path or None, optional): The directory containing the configuration files.
    If not provided, a default path relative to the file structure is used.

Returns:
dict: The consolidated configuration dictionary.

Raises:
FileNotFoundError: If the global default configuration file or the country-specific
    configuration file is not found.
"""

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
