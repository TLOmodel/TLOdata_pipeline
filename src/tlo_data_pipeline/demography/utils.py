#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union
import re

import yaml


_PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_.]+)}")


def _get_by_dotted_key(cfg: Dict[str, Any], key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key for template: {key}")
        cur = cur[part]
    return cur


def resolve_templates(cfg: Dict[str, Any], max_passes: int = 5) -> Dict[str, Any]:
    """Resolve {a.b.c} placeholders inside strings using values from cfg."""
    def resolve_value(v: Any) -> Any:
        if isinstance(v, str):
            def repl(m: re.Match) -> str:
                return str(_get_by_dotted_key(cfg, m.group(1)))
            return _PLACEHOLDER.sub(repl, v)
        if isinstance(v, list):
            return [resolve_value(x) for x in v]
        if isinstance(v, dict):
            return {k: resolve_value(x) for k, x in v.items()}
        return v

    for _ in range(max_passes):
        new_cfg = resolve_value(cfg)
        if new_cfg == cfg:
            break
        cfg = new_cfg
    return cfg


def load_cfg(path: Union[str, Path] = "config/pipeline_setup.yaml") -> Dict[str, Any]:
    """
    Load YAML config from path and resolve any {templates}.
    Usage: cfg = load_cfg("config/mw.yaml")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping/object, got: {type(cfg)}")

    return resolve_templates(cfg)


def load_regions(cfg: Dict[str, Any]) -> List[str]:
    """
    Return regions list from:
      1) cfg['census']['regions'] (inline list), else
      2) cfg['census']['regions_file'] (CSV/TXT), else
      3) error
    """
    census = cfg.get("census", {})
    if not isinstance(census, dict):
        raise ValueError("cfg['census'] must be a mapping")

    # 1) Inline list
    regions = census.get("regions")
    if regions is not None:
        if not isinstance(regions, list) or not all(isinstance(x, str) for x in regions):
            raise ValueError("census.regions must be a list of strings")
        return [r.strip() for r in regions if r.strip()]

    # 2) External file
    regions_file = census.get("regions_file")
    if regions_file:
        p = Path(str(regions_file))
        if not p.exists():
            raise FileNotFoundError(f"Regions file not found: {p}")

        if p.suffix.lower() in [".csv"]:
            col = str(census.get("regions_col", "Region"))
            df = pd.read_csv(p)
            if col not in df.columns:
                raise ValueError(f"Regions CSV missing column '{col}': {p}")
            vals = df[col].dropna().astype(str).tolist()
            vals = [v.strip() for v in vals if v.strip()]
            if not vals:
                raise ValueError(f"No regions found in {p} column '{col}'")
            return vals

        if p.suffix.lower() in [".txt"]:
            vals = [line.strip() for line in p.read_text().splitlines()]
            vals = [v for v in vals if v and not v.startswith("#")]
            if not vals:
                raise ValueError(f"No regions found in {p}")
            return vals

        raise ValueError(f"Unsupported regions_file extension: {p.suffix} (use .csv or .txt)")

    raise ValueError("No regions provided. Set census.regions or census.regions_file.")
