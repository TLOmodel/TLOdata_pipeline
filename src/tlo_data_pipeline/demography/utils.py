#!/usr/bin/env python3
"""
This module provides functionality to load YAML configuration files and resolve
placeholders within the configuration dictionary using template resolution.

The module defines methods to fetch nested configuration values via dotted keys,
resolve placeholders in configuration files, and load configuration files
from the filesystem while handling template substitutions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_.]+)}")


def _get_by_dotted_key(cfg: dict[str, Any], key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key for template: {key}")
        cur = cur[part]
    return cur


def resolve_templates(cfg: dict[str, Any], max_passes: int = 5) -> dict[str, Any]:
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


def load_cfg(path: str | Path = "config/pipeline_setup.yaml") -> dict[str, Any]:
    """
    Load YAML config from path and resolve any {templates}.
    Usage: cfg = load_cfg("config/mw.yaml")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping/object, got: {type(cfg)}")

    return resolve_templates(cfg)
