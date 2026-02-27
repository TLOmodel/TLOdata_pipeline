#!/usr/bin/env python3
"""
Utilities for:
  - loading YAML configs (with {a.b.c} template resolution)
  - resolving input paths consistently in ResourceBuilder-based pipelines
  - common demography lookups (age ranges, calendar periods)

Conventions:
  1) All runners MUST call load_cfg() so templates are resolved uniformly.
  2) All file paths in YAML SHOULD be relative to paths.input_dir (ctx.raw_dir).
     Absolute paths are allowed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from pipeline.components.resource_builder import BuildContext

_PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_.]+)}")


# ---------------------------------------------------------------------
# Config loading + template resolution
# ---------------------------------------------------------------------
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

    # Fail fast if templates still remain (usually indicates a missing key)
    def _has_placeholders(v: Any) -> bool:
        if isinstance(v, str):
            return bool(_PLACEHOLDER.search(v))
        if isinstance(v, list):
            return any(_has_placeholders(x) for x in v)
        if isinstance(v, dict):
            return any(_has_placeholders(x) for x in v.values())
        return False

    if _has_placeholders(cfg):
        raise ValueError(
            "Unresolved {templates} remain in config after resolution. "
            "Check for missing keys or circular references."
        )

    return cfg


def load_cfg(path: str | Path = "config/pipeline_setup.yaml") -> dict[str, Any]:
    """
    Load YAML config from path and resolve any {templates}.
    Usage: cfg = load_cfg("config/tz.yaml")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping/object, got: {type(cfg)}")

    return resolve_templates(cfg)


# ---------------------------------------------------------------------
# Path resolution helpers (ResourceBuilder)
# ---------------------------------------------------------------------
def resolve_input_path(ctx: BuildContext, value: str | Path) -> Path:
    """
    Resolve a config-defined input path.

    Contract:
      - Absolute paths are used as-is.
      - Relative paths are interpreted relative to ctx.raw_dir.
      - If a value still contains '{...}', it means load_cfg() wasn't used or YAML is not clean.

    This keeps behavior deterministic and avoids "double-prefix" or CWD-dependent resolution.
    """
    s = str(value).strip()
    if _PLACEHOLDER.search(s):
        raise ValueError(
            f"Unresolved template in path value: {s!r}. "
            "Ensure runners call load_cfg(), and "
            "YAML paths are not templated with {paths.input_dir}."
        )

    p = Path(s).expanduser()
    if p.is_absolute():
        return p

    return (ctx.raw_dir / p).resolve()


# ---------------------------------------------------------------------
# Demography lookups
# ---------------------------------------------------------------------
def create_age_range_lookup(min_age: int, max_age: int, range_size: int = 5) -> dict[int, str]:
    """
    Map each whole-year age -> age-range label.

    - If min_age > 0, ages < min_age map to "0-min_age"
    - Ages >= max_age map to f"{max_age}+"
    """

    def chunks(items, n):
        for index in range(0, len(items), n):
            yield items[index : index + n]

    parts = chunks(range(min_age, max_age), range_size)

    default_category = f"{max_age}+"
    lookup = defaultdict(lambda: default_category)

    if min_age > 0:
        under_min_age_category = f"0-{min_age}"
        for i in range(0, min_age):
            lookup[i] = under_min_age_category

    for part in parts:
        start = part.start
        end = part.stop - 1
        label = f"{start}-{end}"
        for i in range(start, part.stop):
            lookup[i] = label

    return lookup


def make_calendar_period_lookup() -> dict[int, str]:
    """Map calendar year -> 5-year period (e.g. 1950 -> '1950-1954')."""
    lookup = create_age_range_lookup(1950, 2100, 5)
    for year in range(1950):
        lookup.pop(year, None)
    return lookup
