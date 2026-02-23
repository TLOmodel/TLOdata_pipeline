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
from collections import defaultdict
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


def create_age_range_lookup(min_age: int, max_age: int, range_size: int = 5) -> dict[int, str]:
    """Create age-range categories and a dictionary that will map all whole years to
    age-range categories

    If the minimum age is not zero then a below minimum age category will be made,
    then age ranges until maximum age will be made by the range size,
    all other ages will map to the greater than maximum age category.

    :param min_age: Minimum age for categories,
    :param max_age: Maximum age for categories, a greater than maximum age category will be made
    :param range_size: Size of each category between minimum and maximum ages
    :returns:
        age_categories: ordered list of age categories available
        lookup: Default dict of integers to maximum age mapping to the age categories
    """

    def chunks(items, n):
        """Takes a list and divides it into parts of size n"""
        for index in range(0, len(items), n):
            yield items[index : index + n]

    # split all the ages from min to limit
    parts = chunks(range(min_age, max_age), range_size)

    default_category = f"{max_age}+"
    lookup = defaultdict(lambda: default_category)
    age_categories = []

    # create category for minimum age
    if min_age > 0:
        under_min_age_category = f"0-{min_age}"
        age_categories.append(under_min_age_category)
        for i in range(0, min_age):
            lookup[i] = under_min_age_category

    # loop over each range and map all ages falling within the range to the range
    for part in parts:
        start = part.start
        end = part.stop - 1
        value = f"{start}-{end}"
        age_categories.append(value)
        for i in range(start, part.stop):
            lookup[i] = value

    age_categories.append(default_category)

    return lookup


def make_calendar_period_lookup():
    """Returns a dictionary mapping calendar year (in years) to five year period
    i.e. { 1950: '1950-1954', 1951: '1950-1954, ...}
    """

    # Recycles the code used to make age-range lookups:
    lookup = create_age_range_lookup(1950, 2100, 5)

    # Removes the '0-1950' category
    # ranges.remove('0-1950')

    for year in range(1950):
        lookup.pop(year)

    return lookup
