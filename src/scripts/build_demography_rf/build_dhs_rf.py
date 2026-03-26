#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.components.demography.dhs import DHSBuilder
from pipeline.components.framework.builder import BuildContext
from pipeline.components.framework.utils import load_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    ctx = BuildContext(
        cfg=cfg,
        country=cfg["country_code"],
        input_dir=Path(cfg["paths"]["input_dir"])/"demography",
        resources_dir=Path(cfg["paths"]["resources_dir"]),
        component="demography",
    )

    artifacts = DHSBuilder(ctx, dry_run=args.dry_run).run()
    for a in artifacts:
        print(f"[OK] wrote {a.name} -> {a.path} ({a.rows}x{a.cols})")


if __name__ == "__main__":
    main()
