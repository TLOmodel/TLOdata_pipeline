from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.components.demography.census import CensusBuilder
from pipeline.components.framework.builder import BuildContext
from pipeline.components.framework.utils import load_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # IMPORTANT: use load_cfg so {templates} are resolved
    cfg = load_cfg(args.cfg)

    ctx = BuildContext(
        cfg=cfg,
        country=str(cfg.get("country_code", "")),  # kept for compatibility (even if unused)
        input_dir=Path(cfg["paths"]["input_dir"])/"demography",
        resources_dir=Path(cfg["paths"]["resources_dir"]),
        component="demography",
    )

    artifacts = CensusBuilder(ctx, dry_run=args.dry_run).run()
    for a in artifacts:
        print(f"[OK] wrote {a.name} -> {a.path} ({a.rows}x{a.cols})")


if __name__ == "__main__":
    main()
