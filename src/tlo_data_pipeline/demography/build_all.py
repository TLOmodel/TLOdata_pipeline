#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print(f"\n[run] {' '.join(cmd)}")
    completed = subprocess.run(cmd, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build demography resources (census, dhs, wpp) and generate a QA report."
    )
    parser.add_argument("--country", required=True, help="Country code, e.g. mw or tz")
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Where to write generated report outputs (default: reports/)",
    )
    parser.add_argument(
        "--resources-dir",
        default="resources",
        help="Base resources folder used by your existing scripts (default: resources/)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Only build resources, do not generate report.",
    )
    args, unknown = parser.parse_known_args()

    # 1) Build resources using your existing scripts (pass-through unknown args if you already use them)
    _run([sys.executable, "scripts/census.py", "--country", args.country, *unknown])
    _run([sys.executable, "scripts/dhs.py", "--country", args.country, *unknown])
    _run([sys.executable, "scripts/wpp.py", "--country", args.country, *unknown])

    if args.skip_report:
        print("\n[ok] Resources built; report skipped.")
        return

    # 2) Generate report (reads outputs from resources folder)
    _run([
        sys.executable,
        "-m",
        "reporting.tlo_demography_pipeline.build_report",
        "--country",
        args.country,
        "--resources-dir",
        args.resources_dir,
        "--reports-dir",
        args.reports_dir,
    ])

    print("\n[ok] Done.")


if __name__ == "__main__":
    main()
