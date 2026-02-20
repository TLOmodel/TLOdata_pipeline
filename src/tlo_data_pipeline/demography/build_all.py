#!/usr/bin/env python3
"""
This script builds demographic resources (census, DHS, WPP) for a specified country and optionally
generates a quality assurance report. It uses resource-specific scripts and allows configuration
of directories for inputs and outputs. The script is executed via the command line with several
available arguments.

Functions:
    main: Responsible for orchestrating the building of resources and the optional
    report generation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print(f"\n[run] {' '.join(cmd)}")
    completed = subprocess.run(cmd, text=True, check=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    """
    Main function responsible for building demographic resources and optionally generating
    a quality assurance (QA) report. The function uses existing scripts for building census,
    DHS, and WPP resources based on the specified country code. It also allows generating
    a report that reads outputs from a resources folder and writes to a specified reports
    directory.

    Parameters:
        --country (str): Country code, e.g., "mw" or "tz". This parameter is required.
        --reports-dir (str): Directory where the generated report outputs will be written.
            Defaults to "reports/".
        --resources-dir (str): Base directory for resources used in scripts. Defaults to
            "resources/".
        --skip-report (bool): Flag indicating whether to skip the report generation step.
            Default is False.

    Raises:
        SystemExit: Raised by `argparse.ArgumentParser` if provided arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Build demography resources (census, dhs, wpp) and generate a QA report."
    )
    parser.add_argument("--country", required=True, help="Country code, e.g. mw or tz")
    parser.add_argument(
        "--reports-dir",
        default="outputs/reports",
        help="Where to write generated report outputs (default: reports/)",
    )
    parser.add_argument(
        "--resources-dir",
        default="outputs/resources",
        help="Base resources folder used by your existing scripts (default: resources/)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Only build resources, do not generate report.",
    )
    args, unknown = parser.parse_known_args()

    # 1) Build resources using your existing scripts (pass-through unknown args if
    #    you already use them)
    _run([sys.executable, "src/tlo_data_pipeline/demography/census.py", "--country",
          args.country, *unknown])
    _run([sys.executable, "src/tlo_data_pipeline/demography/dhs.py", "--country",
          args.country, *unknown])
    _run([sys.executable, "src/tlo_data_pipeline/demography/wpp.py", "--country",
          args.country, *unknown])

    if args.skip_report:
        print("\n[ok] Resources built; report skipped.")
        return

    # 2) Generate report (reads outputs from resources folder)
    _run(
        [
            sys.executable,
            "-m",
            "src.tlo_data_pipeline.demography.reporting.build_report",
            "--country",
            args.country,
            "--resources-dir",
            args.resources_dir,
            "--reports-dir",
            args.reports_dir,
        ]
    )

    print("\n[ok] Done.")


if __name__ == "__main__":
    main()
