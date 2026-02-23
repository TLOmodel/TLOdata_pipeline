"""
Report generator for demographic data based on specified country and resources.

This module includes functions to discover resources, perform quality checks, and produce
reports for demographic pipelines. Results are saved in JSON and Markdown formats, providing
metadata and summaries of findings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tlo.methods.artifacts import discover_resources, load_all
from tlo.methods.checks import run_all_checks


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except subprocess.CalledProcessError:
        return None


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_report(resources_dir: Path, reports_dir: Path) -> Path:
    """
    Builds a report by processing discovered resources, running quality checks, and
    generating metadata and markdown outputs summarizing the results. The function
    handles data discovery, validation, metadata generation, and formatting for output.

    Parameters:
    resources_dir (Path): Path to the directory containing input resources.
    reports_dir (Path): Path to the directory where output reports are generated.

    Returns:
    Path: Path to the generated summary Markdown file.

    Raises:
    SystemExit: If hard-fail QA checks are detected.
    """
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_out = reports_dir / "latest"

    artifacts = discover_resources(resources_dir=resources_dir)
    df_map = load_all(artifacts)

    check_results, ok = run_all_checks(df_map)

    # metadata
    meta = {
        "run_timestamp": run_ts,
        "git_hash": _git_hash(),
        "resource_count": len(artifacts),
        "resources": [{"name": a.name, "path": str(a.path)} for a in artifacts],
        "checks": [
            {"name": r.name, "status": r.status, "message": r.message} for r in check_results
        ],
    }
    _write_json(base_out / "run_metadata.json", meta)

    # markdown
    lines = []
    lines.append("# Demography pipeline report\n")
    lines.append(f"- Run time: {run_ts}")
    if meta["git_hash"]:
        lines.append(f"- Git commit: `{meta['git_hash']}`")
    lines.append(f"- Discovered resource CSVs: **{len(artifacts)}**\n")

    # checks summary
    out_warn = {
        "n_pass": sum(r.status == "PASS" for r in check_results),
        "n_warn": sum(r.status == "WARN" for r in check_results),
        "n_fail": sum(r.status == "FAIL" for r in check_results),
        "n_skip": sum(r.status == "SKIP" for r in check_results),
    }
    lines.append("## QA checks\n")
    lines.append(
        f"- PASS: {out_warn['n_pass']} | WARN: {out_warn['n_warn']} "
        f"| FAIL: {out_warn['n_fail']} | SKIP: {out_warn['n_skip']}\n"
    )

    lines.append("| Check | Status | Message |")
    lines.append("|---|---:|---|")
    for r in check_results:
        lines.append(f"| `{r.name}` | **{r.status}** | {r.message} |")
    lines.append("")

    summary_path = base_out / "summary.md"
    _write_md(summary_path, "\n".join(lines))

    if not ok:
        raise SystemExit("Hard-fail QA checks found. See reports summary for details.")

    return summary_path


def main() -> None:
    """
    Main function for generating a demographic QA report based on specified parameters.

    The script reads built resources and generates a QA report for demography data.
    It takes command-line arguments to specify the country to process, input resources
    directory, and output reports directory.

    Parameters:
        --country (str): The target country for the report. This argument is required.
        --resources-dir (str): The base directory where the resource files are located.
            Defaults to "resources".
        --reports-dir (str): The output directory for the generated report. Defaults to "reports".

    Raises:
        SystemExit: Raised when required command-line arguments are missing or invalid.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(
        description="Generate demography QA report (reads built resources)"
    )
    ap.add_argument("--country", required=True)
    ap.add_argument("--resources-dir", default="resources")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    summary = build_report(
        resources_dir=Path(args.resources_dir),
        reports_dir=Path(args.reports_dir),
    )
    print(f"[ok] Wrote report: {summary}")


if __name__ == "__main__":
    main()
