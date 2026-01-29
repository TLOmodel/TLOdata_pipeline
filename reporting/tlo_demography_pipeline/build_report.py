from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any, Dict

from .artifacts import discover_resources, load_all
from .checks import run_all_checks


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return None


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_report(country: str, resources_dir: Path, reports_dir: Path) -> Path:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_out = reports_dir / country / "latest"

    artifacts = discover_resources(resources_dir=resources_dir, country=country)
    df_map = load_all(artifacts)

    check_results, ok = run_all_checks(df_map)

    # metadata
    meta = {
        "country": country,
        "run_timestamp": run_ts,
        "git_hash": _git_hash(),
        "resource_count": len(artifacts),
        "resources": [{"name": a.name, "path": str(a.path)} for a in artifacts],
        "checks": [{"name": r.name, "status": r.status, "message": r.message} for r in check_results],
    }
    _write_json(base_out / "run_metadata.json", meta)

    # markdown
    lines = []
    lines.append(f"# Demography pipeline report: `{country}`\n")
    lines.append(f"- Run time: {run_ts}")
    if meta["git_hash"]:
        lines.append(f"- Git commit: `{meta['git_hash']}`")
    lines.append(f"- Discovered resource CSVs: **{len(artifacts)}**\n")

    # checks summary
    n_pass = sum(r.status == "PASS" for r in check_results)
    n_warn = sum(r.status == "WARN" for r in check_results)
    n_fail = sum(r.status == "FAIL" for r in check_results)
    n_skip = sum(r.status == "SKIP" for r in check_results)
    lines.append("## QA checks\n")
    lines.append(f"- PASS: {n_pass} | WARN: {n_warn} | FAIL: {n_fail} | SKIP: {n_skip}\n")

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
    ap = argparse.ArgumentParser(description="Generate demography QA report (reads built resources)")
    ap.add_argument("--country", required=True)
    ap.add_argument("--resources-dir", default="resources")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    summary = build_report(
        country=args.country,
        resources_dir=Path(args.resources_dir),
        reports_dir=Path(args.reports_dir),
    )
    print(f"[ok] Wrote report: {summary}")


if __name__ == "__main__":
    main()
