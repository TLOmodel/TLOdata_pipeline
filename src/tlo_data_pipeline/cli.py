from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="tlo-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="Print version information")

    args = parser.parse_args()

    if args.cmd == "version":
        print("tlo-data-pipeline 0.1.0")
