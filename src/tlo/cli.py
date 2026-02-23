"""
A command-line tool for the TLO data pipeline.

This module provides a command-line utility with multiple commands for
interacting with the TLO data pipeline. It includes functionality such as
printing version information.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """
    The main module of the script provides the command-line interface for interacting with
    the TLO Data Pipeline application. It supports various subcommands that allow users to
    perform specific tasks.

    Functions:
        main: The entry point of the script, responsible for setting up argument parsing
        and handling subcommands.

    Raises:
        SystemExit: Raised when invalid arguments are provided or CLI invocation is incorrect.
    """
    parser = argparse.ArgumentParser(prog="tlo-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="Print version information")

    args = parser.parse_args()

    if args.cmd == "version":
        print("tlo-data-pipeline 0.1.0")
