# src/tlo/paths.py
"""
Utilities for handling consistent directory and file path management
within the project structure.

This module provides functions to dynamically compute and retrieve
various directory and file paths under the project's directory layout
specified relative to the project root directory. It ensures flexibility
and portability by resolving paths based on the current file's location.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """
    Resolve project root assuming standard layout:
    project_root/
      src/tlo/paths.py
    """
    return Path(__file__).resolve().parents[2]


def inputs_dir() -> Path:
    """
    Returns the directory path for the inputs folder within the project.

    This function calculates the directory path based on the project root
    location. It ensures the correct path is always obtained dynamically
    relative to the root directory of the project.

    Returns:
        Path: The directory path for the inputs folder.
    """
    return project_root() / "inputs"


def outputs_dir() -> Path:
    """
    Returns the directory path for storing output files.

    The function determines the outputs directory by appending the "outputs"
    subdirectory to the project's root directory.

    Returns:
        Path: A Path object representing the outputs directory.
    """
    return project_root() / "outputs"


def demography_inputs(domain: str) -> Path:
    """
    Generates the file path for the demography inputs of a specific domain.

    This function constructs a Path object pointing to the required demography input
    directory for the given domain by concatenating the base inputs directory,
    the "demography" subdirectory, and the specific domain.

    Args:
        domain (str): The name of the specific domain for which the demography inputs
        are required.

    Returns:
        Path: A Path object representing the full path to the designated
        demography input directory.
    """
    return inputs_dir() / "demography" / domain


def demography_resources(country: str) -> Path:
    """
    Provides a file path to access demography resources for a specified
    country. This function is used to construct the directory path where
    demographic data for the given country is stored.

    Parameters:
    country: str
        The name of the country whose demography resources path is to be
        retrieved.

    Returns:
    Path
        A Path object indicating the location of the demography resources
        directory for the specified country.
    """
    return outputs_dir() / "resources" / country / "demography"


def demography_reports(country: str) -> Path:
    """
    Generates the file path for storing demography reports of a given country.

    The function concatenates a base directory path for outputs with specific
    subdirectories for "reports" and the specified country name to construct
    the full file path.

    Args:
        country (str): The name of the country for which the report file path
        is being generated.

    Returns:
        Path: The full file path as a Path object for the demography report
        of the specified country.
    """
    return outputs_dir() / "reports" / country
