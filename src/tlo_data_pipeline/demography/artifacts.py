"""
Provides functionality to manage and manipulate resource artifacts.

This module includes classes and functions to handle resource artifacts
associated with demography data. It facilitates discovering artifacts for a
specific country, safely reading CSV files, and loading all discovered
artifacts into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ResourceArtifact:
    """
    Represents an artifact in a resource.

    This class is used to encapsulate details about a specific resource artifact
    including its name and file path. It is immutable, ensuring the integrity of
    the data it holds once an instance is created.

    Attributes:
        name: The name of the resource artifact.
        path: The file system path where the artifact is located.
    """

    name: str
    path: Path


def _guess_country_demog_dir(resources_dir: Path) -> Path:
    """
    Determines the directory for a country's demography data based on the provided
    resources directory and country name. This function assumes a specific
    directory layout used in the project.

    Args:
        resources_dir (Path): The base directory where resource files are located.
        country (str): The name of the country whose demography directory is to
            be guessed.

    Returns:
        Path: The guessed path to the demography directory for the specified
            country.
    """
    # Adjust if your repo uses a different layout.
    # Common in your project: resources/demography/<country>/
    return resources_dir


def discover_resources(resources_dir: Path) -> list[ResourceArtifact]:
    """
    Discovers resource artifacts for a specified country by searching the provided
    resources directory.

    Searches recursively for CSV files within the demography resources
    directory for the given country. Each CSV file found is converted into a
    ResourceArtifact object, where the name is the relative path without the file
    extension, ensuring uniqueness.

    Parameters:
    resources_dir (Path): The base directory containing resource files.
    country (str): The name of the country for which resources are to be discovered.

    Returns:
    list[ResourceArtifact]: A list of discovered resource artifacts.

    Raises:
    FileNotFoundError: If the expected demography resources directory is not found.
    """
    base = _guess_country_demog_dir(resources_dir)
    if not base.exists():
        raise FileNotFoundError(f"Expected demography resources directory not found: {base}")

    artifacts: list[ResourceArtifact] = []
    for p in sorted(base.rglob("*.csv")):
        # name is relative path without extension (keeps uniqueness)
        rel = p.relative_to(base).as_posix()
        name = rel.removesuffix(".csv")
        artifacts.append(ResourceArtifact(name=name, path=p))
    return artifacts


def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Reads a CSV file from the given path and returns its contents as a pandas DataFrame.

    This function provides a mechanism to safely read a CSV file and returns
    the parsed content as a pandas DataFrame for further data analysis or processing.

    Parameters:
    path : Path
        The file system path pointing to the CSV file to be read.

    Returns:
    pd.DataFrame
        A pandas DataFrame containing the data from the CSV file.

    Raises:
    FileNotFoundError
        If the given path does not exist.
    pd.errors.EmptyDataError
        If the file is empty.
    pd.errors.ParserError
        If the file cannot be parsed as CSV.
    """
    df = pd.read_csv(path)
    return df


def load_all(artifacts: list[ResourceArtifact]) -> dict[str, pd.DataFrame]:
    """
    Loads all artifacts into a dictionary mapping artifact names to Pandas DataFrame objects.

    This function takes a list of resource artifacts, reads their contents safely from
    their respective paths, and maps them to their names in the resulting dictionary.

    Parameters:
        artifacts: list[ResourceArtifact]
            A list of ResourceArtifact objects that contain the name and path of the
            resource to be loaded.

    Returns:
        dict[str, pd.DataFrame]:
            A dictionary where each key is the name of the artifact, and the corresponding
            value is a Pandas DataFrame containing the data from the artifact.

    Raises:
        Any errors raised by the `read_csv_safely` function may propagate.
    """
    out: dict[str, pd.DataFrame] = {}
    for a in artifacts:
        out[a.name] = read_csv_safely(a.path)
    return out
