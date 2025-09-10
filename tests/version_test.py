"""Test the version follows SemVer and is consistent across files."""

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of mira-omf package.                                      '
#                                                                              '
#  mira-omf is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                 '
#                                                                              '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import yaml
from jinja2 import Template
from packaging.version import InvalidVersion, Version

import omf


def get_conda_recipe_version() -> str | None:
    recipe_path = Path(__file__).resolve().parents[1] / "recipe.yaml"

    with recipe_path.open(encoding="utf-8") as file:
        recipe = yaml.safe_load(file)

    return recipe["context"]["version"]


def test_version_is_consistent():
    project_version = Version(omf.__version__)
    conda_version = Version(get_conda_recipe_version())
    assert conda_version.base_version == project_version.base_version


def _version_module_exists():
    try:
        importlib.import_module("omf._version")
        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(
    _version_module_exists(),
    reason="omf._version can be found: package is built",
)
def test_fallback_version_is_zero():
    project_version = Version(omf.__version__)
    fallback_version = Version("0.0.0.dev0")
    assert project_version.base_version == fallback_version.base_version
    assert project_version.pre is None
    assert project_version.post is None
    assert project_version.dev == fallback_version.dev


@pytest.mark.skipif(
    not _version_module_exists(),
    reason="omf._version cannot be found: uses a fallback version",
)
def test_conda_version_is_consistent():
    project_version = Version(omf.__version__)
    conda_version = Version(get_conda_recipe_version())

    assert conda_version.is_devrelease == project_version.is_devrelease
    assert conda_version.is_prerelease == project_version.is_prerelease
    assert conda_version.is_postrelease == project_version.is_postrelease
    assert conda_version == project_version


def version_base_and_pre() -> tuple[str, str]:
    """
    Return a tuple ith the version base and its prerelease segment
    (if present the build segment (+) is also in the 2nd tuple element).
    """
    version_re = r"^([^-+\s]*)(-\S*)?\s*$"
    match = re.match(version_re, omf.__version__)
    assert match is not None
    return match[1], match[2]


def validate_version(version_str):
    try:
        version = Version(version_str)
        return (version.major, version.minor, version.micro, version.pre, version.post)
    except InvalidVersion:
        return None


def test_version_is_valid():
    assert validate_version(omf.__version__) is not None
