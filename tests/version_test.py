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

from pathlib import Path

import tomli as toml
import yaml
from jinja2 import Template
from packaging.version import InvalidVersion, Version

import omf


def get_pyproject_version():
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    with open(str(path), encoding="utf-8") as file:
        pyproject = toml.loads(file.read())

    return pyproject["project"]["version"]


def get_conda_recipe_version():
    path = Path(__file__).resolve().parents[1] / "recipe.yaml"

    with open(str(path), encoding="utf-8") as file:
        content = file.read()

    template = Template(content)
    rendered_yaml = template.render()

    recipe = yaml.safe_load(rendered_yaml)

    return recipe["context"]["version"]


def test_version_is_consistent():
    assert omf.__version__ == get_pyproject_version()
    normalized_conda_version = Version(get_conda_recipe_version())
    normalized_version = Version(omf.__version__)
    assert normalized_conda_version == normalized_version


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