"""Test the version follows SemVer and is consistent across files."""

from __future__ import annotations

import re
from pathlib import Path

import tomli as toml

import omf


def get_version():
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    with open(str(path), encoding="utf-8") as file:
        pyproject = toml.loads(file.read())

    return pyproject["tool"]["poetry"]["version"]


def get_version_in_readme() -> str | None:
    path = Path(__file__).resolve().parents[1] / "README.rst"

    version_re = r"^\s*Version:\s*(\S.*)\s*"
    with open(str(path), encoding="utf-8") as file:
        for line in file:
            match = re.match(version_re, line)
            if match:
                return match[1]
    return None


def test_version_is_consistent():
    assert omf.__version__ == get_version()


def test_version_in_readme():
    assert omf.__version__ == get_version_in_readme()


def test_version_is_semver():
    semver_re = (
        r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
        r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    assert re.search(semver_re, omf.__version__) is not None
