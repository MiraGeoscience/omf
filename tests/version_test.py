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

    version_re = r"^\s*Version:\s*(\S*)\s*$"
    with open(str(path), encoding="utf-8") as file:
        for line in file:
            match = re.match(version_re, line)
            if match:
                return match[1]
    return None


def test_version_is_consistent():
    assert omf.__version__ == get_version()


def version_base_and_pre() -> tuple[str, str]:
    """
    Return a tuple ith the version base and its prerelease segment
    (if present the build segment (+) is also in the 2nd tuple element).
    """
    version_re = r"^([^-+\s]*)(-\S*)?\s*$"
    match = re.match(version_re, omf.__version__)
    assert match is not None
    return match[1], match[2]


def test_version_in_readme():
    version_base, prerelease = version_base_and_pre()
    version_readme = get_version_in_readme()
    assert version_readme is not None
    if prerelease is not None and prerelease.startswith("-rc."):
        assert version_readme in [omf.__version__, version_base]
    else:
        assert version_readme == omf.__version__


def test_version_is_semver():
    semver_re = (
        r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
        r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    assert re.search(semver_re, omf.__version__) is not None
