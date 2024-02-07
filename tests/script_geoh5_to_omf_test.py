from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import omf
from omf.scripts import geoh5_to_omf

# pylint: disable=duplicate-code


@pytest.fixture(
    scope="module",
    name="geoh5_input_path",
    params=["test file.geoh5", "test_file.geoh5"],
)
def geoh5_input_path_fixture(request, tmp_path_factory) -> Path:
    points = omf.PointSetElement(
        name="Random Points",
        description="Just random points",
        geometry=omf.PointSetGeometry(vertices=np.random.rand(100, 3)),
        data=[
            omf.ScalarData(
                name="rand data", array=np.random.randn(100), location="vertices"
            ),
        ],
    )

    file_path = tmp_path_factory.mktemp("input") / request.param
    omf.OMFWriter(points, str(file_path))
    return file_path


def test_geoh5_to_omf_without_output_name(geoh5_input_path: Path):
    """Test the geoh5_to_omf script."""

    with patch("sys.argv", ["geoh5_to_omf", str(geoh5_input_path)]):
        geoh5_to_omf.main()

    assert (geoh5_input_path.with_suffix(".omf")).exists()


@pytest.mark.parametrize(
    "output_name", ["my_output.omf", "my output.omf", "my_output", "my output"]
)
def test_geoh5_to_omf_with_output_name(
    tmp_path, monkeypatch, geoh5_input_path: Path, output_name: str
):
    """Test the geoh5_to_omf script."""

    working_dir = tmp_path / "output"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    with patch(
        "sys.argv", ["geoh5_to_omf", str(geoh5_input_path), "-o", f"{output_name}"]
    ):
        geoh5_to_omf.main()

    expected_output = working_dir / output_name
    if not expected_output.suffix:
        expected_output = expected_output.with_suffix(".omf")
    assert expected_output.exists()


@pytest.mark.parametrize(
    "output_name", ["my_output.omf", "my output.omf", "my_output", "my output"]
)
def test_geoh5_to_omf_with_absolute_output_path(
    tmp_path, geoh5_input_path: Path, output_name: str
):
    """Test the geoh5_to_omf script."""

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with patch(
        "sys.argv",
        [
            "geoh5_to_omf",
            str(geoh5_input_path),
            "-o",
            f"{(output_dir / output_name).absolute()}",
        ],
    ):
        geoh5_to_omf.main()

    expected_output = output_dir / output_name
    if not expected_output.suffix:
        expected_output = expected_output.with_suffix(".omf")
    assert expected_output.exists()
