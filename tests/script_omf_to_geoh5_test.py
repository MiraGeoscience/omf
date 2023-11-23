from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import omf
from omf.scripts import omf_to_geoh5


@pytest.fixture(
    scope="module", name="omf_input_path", params=["test file.omf", "test_file.omf"]
)
def omf_input_path_fixture(request, tmp_path_factory) -> Path:
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


def test_omf_to_geoh5_without_output_name(omf_input_path: Path):
    """Test the omf_to_geoh5 script."""

    with patch("sys.argv", ["omf_to_geoh5", str(omf_input_path)]):
        omf_to_geoh5.run()

    assert (omf_input_path.with_suffix(".geoh5")).exists()


@pytest.mark.parametrize(
    "output_name", ["my_output.geoh5", "my output.geoh5", "my_output", "my output"]
)
def test_omf_to_geoh5_with_output_name(
    tmp_path, monkeypatch, omf_input_path: Path, output_name: str
):
    """Test the omf_to_geoh5 script."""

    working_dir = tmp_path / "output"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    with patch(
        "sys.argv", ["omf_to_geoh5", str(omf_input_path), "-o", f"{output_name}"]
    ):
        omf_to_geoh5.run()

    expected_output = working_dir / output_name
    if not expected_output.suffix:
        expected_output = expected_output.with_suffix(".geoh5")
    assert expected_output.exists()


@pytest.mark.parametrize(
    "output_name", ["my_output.geoh5", "my output.geoh5", "my_output", "my output"]
)
def test_omf_to_geoh5_with_absolute_output_path(
    tmp_path, omf_input_path: Path, output_name: str
):
    """Test the omf_to_geoh5 script."""

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with patch(
        "sys.argv",
        [
            "omf_to_geoh5",
            str(omf_input_path),
            "-o",
            f"{(output_dir / output_name).absolute()}",
        ],
    ):
        omf_to_geoh5.run()

    expected_output = output_dir / output_name
    if not expected_output.suffix:
        expected_output = expected_output.with_suffix(".geoh5")
    assert expected_output.exists()
