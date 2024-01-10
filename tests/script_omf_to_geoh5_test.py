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
    omf_path = tmp_path_factory.mktemp("input") / request.param
    create_omf_file(omf_path)
    return omf_path


def create_omf_file(omf_file_path: Path) -> None:
    """Create an OMF file with random data."""
    points = omf.PointSetElement(
        name="Random Points",
        description="Some random points",
        geometry=omf.PointSetGeometry(vertices=np.random.rand(100, 3)),
        data=[
            omf.ScalarData(
                name="rand data", array=np.random.randn(100), location="vertices"
            ),
        ],
    )

    omf.OMFWriter(points, str(omf_file_path))
    assert omf_file_path.exists()


def test_omf_to_geoh5_without_output_name(omf_input_path: Path):
    """Test the omf_to_geoh5 script."""

    with patch("sys.argv", ["omf_to_geoh5", str(omf_input_path)]):
        omf_to_geoh5.run()

    assert (omf_input_path.with_suffix(".geoh5")).exists()


@pytest.mark.parametrize(
    "output_name", ["my_output.geoh5", "my output.geoh5", "my_output", "my output"]
)
def test_omf_to_geoh5_with_output_name(
    tmp_path: Path, monkeypatch, omf_input_path: Path, output_name: str
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
    tmp_path: Path, omf_input_path: Path, output_name: str
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


@pytest.mark.parametrize("gzip_level", range(0, 10))
def test_omf_to_geoh5_with_gzip_level(tmp_path: Path, gzip_level: int):
    """Test the omf_to_geoh5 script."""

    omf_path = tmp_path / "test_file.omf"
    create_omf_file(omf_path)
    output_name = f"{omf_path.stem}_{gzip_level}.geoh5"
    output_dir = tmp_path / "output"
    output_path = output_dir / output_name
    output_dir.mkdir()
    with patch(
        "sys.argv",
        [
            "omf_to_geoh5",
            str(omf_path),
            "--gzip",
            f"{gzip_level}",
            "-o",
            f"{output_path.absolute()}",
        ],
    ):
        omf_to_geoh5.run()

    assert output_path.exists()


def test_omf_to_geoh5_with_gzip_level_too_high(capsys, tmp_path: Path):
    """Test the omf_to_geoh5 script."""

    omf_path = tmp_path / "test_file.omf"
    create_omf_file(omf_path)
    output_name = omf_path.with_suffix(".geoh5").name
    output_dir = tmp_path / "output"
    output_path = output_dir / output_name
    output_dir.mkdir()
    with pytest.raises(SystemExit) as captured_exception:
        with patch(
            "sys.argv",
            [
                "omf_to_geoh5",
                str(omf_path),
                "--gzip",
                "10",
                "-o",
                f"{output_path.absolute()}",
            ],
        ):
            omf_to_geoh5.run()

    assert not output_path.exists()
    assert captured_exception.value.code == 2
    captured_err = capsys.readouterr().err
    assert any(
        "error: argument --gzip: invalid choice: 10" in line
        for line in captured_err.splitlines()
    )
