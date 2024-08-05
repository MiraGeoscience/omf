# pylint: disable=duplicate-code

from pathlib import Path

import numpy as np
import pytest

import omf


@pytest.fixture
def random_project() -> omf.Project:
    tests_dir = Path(__file__).resolve().parent
    png_file_path = tests_dir.parent / "docs" / "images" / "PointSetGeometry.png"
    proj = omf.Project(name="Test project", description="Just some assorted elements")

    pts = omf.PointSetElement(
        name="Random Points",
        description="Just random points",
        geometry=omf.PointSetGeometry(vertices=np.random.rand(100, 3)),
        data=[
            omf.ScalarData(
                name="rand data", array=np.random.rand(100), location="vertices"
            ),
            omf.ScalarData(
                name="More rand data",
                array=np.random.rand(100),
                location="vertices",
            ),
        ],
        textures=[
            omf.ImageTexture(
                name="test image",
                image=str(png_file_path),
                origin=[0, 0, 0],
                axis_u=[1, 0, 0],
                axis_v=[0, 1, 0],
            ),
            omf.ImageTexture(
                name="test image",
                image=str(png_file_path),
                origin=[0, 0, 0],
                axis_u=[1, 0, 0],
                axis_v=[0, 0, 1],
            ),
        ],
        color="green",
    )

    lin = omf.LineSetElement(
        name="Random Line",
        geometry=omf.LineSetGeometry(
            vertices=np.random.rand(100, 3),
            segments=np.floor(np.random.rand(50, 2) * 100).astype(int),
        ),
        data=[
            omf.ScalarData(
                name="rand vert data",
                array=np.random.rand(100),
                location="vertices",
            ),
            omf.ScalarData(
                name="rand segment data",
                array=np.random.rand(50),
                location="segments",
            ),
        ],
        color="#0000FF",
    )

    surf = omf.SurfaceElement(
        name="trisurf",
        geometry=omf.SurfaceGeometry(
            vertices=np.random.rand(100, 3),
            triangles=np.floor(np.random.rand(50, 3) * 100).astype(int),
        ),
        data=[
            omf.ScalarData(
                name="rand vert data",
                array=np.random.rand(100),
                location="vertices",
            ),
            omf.ScalarData(
                name="rand face data", array=np.random.rand(50), location="faces"
            ),
        ],
        color=[100, 200, 200],
    )

    grid = omf.SurfaceElement(
        name="gridsurf",
        geometry=omf.SurfaceGridGeometry(
            tensor_u=np.ones(10).astype(float),
            tensor_v=np.ones(15).astype(float),
            origin=[50.0, 50.0, 50.0],
            axis_u=[1.0, 0, 0],
            axis_v=[0, 0, 1.0],
            offset_w=np.random.rand(11, 16).flatten(),
        ),
        data=[
            omf.ScalarData(
                name="rand vert data",
                array=np.random.rand(11, 16).flatten(),
                location="vertices",
            ),
            omf.ScalarData(
                name="rand face data",
                array=np.random.rand(10, 15).flatten(order="f"),
                location="faces",
            ),
        ],
        textures=[
            omf.ImageTexture(
                name="test image",
                image=str(png_file_path),
                origin=[2.0, 2.0, 2.0],
                axis_u=[5.0, 0, 0],
                axis_v=[0, 2.0, 5.0],
            )
        ],
    )

    vol = omf.VolumeElement(
        name="vol",
        geometry=omf.VolumeGridGeometry(
            tensor_u=np.ones(10).astype(float),
            tensor_v=np.ones(15).astype(float),
            tensor_w=np.ones(20).astype(float),
            origin=[10.0, 10.0, -10],
        ),
        data=[
            omf.ScalarData(
                name="Random Data",
                location="cells",
                array=np.random.rand(10, 15, 20).flatten(),
            )
        ],
    )

    proj.elements = [pts, lin, surf, grid, vol]

    return proj


def test_doc_ex(tmp_path: Path, random_project):
    assert random_project.validate()

    serial_file = str(tmp_path / "out.omf")
    omf.OMFWriter(random_project, serial_file)
    reader = omf.OMFReader(serial_file)
    new_proj = reader.get_project()

    assert new_proj.validate()
    assert str(new_proj.elements[3].textures[0].uid) == str(
        random_project.elements[3].textures[0].uid
    )

    proj_overview = reader.get_project_overview()
    for elem in proj_overview.elements:
        assert len(elem.data) == 0

    del reader
