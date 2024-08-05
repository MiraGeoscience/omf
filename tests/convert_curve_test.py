"""Tests for PointSet validation"""

from pathlib import Path

import numpy as np
from geoh5py.workspace import Workspace

import omf


def test_lineset_to_geoh5(tmp_path: Path):
    """Test pointset geometry validation"""
    line = omf.LineSetElement(
        name="Random Line",
        geometry=omf.LineSetGeometry(
            vertices=np.random.rand(100, 3),
            segments=np.floor(np.random.rand(50, 2) * 100).astype(int),
        ),
        data=[
            omf.ScalarData(
                name="rand vert data", array=np.random.rand(100), location="vertices"
            ),
            omf.ScalarData(
                name="rand segment data", array=np.random.rand(50), location="segments"
            ),
        ],
        color="#0000FF",
    )
    file = str(tmp_path / "lineset.geoh5")
    omf.OMFWriter(line, file)

    with Workspace(file) as workspace:
        curve = workspace.get_entity("Random Line")[0]
        np.testing.assert_array_almost_equal(
            np.r_[line.geometry.vertices.array], curve.vertices
        )

        data = curve.get_entity("rand vert data")[0]
        np.testing.assert_array_almost_equal(np.r_[line.data[0].array], data.values)

        data = curve.get_entity("rand segment data")[0]
        np.testing.assert_array_almost_equal(np.r_[line.data[1].array], data.values)

        converter = omf.fileio.geoh5.get_conversion_map(curve, workspace)
        converted_omf = converter.from_geoh5(curve)

    omf.fileio.utils.compare_elements(converted_omf, line)

    project = omf.fileio.geoh5.GeoH5Reader(file).project
    omf_line = project.elements[0]

    omf.fileio.utils.compare_elements(omf_line, line)
