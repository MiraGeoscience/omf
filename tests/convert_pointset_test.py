"""Tests for PointSet validation"""

import numpy as np
import numpy.testing
from geoh5py.workspace import Workspace

import omf


def test_pointset_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
    orig_pts = omf.PointSetElement(
        name="Random Points",
        description="Just random points",
        geometry=omf.PointSetGeometry(vertices=np.random.rand(100, 3)),
        data=[
            omf.ScalarData(
                name="rand data", array=np.random.rand(100), location="vertices"
            ),
        ],
    )

    file = str(tmp_path / "pointset.geoh5")
    omf.OMFWriter(orig_pts, file)

    with Workspace(file) as workspace:
        points = workspace.get_entity("Random Points")[0]
        np.testing.assert_array_almost_equal(
            np.r_[orig_pts.geometry.vertices.array], points.vertices
        )

        data = points.get_entity("rand data")[0]
        np.testing.assert_array_almost_equal(np.r_[orig_pts.data[0].array], data.values)

    omf_pts = omf.fileio.geoh5.GeoH5Reader(points).element

    compare_elements(omf_pts, orig_pts)


def compare_elements(elem_a, elem_b):
    """Cycle through attributes and check equal."""

    assert elem_a.name == elem_b.name

    if hasattr(elem_a, "geometry"):
        for attr in elem_a.geometry._valid_locations:
            numpy.testing.assert_allclose(
                getattr(elem_a.geometry, attr).array,
                getattr(elem_b.geometry, attr).array,
            )

    if hasattr(elem_a, "array"):
        numpy.testing.assert_allclose(elem_a.array.array, elem_b.array.array)

    if hasattr(elem_a, "data") and elem_a.data:
        for data_a, data_b in zip(elem_a.data, elem_b.data):
            compare_elements(data_a, data_b)
