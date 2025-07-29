"""Tests for PointSet validation"""

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of mira-omf package.                                      '
#                                                                              '
#  mira-omf is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                 '
#                                                                              '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from pathlib import Path

import numpy as np
from geoh5py.workspace import Workspace

import omf


def test_pointset_to_geoh5(tmp_path: Path, caplog):
    """Test pointset geometry validation"""
    colormap = omf.ColorArray(
        array=[
            tuple(row)
            for row in np.c_[
                np.linspace(0, 255, 128),
                np.linspace(0, 2, 128),
                np.linspace(255, 0, 128),
            ]
            .astype(int)
            .tolist()
        ]
    )
    orig_pts = omf.PointSetElement(
        name="Random Points",
        description="Just random points",
        geometry=omf.PointSetGeometry(vertices=np.random.rand(100, 3)),
        data=[
            omf.ScalarData(
                name="rand data",
                array=np.random.randn(100),
                location="vertices",
                colormap=omf.ScalarColormap(limits=[-1, 1], gradient=colormap),
            ),
        ],
    )

    file = str(tmp_path / "pointset.geoh5")
    omf.OMFWriter(orig_pts, file)

    # Check that the file was created
    with Workspace(file) as workspace:
        geoh5_points = workspace.get_entity("Random Points")[0]
        np.testing.assert_array_almost_equal(
            np.r_[orig_pts.geometry.vertices.array], geoh5_points.vertices
        )

        geoh5_points.add_default_visual_parameters()
        data = geoh5_points.get_entity("rand data")[0]
        np.testing.assert_array_almost_equal(np.r_[orig_pts.data[0].array], data.values)

        converter = omf.fileio.geoh5.get_conversion_map(geoh5_points, workspace)

        with caplog.at_level("WARNING"):
            converted_omf = converter.from_geoh5(geoh5_points)

        assert len(caplog.text) == 0, "No warnings should be raised during conversion"
        assert len(converted_omf.data) == 1  # Skip the visual parameters
    omf.fileio.utils.compare_elements(converted_omf, orig_pts)

    project = omf.fileio.geoh5.GeoH5Reader(file).project
    omf_pts = project.elements[0]

    omf.fileio.utils.compare_elements(omf_pts, orig_pts)
