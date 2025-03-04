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


def test_surface_to_geoh5(tmp_path: Path):
    """Test pointset geometry validation"""
    surf = omf.SurfaceElement(
        name="trisurf",
        geometry=omf.SurfaceGeometry(
            vertices=np.random.rand(100, 3),
            triangles=np.floor(np.random.rand(50, 3) * 100).astype(int),
        ),
        data=[
            omf.ScalarData(
                name="rand vert data", array=np.random.rand(100), location="vertices"
            ),
            omf.ScalarData(
                name="rand face data", array=np.random.rand(50), location="faces"
            ),
        ],
        color=[100, 200, 200],
    )
    file = str(tmp_path / "surface.geoh5")
    omf.OMFWriter(surf, file)

    with Workspace(file) as workspace:
        geoh5_surf = workspace.get_entity("trisurf")[0]
        np.testing.assert_array_almost_equal(
            np.r_[surf.geometry.vertices.array], geoh5_surf.vertices
        )

        data = geoh5_surf.get_entity("rand vert data")[0]
        np.testing.assert_array_almost_equal(np.r_[surf.data[0].array], data.values)

        data = geoh5_surf.get_entity("rand face data")[0]
        np.testing.assert_array_almost_equal(np.r_[surf.data[1].array], data.values)

        converter = omf.fileio.geoh5.get_conversion_map(geoh5_surf, workspace)
        converted_omf = converter.from_geoh5(geoh5_surf)

    omf.fileio.utils.compare_elements(converted_omf, surf)

    project = omf.fileio.geoh5.GeoH5Reader(file).project
    omf_surf = project.elements[0]

    omf.fileio.utils.compare_elements(omf_surf, surf)
