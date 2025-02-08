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

import logging
from pathlib import Path

import numpy as np
from geoh5py.workspace import Workspace

import omf


def test_grid2d_to_geoh5(tmp_path: Path, caplog):
    """Test pointset geometry validation"""

    dip = np.random.uniform(low=0.0, high=90, size=1)
    rotation = np.random.uniform(low=-180, high=180, size=1)
    rot_op = omf.fileio.geoh5.rotation_opt(np.deg2rad(rotation), np.deg2rad(dip))
    grid = omf.SurfaceElement(
        name="gridsurf",
        geometry=omf.SurfaceGridGeometry(
            tensor_u=np.ones(10).astype(float),
            tensor_v=np.ones(15).astype(float),
            origin=[50.0, 50.0, 50.0],
            axis_u=rot_op @ np.c_[1, 0, 0].T.flatten(),
            axis_v=rot_op @ np.c_[0, 1, 0].T.flatten(),
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
    )
    file = str(tmp_path / "grid2d.geoh5")

    omf.OMFWriter(grid, file)
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1

    with Workspace(file) as workspace:
        grid2d = workspace.get_entity("gridsurf")[0]

        np.testing.assert_array_almost_equal(grid2d.dip, dip)
        np.testing.assert_array_almost_equal(grid2d.rotation, rotation)

        data = grid2d.get_entity("rand vert data")[0]
        np.testing.assert_array_almost_equal(np.r_[grid.data[0].array], data.values)

        data = grid2d.get_entity("rand face data")[0]
        np.testing.assert_array_almost_equal(np.r_[grid.data[1].array], data.values)

        converter = omf.fileio.geoh5.get_conversion_map(grid2d, workspace)
        converted_omf = converter.from_geoh5(grid2d)

    omf.fileio.utils.compare_elements(converted_omf, grid)

    project = omf.fileio.geoh5.GeoH5Reader(file).project
    omf_grid = project.elements[0]

    omf.fileio.utils.compare_elements(omf_grid, grid)
