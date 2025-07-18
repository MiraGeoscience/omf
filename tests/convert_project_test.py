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
import os
from pathlib import Path

from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace

import omf


def test_project_to_geoh5(random_project: omf.Project, tmp_path: Path, caplog):
    """Test pointset geometry validation"""
    file = str(tmp_path / "project.geoh5")

    omf.OMFWriter(random_project, file)
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1

    with Workspace(file) as workspace:
        assert len(workspace.objects) == len(random_project.elements)

    project = omf.fileio.geoh5.GeoH5Reader(file).project

    assert len(project.elements) == len(random_project.elements)


def test_project_compression(random_project: omf.Project, tmp_path: Path):
    """Test pointset geometry validation"""
    file_low_comp = str(tmp_path / "project_low_comp.geoh5")
    file_med_comp = str(tmp_path / "project_med_comp.geoh5")
    file_high_comp = str(tmp_path / "project_high_comp.geoh5")

    omf.OMFWriter(random_project, file_low_comp, compression=1)
    omf.OMFWriter(random_project, file_med_comp, compression=5)
    omf.OMFWriter(random_project, file_high_comp, compression=9)

    size_low_comp = os.stat(file_low_comp).st_size
    size_med_comp = os.stat(file_med_comp).st_size
    size_high_comp = os.stat(file_high_comp).st_size

    assert size_low_comp > size_med_comp > size_high_comp


def test_container_group(random_project: omf.Project, tmp_path: Path):
    """Test that a container group is flatten in the omf file."""
    file = str(tmp_path / f"{__name__}.geoh5")

    omf.OMFWriter(random_project, file)
    with Workspace(tmp_path / f"{__name__}.geoh5") as ws:
        group = ContainerGroup.create(ws, name="Test Group")
        for obj in ws.objects:
            obj.parent = group

        parent = ContainerGroup.create(ws, name="Parent Group")
        group.parent = parent

    reader = omf.fileio.geoh5.GeoH5Reader(file)

    assert len(reader.project.elements) == len(random_project.elements)
