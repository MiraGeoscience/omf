"""Tests for PointSet validation"""

import logging
import os

from geoh5py.workspace import Workspace

import omf

from .doc_example_test import TestDocEx


def test_project_to_geoh5(tmp_path, caplog):
    """Test pointset geometry validation"""
    proj = TestDocEx.make_random_project()

    file = str(tmp_path / "project.geoh5")

    omf.OMFWriter(proj, file)
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1

    with Workspace(file) as workspace:
        assert len(workspace.objects) == len(proj.elements)

    project = omf.fileio.geoh5.GeoH5Reader(file).project

    assert len(project.elements) == len(proj.elements)


def test_project_compression(tmp_path):
    """Test pointset geometry validation"""
    proj = TestDocEx.make_random_project()

    file_low_comp = str(tmp_path / "project_low_comp.geoh5")
    file_med_comp = str(tmp_path / "project_med_comp.geoh5")
    file_high_comp = str(tmp_path / "project_high_comp.geoh5")

    omf.OMFWriter(proj, file_low_comp, compression=1)
    omf.OMFWriter(proj, file_med_comp, compression=5)
    omf.OMFWriter(proj, file_high_comp, compression=9)

    size_low_comp = os.stat(file_low_comp).st_size
    size_med_comp = os.stat(file_med_comp).st_size
    size_high_comp = os.stat(file_high_comp).st_size

    assert size_low_comp > size_med_comp > size_high_comp
