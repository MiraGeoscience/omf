"""Tests for PointSet validation"""
import logging

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
        assert len(workspace.objects) == len(proj.elements) - 1

    project = omf.fileio.geoh5.GeoH5Reader(file).project

    assert len(project.elements) == len(proj.elements) - 1
