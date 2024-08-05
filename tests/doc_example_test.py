from pathlib import Path

import omf


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
