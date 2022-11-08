"""Tests for PointSet validation"""

import numpy as np
from geoh5py.workspace import Workspace

import omf


def test_volume_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
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
                array=np.arange(10 * 15 * 20).flatten().astype(np.int32),
            ),
            omf.MappedData(
                name="Reference Data",
                location="cells",
                array=np.random.randint(-1, 3, 10 * 15 * 20).flatten().astype(np.int32),
                legends=[
                    omf.Legend(values=omf.StringArray(array=["abc", "123", "@#$%"])),
                    omf.Legend(
                        values=omf.ColorArray(
                            array=[
                                [255, 0, 255],
                                [255, 255, 0],
                                [255, 0, 0],
                            ]
                        )
                    ),
                ],
            ),
        ],
    )

    file = str(tmp_path / "block_model.geoh5")
    omf.OMFWriter(vol, file)

    with Workspace(file) as workspace:
        block_model = workspace.get_entity("vol")[0]
        data = block_model.get_entity("Random Data")[0]
        np.testing.assert_array_almost_equal(np.r_[vol.data[0].array], data.values)

    project = omf.fileio.geoh5.GeoH5Reader(file).project
    omf_vol = project.elements[0]

    omf.fileio.utils.compare_elements(omf_vol, vol)
