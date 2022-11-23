"""Tests for PointSet validation"""

import numpy as np
from geoh5py.workspace import Workspace

import omf
from omf.fileio.geoh5 import block_model_reordering


def test_volume_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
    dims = [10, 15, 20]
    vol = omf.VolumeElement(
        name="vol",
        geometry=omf.VolumeGridGeometry(
            tensor_u=np.ones(dims[0]).astype(float),
            tensor_v=np.ones(dims[1]).astype(float),
            tensor_w=np.ones(dims[2]).astype(float),
            origin=[10.0, 10.0, -10],
        ),
        data=[
            omf.ScalarData(
                name="Random Int Data",
                location="cells",
                array=np.arange(np.prod(dims)).flatten().astype(np.int32),
            ),
            omf.ScalarData(
                name="Random Float Data",
                location="cells",
                array=np.random.randn(np.prod(dims)),
            ),
            omf.MappedData(
                name="Reference Data",
                location="cells",
                array=np.random.randint(-1, 3, np.prod(dims))
                .flatten()
                .astype(np.int32),
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
        data = block_model.get_entity("Random Int Data")[0]
        np.testing.assert_array_almost_equal(
            np.r_[vol.data[0].array], block_model_reordering(block_model, data.values)
        )

        converter = omf.fileio.geoh5.get_conversion_map(block_model, workspace)
        converted_omf = converter.from_geoh5(block_model)

    omf.fileio.utils.compare_elements(converted_omf, vol)

    project = omf.fileio.geoh5.GeoH5Reader(file)()
    omf_vol = project.elements[0]

    omf.fileio.utils.compare_elements(omf_vol, vol)
