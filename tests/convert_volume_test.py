"""Tests for PointSet validation"""

import numpy as np
from geoh5py.objects import BlockModel
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
            omf.MappedData(
                name="Reference Data 2",
                location="cells",
                array=np.random.randint(-1, 3, np.prod(dims))
                .flatten()
                .astype(np.int32),
                legends=[
                    omf.Legend(
                        values=omf.ColorArray(
                            array=[
                                [255, 0, 255],
                                [255, 255, 0],
                                [255, 0, 0],
                            ]
                        )
                    ),
                    omf.Legend(values=omf.StringArray(array=["abc", "123", "@#$%"])),
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

        # Compare reference data created two ways
        ref_a = block_model.get_entity("Reference Data")[0]
        ref_b = block_model.get_entity("Reference Data 2")[0]

        assert all(
            [
                key in ref_a.value_map.map and value == ref_a.value_map.map[key]
                for key, value in ref_b.value_map.map.items()
            ]
        )

    omf.fileio.utils.compare_elements(converted_omf, vol)

    project = omf.fileio.geoh5.GeoH5Reader(file)()
    omf_vol = project.elements[0]

    omf.fileio.utils.compare_elements(omf_vol, vol)


def test_volume_flip_origin_z(tmp_path):
    dims = [10, 15, 20]
    vol = omf.VolumeElement(
        name="vol",
        geometry=omf.VolumeGridGeometry(
            tensor_u=np.ones(dims[0]).astype(float),
            tensor_v=np.ones(dims[1]).astype(float),
            tensor_w=np.ones(dims[2]).astype(float),
            axis_w=np.r_[0, 0, -1],
            origin=[10.0, 10.0, -10],
        ),
    )

    file = str(tmp_path / "block_model.geoh5")
    omf.OMFWriter(vol, file)

    with Workspace(file) as workspace:
        block_model = workspace.get_entity("vol")[0]

        assert block_model.z_cell_delimiters[-1] < 0
        assert (
            block_model.origin["z"]
            == vol.geometry.origin[2] + vol.geometry.tensor_w[0] / 2
        )

    with Workspace(file) as workspace:
        rotation = np.random.normal(-180, 180, 1)
        block = BlockModel.create(
            workspace,
            origin=[0, 0, 0],
            u_cell_delimiters=np.arange(0, 11) * 2.5,
            v_cell_delimiters=np.arange(0, 22) * 3.6,
            z_cell_delimiters=np.arange(0, 33) * 4.7,
            rotation=rotation,
        )

        converter = omf.fileio.geoh5.get_conversion_map(block, workspace)
        converted_omf = converter.from_geoh5(block)

        with Workspace(str(tmp_path / "block_model_converted.geoh5")) as out_ws:
            converter = omf.fileio.geoh5.get_conversion_map(converted_omf, out_ws)
            converted_geoh5 = converter.from_omf(converted_omf)

            np.testing.assert_array_almost_equal(
                block.centroids, converted_geoh5.centroids
            )
