from numpy import testing


def compare_elements(elem_a, elem_b):
    """Cycle through attributes and check equal."""

    assert elem_a.name == elem_b.name

    if hasattr(elem_a, "geometry"):
        for attr in elem_a.geometry._valid_locations:
            if getattr(elem_a.geometry, attr, None) is not None:
                testing.assert_allclose(
                    getattr(elem_a.geometry, attr).array,
                    getattr(elem_b.geometry, attr).array,
                )

        if getattr(elem_a.geometry, "origin", None) is not None:
            testing.assert_allclose(elem_a.geometry.origin, elem_b.geometry.origin)

    if hasattr(elem_a, "array"):
        testing.assert_allclose(elem_a.array.array, elem_b.array.array)

    if hasattr(elem_a, "data") and elem_a.data:
        for data_a in elem_a.data:
            for data_b in elem_b.data:
                if data_b.uid == data_a.uid:
                    compare_elements(data_a, data_b)

    if hasattr(elem_a, "colormap") and elem_a.colormap:
        compare_elements(elem_a.colormap, elem_b.colormap)
