from __future__ import annotations

import warnings
from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from geoh5py.data import Data, FloatData, ReferencedData
from geoh5py.groups import RootGroup
from geoh5py.objects import BlockModel, Curve, Grid2D, ObjectBase, Points, Surface
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace

from omf.base import Project, UidModel
from omf.data import (
    ColorArray,
    Int2Array,
    MappedData,
    ScalarArray,
    ScalarColormap,
    ScalarData,
    StringArray,
    Vector3Array,
)
from omf.lineset import LineSetElement, LineSetGeometry
from omf.pointset import PointSetElement, PointSetGeometry
from omf.surface import SurfaceElement, SurfaceGeometry, SurfaceGridGeometry
from omf.volume import VolumeElement, VolumeGridGeometry


class OMFtoGeoh5NotImplemented(NotImplementedError):
    """Custom error message for attributes not implemented by geoh5."""

    def __init__(
        self,
        name: str,
    ):
        super().__init__(OMFtoGeoh5NotImplemented.message(name))

    @staticmethod
    def message(info):
        """Custom error message."""
        return f"Cannot perform the conversion from OMF to geoh5. {info}"


class GeoH5Writer:  # pylint: disable=too-few-public-methods
    """
    OMF to geoh5 class converter
    """

    def __init__(self, element: UidModel, file_name: str | Path):

        if not isinstance(file_name, (str, Path)):
            raise TypeError("Input 'file' must be of str or Path.")

        self.file = file_name
        self.entity = element

    @property
    def entity(self):
        return self._entity

    @entity.setter
    def entity(self, element):
        converter = get_conversion_map(element, self.file)
        self._entity = converter.from_omf(element)


def get_conversion_map(element: UidModel | Entity, workspace: str | Path | Workspace):
    """
    Utility method to get the appropriate conversion class is it exists.

    :param element: Either an omf or geoh5 class.
    :param workspace: Path to a geoh5 or active :obj:`geoh5py.workspace.Workspace`.

    :returns: A sub-class of BaseConversion for the given element.
    """
    if type(element) not in _CONVERSION_MAP:
        raise OMFtoGeoh5NotImplemented(
            f"Element of type {type(element)} currently not implemented."
        )

    # Special case for SurfaceElement describing Grid2D
    if isinstance(element, SurfaceElement) and isinstance(
        element.geometry, SurfaceGridGeometry
    ):
        return SurfaceGridConversion(element, workspace)

    return _CONVERSION_MAP[type(element)](element, workspace)


class GeoH5Reader:  # pylint: disable=too-few-public-methods
    """
    Geoh5 to omf class converter
    """

    def __init__(self, file_name: str | Path):

        with Workspace(file_name, mode="r") as workspace:
            self.file = workspace
            converter = ProjectConversion(workspace.root, self.file)
            self.project = converter.from_geoh5(workspace.root)


class BaseConversion(ABC):
    """
    Base conversion between OMF and geoh5 format.
    """

    geoh5: str | Path | Workspace = None
    geoh5_type: type[Entity]
    omf_type: type[UidModel]
    _attribute_map: dict = {
        "uid": "uid",
        "name": "name",
    }
    _element = None
    _entity = None

    def __init__(self, element, geoh5: str | Path | Workspace):
        if element is None:
            raise ValueError("Input 'element' cannot be None.")

        self.geoh5 = geoh5

    def collect_omf_attributes(self, element, **kwargs):
        with fetch_h5_handle(self.geoh5) as workspace:
            for key, alias in self._attribute_map.items():
                prop = getattr(element, key, None)

                if prop is None:
                    continue

                if isinstance(alias, type(BaseConversion)):
                    conversion = alias(  # pylint: disable=not-callable
                        element, workspace
                    )
                    kwargs = conversion.collect_omf_attributes(prop, **kwargs)
                else:
                    kwargs[alias] = prop

        return kwargs

    def collect_h5_attributes(self, entity, **kwargs):
        with fetch_h5_handle(self.geoh5) as workspace:
            for key, alias in self._attribute_map.items():

                if isinstance(alias, type(BaseConversion)):
                    conversion = alias(  # pylint: disable=not-callable
                        entity, workspace
                    )
                    prop = conversion.from_geoh5(entity)
                else:
                    prop = getattr(entity, alias, None)

                if prop is not None:
                    kwargs[key] = prop

        return kwargs

    def from_omf(self, element, **kwargs) -> Entity | None:
        """Convert omf element to geoh5 entity."""

    def from_geoh5(self, entity, **kwargs) -> UidModel | None:
        """Generate an omf element from geoh5 attributes."""
        with fetch_h5_handle(self.geoh5):
            kwargs = self.collect_h5_attributes(entity, **kwargs)

        if kwargs:
            return self.omf_type(**kwargs)

        return None

    def process_dependents(self, element, parent: UidModel | Entity, workspace) -> list:
        children = []
        children_list = []
        if isinstance(element, UidModel):
            method = "from_omf"
            if getattr(element, "data", None):
                children_list = element.data
            elif getattr(element, "elements", None):
                children_list = element.elements
            kwargs = {"parent": parent}

        else:
            method = "from_geoh5"
            children_list = getattr(element, "children", [])
            kwargs = {}

        for child in children_list:
            try:
                converter = get_conversion_map(child, workspace)
                children += [getattr(converter, method)(child, **kwargs)]
            except OMFtoGeoh5NotImplemented as error:
                warnings.warn(error.args[0])
                continue

        return children


class DataConversion(BaseConversion):
    """
    Conversion between :obj:`omf.data.Data` and
    :obj:`geoh5py.data.Data`
    """

    _attribute_map: dict[str, Any] = {
        "uid": "uid",
        "name": "name",
    }

    def from_omf(self, element, **kwargs) -> Data | dict:
        with fetch_h5_handle(self.geoh5):

            parent = kwargs.get("parent", None)

            if not isinstance(parent, ObjectBase):
                raise UserWarning(
                    "Input argument for DataConversion.from_omf requires a "
                    "'parent' of type ObjectBase."
                )

            kwargs = self.collect_omf_attributes(element, **kwargs)

            if element.location in ["faces", "cells", "segments"]:
                kwargs["association"] = "CELL"
            else:
                kwargs["association"] = "VERTEX"

            colormap = None
            if "color_map" in kwargs:
                colormap = kwargs.pop("color_map")

            if "referenced" in kwargs and "values" in kwargs:
                kwargs["values"] += 1

            entity = parent.add_data({element.name: kwargs})

            if colormap is not None:
                entity.entity_type.color_map = colormap

        return entity

    def from_geoh5(self, entity, **kwargs) -> UidModel:
        """Convert a geoh5 data to omf element."""
        with fetch_h5_handle(self.geoh5):
            kwargs = self.collect_h5_attributes(entity, **kwargs)
            uid = kwargs.pop("uid")

            if entity.association.name == "VERTEX":
                kwargs["location"] = "vertices"
            else:
                kwargs["location"] = _ASSOCIATION_MAP[type(entity.parent)]

            element = self.omf_type(**kwargs)
            if hasattr(element, "_backend"):
                element._backend.update({"uid": uid})  # pylint: disable=W0212

        return element


class ElementConversion(BaseConversion):
    """
    Conversion between :obj:`omf.pointset.PointSetElement` and
    :obj:`geoh5py.objects.Points`
    """

    _attribute_map: dict[str, Any] = {
        "name": "name",
        "uid": "uid",
    }

    def __init__(self, obj: UidModel | Entity, geoh5: str | Path | Workspace):
        super().__init__(obj, geoh5)

        if isinstance(obj, UidModel) and hasattr(obj, "geometry"):
            self.geoh5_type = _CLASS_MAP[type(obj.geometry)]

    def from_omf(self, element, **kwargs) -> Entity | None:
        """Convert omf element to geoh5 entity."""
        with fetch_h5_handle(self.geoh5) as workspace:
            try:
                kwargs = self.collect_omf_attributes(element, **kwargs)
            except OMFtoGeoh5NotImplemented as error:
                warnings.warn(error.args[0])
                return None

            entity = workspace.create_entity(self.geoh5_type, **{"entity": kwargs})
            self.process_dependents(element, entity, workspace)

        return entity

    def from_geoh5(self, entity, **kwargs) -> UidModel:
        """Convert a geoh5 entity to omf element."""
        with fetch_h5_handle(self.geoh5) as workspace:
            kwargs = self.collect_h5_attributes(entity, **kwargs)
            uid = kwargs.pop("uid")
            element = self.omf_type(**kwargs)

            if hasattr(element, "_backend"):
                element._backend.update({"uid": uid})  # pylint: disable=W0212

            element.data = self.process_dependents(entity, element, workspace)

        return element


class ProjectConversion(BaseConversion):
    """
    Conversion between a :obj:`omf.base.Project` and :obj:`geoh5py.groups.RootGroup`
    """

    omf_type = Project
    geoh5_type = RootGroup
    _attribute_map: dict = {
        "uid": "uid",
        "name": "name",
        "units": "distance_unit",
        "revision": "version",
    }

    def from_omf(self, element, **kwargs) -> RootGroup:
        """Convert omf project to geoh5 root."""
        with fetch_h5_handle(self.geoh5) as workspace:
            kwargs = self.collect_omf_attributes(element, **kwargs)
            root: RootGroup = workspace.root

            for key, value in kwargs.items():
                setattr(root, key, value)

            self.process_dependents(element, root, workspace)

        return root

    def from_geoh5(self, entity, **kwargs) -> Project:
        """Convert RootGroup to omf Project."""
        with fetch_h5_handle(self.geoh5) as workspace:
            kwargs = self.collect_h5_attributes(entity, **kwargs)
            uid = kwargs.pop("uid")
            project = self.omf_type(**kwargs)
            project._backend.update({"uid": uid})  # pylint: disable=W0212
            project.elements = self.process_dependents(entity, project, workspace)

        return project


# class ArrayConversion(BaseConversion):
#     """
#     Conversion from :obj:`omf.data.Int2Array` or `Vector3Array` to :obj:`numpy.ndarray`
#     """
#
#     omf_type = ScalarArray
#     geoh5_type = np.ndarray
#     _attribute_map: dict = {}
#
#     def from_omf(self, **kwargs) -> dict:
#
#         if getattr(self.element, "array", None) is not None:
#             kwargs["values"] = np.c_[self.element.array]
#         return kwargs


class ValuesConversion(DataConversion):
    """
    Conversion between :obj:`omf.data.ScalarArray` and
    :obj:`geoh5py.data.Data.values`
    """

    omf_type = ScalarArray
    geoh5_type = Data
    _attribute_map: dict = {"array": "values"}

    def from_omf(self, element, **kwargs) -> dict:
        kwargs = self.collect_omf_attributes(element, **kwargs)
        return kwargs

    def from_geoh5(self, entity, **kwargs) -> np.ndarray:
        kwargs = self.collect_h5_attributes(entity, **kwargs)
        return kwargs["array"]


class StringArrayConversion(DataConversion):
    """
    Conversion between :obj:`omf.data.ScalarArray` and
    :obj:`geoh5py.data.Data.values`
    """

    omf_type = StringArray
    geoh5_type = list
    _attribute_map: dict = {"array": "values"}


class ReferenceMapConversion(BaseConversion):

    geoh5_type = ReferencedData

    def collect_omf_attributes(self, element, **kwargs) -> dict:
        value_map = {count + 1: val for count, val in enumerate(element[0].values)}
        color_map = np.vstack(
            [np.r_[count + 1, val, 1.0] for count, val in enumerate(element[1].values)]
        )
        kwargs["value_map"] = value_map
        kwargs["type"] = "referenced"
        kwargs["color_map"] = color_map
        return kwargs


class MappedDataConversion(DataConversion):
    """
    Conversion from :obj:`omf.data.MappedData` to :obj:`geoh5py.data.referenced_data`
    """

    omf_type = MappedData
    geoh5_type = ReferencedData
    _attribute_map = DataConversion._attribute_map.copy()
    _attribute_map.update(
        {"indices": ValuesConversion, "legends": ReferenceMapConversion}
    )


class ColormapConversion(BaseConversion):
    """
    Conversion from :obj:`omf.data.ColorMap` :obj:`numpy.ndarray`
    """

    omf_type = ScalarColormap
    geoh5_type = Data
    _attribute_map: dict = {"colormap": "color_map"}

    def collect_omf_attributes(self, element, **kwargs) -> dict:
        colors = np.vstack(element.gradient.array)
        values = np.linspace(element.limits[0], element.limits[1], colors.shape[0])

        kwargs["color_map"] = np.c_[values, colors, np.ones_like(values)]
        return kwargs

    def from_geoh5(self, entity, **kwargs) -> UidModel | None:

        if getattr(entity.entity_type, "color_map", None) is not None:
            cmap = entity.entity_type.color_map
            ind = np.argsort(cmap.values[0, :])
            values = cmap.values[0, ind]
            limits = [values[0], values[-1]]
            colors = cmap.values[1:-1, ind]  # Drop val and alpha

            if colors.shape[1] != 128:
                new_vals = np.linspace(limits[0], limits[1], 128)
                c_array = []
                for vec in colors.tolist():
                    c_array += [np.interp(new_vals, values, vec)]

                colors = np.vstack(c_array)

            color_array = ColorArray(
                array=[tuple(row) for row in colors.T.astype(int).tolist()]
            )

            return ScalarColormap(limits=limits, gradient=color_array)

        return None


class ScalarDataConversion(DataConversion):

    omf_type = ScalarData
    geoh5_type = FloatData
    _attribute_map = DataConversion._attribute_map.copy()
    _attribute_map.update({"array": ValuesConversion, "colormap": ColormapConversion})


class BaseGeometryConversion(BaseConversion):
    """
    Base geometry operations.
    """

    def collect_omf_attributes(self, element, **kwargs):
        for key, alias in self._attribute_map.items():
            kwargs[alias] = np.vstack(getattr(element, key))

        return kwargs


class PointSetGeometryConversion(BaseGeometryConversion):
    """
    Conversion between :obj:`omf.pointset.PointSetGeometry` and
    :obj:`geoh5py.objects.Points.vertices`
    """

    omf_type = PointSetGeometry
    geoh5_type = Points
    _attribute_map: dict = {"vertices": "vertices"}


class LineSetGeometryConversion(BaseGeometryConversion):
    """
    Conversion between :obj:`omf.lineset.LineSetElement` and
    :obj:`geoh5py.objects.Curve` `vertices` and `cells`
    """

    omf_type = LineSetGeometry
    geoh5_type = Curve
    _attribute_map: dict = {"vertices": "vertices", "segments": "cells"}


class SurfaceGeometryConversion(BaseGeometryConversion):
    """
    Conversion between :obj:`omf.lineset.LineSetElement` and
    :obj:`geoh5py.objects.Curve` `vertices` and `cells`
    """

    omf_type = SurfaceGeometry
    geoh5_type = Surface
    _attribute_map: dict = {"vertices": "vertices", "triangles": "cells"}


class SurfaceGridGeometryConversion(BaseGeometryConversion):
    """
    Conversion between :obj:`omf.lineset.LineSetElement` and
    :obj:`geoh5py.objects.Curve` `vertices` and `cells`
    """

    omf_type = SurfaceGridGeometry
    geoh5_type = Grid2D
    _attribute_map: dict = {
        "u": "u",
        "v": "v",
    }

    def collect_omf_attributes(self, element, **kwargs):
        """Convert attributes from omf to geoh5."""
        if element.axis_v[-1] != 0:
            raise OMFtoGeoh5NotImplemented(
                f"{SurfaceGridGeometry} with 3D rotation axes."
            )

        for key, alias in self._attribute_map.items():
            tensor = getattr(element, f"tensor_{key}")
            if len(np.unique(tensor)) > 1:
                raise OMFtoGeoh5NotImplemented(
                    f"{SurfaceGridGeometry} with variable cell sizes along the {key} axis."
                )

            kwargs.update(
                {f"{alias}_cell_size": tensor[0], f"{alias}_count": len(tensor)}
            )

        azimuth = (
            450 - np.rad2deg(np.arctan2(element.axis_v[1], element.axis_v[0]))
        ) % 360

        if azimuth != 0:
            kwargs.update({"rotation": azimuth})

        if element.axis_u[-1] != 0:
            dip = np.rad2deg(
                np.arcsin(element.axis_u[-1] / np.linalg.norm(element.axis_u))
            )
            kwargs.update({"dip": dip})

        if element.offset_w is not None:
            warnings.warn(
                str(OMFtoGeoh5NotImplemented("Warped Grid2D with 'offset_w'."))
            )
        return kwargs


class VolumeGridGeometryConversion(BaseGeometryConversion):
    """
    Conversion between :obj:`omf.lineset.LineSetElement` and
    :obj:`geoh5py.objects.Curve` `vertices` and `cells`
    """

    omf_type = VolumeGridGeometry
    geoh5_type = BlockModel
    _attribute_map: dict = {"u": "u", "v": "v", "w": "z"}

    def collect_omf_attributes(self, element, **kwargs) -> dict:

        if not np.allclose(np.cross(element.axis_w, [0, 0, 1]), [0, 0, 0]):
            raise OMFtoGeoh5NotImplemented(
                f"{VolumeGridGeometry} with 3D rotation axes."
            )

        for key, alias in self._attribute_map.items():
            tensor = getattr(element, f"tensor_{key}")
            cell_delimiter = np.r_[0, np.cumsum(tensor)]
            kwargs.update({f"{alias}_cell_delimiters": cell_delimiter})

        azimuth = (
            450 - np.rad2deg(np.arctan2(element.axis_v[1], element.axis_v[0]))
        ) % 360

        if azimuth != 0:
            kwargs.update({"rotation": azimuth})

        kwargs.update({"origin": np.r_[element.origin]})

        return kwargs


class PointsConversion(ElementConversion):
    """
    Conversion between :obj:`omf.pointset.PointSetElement` and
    :obj:`geoh5py.objects.Points`
    """

    omf_type = PointSetElement
    geoh5_type = Points
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = PointSetGeometryConversion


class CurveConversion(ElementConversion):
    """
    Conversion between :obj:`omf.lineset.LineSetElement` and
    :obj:`geoh5py.objects.Curve`
    """

    omf_type = LineSetElement
    geoh5_type: Curve
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = LineSetGeometryConversion


class SurfaceConversion(ElementConversion):
    """
    Conversion between :obj:`omf.lineset.SurfaceElement` and
    :obj:`geoh5py.objects.Surface`
    """

    omf_type = SurfaceElement
    geoh5_type: Surface
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = SurfaceGeometryConversion


class SurfaceGridConversion(ElementConversion):
    """
    Conversion between :obj:`omf.lineset.SurfaceElement` and
    :obj:`geoh5py.objects.Grid2D`
    """

    omf_type = SurfaceElement
    geoh5_type: Grid2D
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = SurfaceGridGeometryConversion


class VolumeConversion(ElementConversion):
    """
    Conversion between :obj:`omf.volume.VolumeElement` and
    :obj:`geoh5py.objects.BlockModel`
    """

    omf_type = VolumeElement
    geoh5_type: BlockModel
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = VolumeGridGeometryConversion


@contextmanager
def fetch_h5_handle(file: str | Workspace | Path, mode: str = "a") -> Workspace:
    """
    Open in read+ mode a geoh5 file from string.
    If receiving a file instead of a string, merely return the given file.

    :param file: Name or handle to a geoh5 file.
    :param mode: Set the h5 read/write mode

    :return h5py.File: Handle to an opened h5py file.
    """
    if isinstance(file, Workspace):
        try:
            yield file
        finally:
            pass
    else:
        if Path(file).suffix != ".geoh5":
            raise ValueError("Input h5 file must have a 'geoh5' extension.")

        h5file = Workspace(file, mode)

        try:
            yield h5file
        finally:
            h5file.close()


_ASSOCIATION_MAP = {
    Curve: "segments",
    Surface: "faces",
    Grid2D: "faces",
    BlockModel: "cells",
}

_CLASS_MAP = {
    PointSetGeometry: Points,
    LineSetGeometry: Curve,
    SurfaceGeometry: Surface,
    SurfaceGridGeometry: Grid2D,
    VolumeGridGeometry: BlockModel,
}

_CONVERSION_MAP = {
    BlockModel: VolumeConversion,
    Curve: CurveConversion,
    FloatData: ScalarDataConversion,
    Grid2D: SurfaceGridConversion,
    Int2Array: ValuesConversion,
    LineSetElement: CurveConversion,
    MappedData: MappedDataConversion,
    Points: PointsConversion,
    PointSetElement: PointsConversion,
    Project: ProjectConversion,
    RootGroup: ProjectConversion,
    ScalarArray: ValuesConversion,
    ScalarColormap: ColormapConversion,
    ScalarData: ScalarDataConversion,
    Surface: SurfaceConversion,
    SurfaceElement: SurfaceConversion,
    Vector3Array: ValuesConversion,
    VolumeElement: VolumeConversion,
}
