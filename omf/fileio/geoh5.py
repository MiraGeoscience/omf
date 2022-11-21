# pylint: disable=too-many-lines

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from geoh5py.data import Data, FloatData, IntegerData, ReferencedData
from geoh5py.groups import RootGroup
from geoh5py.objects import BlockModel, Curve, Grid2D, ObjectBase, Points, Surface
from geoh5py.shared import FLOAT_NDV, INTEGER_NDV, Entity
from geoh5py.workspace import Workspace

from omf.base import Project, UidModel
from omf.data import (
    ColorArray,
    Int2Array,
    Legend,
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
    OMF to geoh5 file converter.

    :param element: Input :obj:`omf.base.UidModel` element to be converted.
    :param file_name: Input file name with *.geoh5 extension.
    """

    def __init__(self, element: UidModel, file_name: str | Path):

        if not isinstance(file_name, (str, Path)):
            raise TypeError("Input 'file' must be of str or Path.")

        self.file = file_name
        self.entity = element
        self.element = element

    @property
    def entity(self):
        """Pointer to a converted :obj:`geoh5py.shared.entity.Entity`."""
        return self._entity

    @entity.setter
    def entity(self, element):
        converter = get_conversion_map(element, self.file)
        self._entity = converter.from_omf(element)

    def __call__(self):
        return self.entity.workspace


def get_conversion_map(
    element: UidModel | Entity, workspace: str | Path | Workspace, parent=None
):
    """
    Utility method to get the appropriate conversion class is it exists.

    :param element: Either an omf or geoh5 class.
    :param workspace: Path to a geoh5 or active :obj:`geoh5py.workspace.Workspace`.
    :param parent: Optional parent object used for conversion.

    :returns: A sub-class of BaseConversion for the provided element.
    """
    if type(element) not in _CONVERSION_MAP:
        raise OMFtoGeoh5NotImplemented(
            f"Element of type {type(element)} currently not implemented."
        )

    # Special case for SurfaceElement describing Grid2D
    if isinstance(element, SurfaceElement) and isinstance(
        element.geometry, SurfaceGridGeometry
    ):
        return SurfaceGridConversion(element, workspace, parent=parent)

    return _CONVERSION_MAP[type(element)](element, workspace, parent=parent)


class GeoH5Reader:  # pylint: disable=too-few-public-methods
    """
    Geoh5 to omf class converter

    :param file_name: Input file name with *.geoh5 extension.
    """

    def __init__(self, file_name: str | Path):

        with Workspace(file_name, mode="r") as workspace:
            self.file = workspace
            converter = ProjectConversion(workspace.root, self.file)
            self.project = converter.from_geoh5(workspace.root)

    def __call__(self) -> Project:
        return self.project


class BaseConversion(ABC):
    """
    Base conversion between OMF and geoh5 format.

    :param element: Either an omf or geoh5 class.
    :param geoh5: Path to a geoh5 or active :obj:`geoh5py.workspace.Workspace`.
    :param parent: (Optional) Parental object
    """

    geoh5: str | Path | Workspace = None
    geoh5_type: type[Entity]
    omf_type: type[UidModel]
    _attribute_map: dict = {
        "uid": "uid",
        "name": "name",
    }
    _parent = None
    _mapping = None

    def __init__(self, element, geoh5: str | Path | Workspace, parent=None):
        if element is None:
            raise ValueError("Input 'element' cannot be None.")

        self.geoh5 = geoh5
        self._parent = parent

    @abstractmethod
    def from_omf(self, element, **kwargs) -> Entity | None:
        """Convert omf element to geoh5 entity."""

    @abstractmethod
    def from_geoh5(self, entity, **kwargs) -> UidModel | list | None:
        """Generate an omf element from geoh5 attributes."""

    @staticmethod
    def process_dependents(
        element: UidModel | Entity, parent: UidModel | Entity, workspace
    ) -> list:
        """
        Convert the children elements or entities.

        :param element: Either an omf or geoh5 class.
        :param parent: Parental omf or geoh5 class.
        :param workspace: Path to a geoh5 or active :obj:`geoh5py.workspace.Workspace`.

        :return: List of children UiDModel or Entity objects.
        """
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
                converter = get_conversion_map(child, workspace, parent=element)
                children += [getattr(converter, method)(child, **kwargs)]
            except OMFtoGeoh5NotImplemented as error:
                warnings.warn(error.args[0])
                continue

        return children

    def collect_attributes(self, element, workspace, **kwargs) -> dict:
        """
        Collect and convert attributes needed to construct an omf or geoh5 object.

        :param element: Either an omf or geoh5 class.
        :param workspace: Path to a geoh5 or active :obj:`geoh5py.workspace.Workspace`.
        :param kwargs: Input dictionary of attributes to be appended.

        :return: Updated arguments.
        """

        with fetch_h5_handle(workspace):
            for key, alias in self._attribute_map.items():

                if inspect.isclass(alias) and issubclass(alias, BaseConversion):
                    conversion = alias(  # pylint: disable=not-callable
                        element, workspace, parent=self._parent
                    )
                    kwargs = conversion.collect_attributes(element, workspace, **kwargs)
                else:
                    if isinstance(element, UidModel):
                        prop = getattr(element, key, None)
                        label = alias
                    else:
                        prop = getattr(element, alias, None)
                        label = key

                    if prop is None:
                        continue

                    kwargs[label] = prop

        return kwargs


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
        """
        Convert :obj:`omf.data.Data` to :obj:`geoh5.data.Data` entity.

        :param element: Input :obj:`omf.data.Data` object.
        """

        with fetch_h5_handle(self.geoh5) as workspace:

            kwargs = self.collect_attributes(element, workspace, **kwargs)
            parent = kwargs.pop("parent", None)

            if not isinstance(parent, ObjectBase):
                raise UserWarning(
                    "Input argument for DataConversion.from_omf requires a "
                    "'parent' of type ObjectBase."
                )

            if element.location in ["faces", "cells", "segments"]:
                kwargs["association"] = "CELL"
            else:
                kwargs["association"] = "VERTEX"

            colormap = kwargs.pop("color_map", None)
            entity = parent.add_data({element.name: kwargs})

            if colormap is not None:
                entity.entity_type.color_map = colormap

        return entity

    def from_geoh5(self, entity, **kwargs) -> UidModel | list:
        """
        Convert :obj:`geoh5.data.Data` to :obj:`omf.data.Data` entity.

        :param entity: Input :obj:`geoh5.data.Data` entity.
        """
        with fetch_h5_handle(self.geoh5) as workspace:
            kwargs = self.collect_attributes(entity, workspace, **kwargs)
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

    def __init__(self, obj: UidModel | Entity, geoh5: str | Path | Workspace, **kwargs):
        super().__init__(obj, geoh5, **kwargs)

        if isinstance(obj, UidModel) and hasattr(obj, "geometry"):
            self.geoh5_type = _CLASS_MAP[type(obj.geometry)]

    def from_omf(self, element, **kwargs) -> Entity | None:
        """
        Convert :obj:`omf.base.Element` object to :obj:`geoh5.objects` class.

        :param element: Input :obj:`omf.base.Element`object.
        """
        with fetch_h5_handle(self.geoh5) as workspace:
            try:
                kwargs = self.collect_attributes(element, workspace, **kwargs)
            except OMFtoGeoh5NotImplemented as error:
                warnings.warn(error.args[0])
                return None

            entity = workspace.create_entity(self.geoh5_type, **{"entity": kwargs})
            self.process_dependents(element, entity, workspace)

        return entity

    def from_geoh5(self, entity: ObjectBase, **kwargs) -> UidModel:
        """
        Convert :obj:`omf.base.Element` object to :obj:`geoh5.objects` class.

        :param entity: Input :obj:`geoh5.objects` class.
        """
        with fetch_h5_handle(self.geoh5) as workspace:
            kwargs = self.collect_attributes(entity, workspace, **kwargs)
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

    def from_omf(self, element: Project, **kwargs) -> RootGroup:
        """Convert omf project to geoh5 root."""
        with fetch_h5_handle(self.geoh5) as workspace:
            kwargs = self.collect_attributes(element, workspace, **kwargs)
            root: RootGroup = workspace.root

            for key, value in kwargs.items():
                setattr(root, key, value)

            self.process_dependents(element, root, workspace)

        return root

    def from_geoh5(self, entity: RootGroup, **kwargs) -> Project:
        """Convert RootGroup to omf Project."""
        with fetch_h5_handle(self.geoh5) as workspace:
            kwargs = self.collect_attributes(entity, workspace, **kwargs)
            uid = kwargs.pop("uid")
            project = self.omf_type(**kwargs)
            getattr(project, "_backend").update({"uid": uid})  # pylint: disable=W0212
            project.elements = self.process_dependents(entity, project, workspace)

        return project


class ArrayConversion(BaseConversion):
    """
    Conversion from :obj:`omf.data.Int2Array` or `Vector3Array` to :obj:`numpy.ndarray`
    """

    omf_type = ScalarArray
    geoh5_type = Data
    _attribute_map: dict = {"array": "values"}

    def from_omf(self, element, **kwargs):
        """Convert omf element to geoh5 entity."""
        kwargs = self.collect_attributes(element, self.geoh5)

        return kwargs

    def from_geoh5(self, entity, **kwargs):
        """Generate an omf element from geoh5 attributes."""
        kwargs = self.collect_attributes(entity, self.geoh5)

        return kwargs

    def collect_attributes(self, element, workspace, **kwargs) -> dict:
        with fetch_h5_handle(workspace):
            if isinstance(element, UidModel):
                values = np.r_[getattr(element, "array")]

                if np.issubdtype(values.dtype, np.floating):
                    values[np.isclose(values, FLOAT_NDV)] = np.nan
                    values = values.astype(np.float32)
                else:
                    values = values.astype(np.int32)

            else:
                values = getattr(element, "values", None)

                if np.issubdtype(values.dtype, np.floating):
                    values[np.isnan(values)] = FLOAT_NDV
                else:
                    values[np.isclose(values, INTEGER_NDV)] = 0

            if values is not None:
                conversion = _VALUE_MAP.get(type(self._parent), None)
                if conversion is not None:
                    values = conversion(self._parent, values)

                if isinstance(element, UidModel):
                    kwargs.update({"values": values.copy()})
                else:
                    kwargs.update({"array": values.copy()})
            return kwargs


class IndicesConversion(ArrayConversion):
    """
    Conversion from :obj:`omf.data.Scalar` of indices to :obj:`numpy.ndarray`
    handling the conversion for 'unknown': -1 <-> 0
    """

    def collect_attributes(self, element, workspace, **kwargs) -> dict:
        with fetch_h5_handle(workspace):
            if isinstance(element, UidModel):
                values = np.r_[getattr(element, "array")]
            else:
                values = getattr(element, "values", None)
                values[np.isclose(values, INTEGER_NDV)] = 0

            if values is not None:
                conversion = _VALUE_MAP.get(type(self._parent), None)
                if conversion is not None:
                    values = conversion(self._parent, values)

                if isinstance(element, UidModel):
                    kwargs.update({"values": (values + 1).astype(int)})
                else:
                    values = (values - 1).astype(int)
                    kwargs.update({"indices": values})
            return kwargs


class StringArrayConversion(ArrayConversion):
    """
    Conversion between :obj:`omf.data.ScalarArray` and
    :obj:`geoh5py.data.Data.values`
    """

    omf_type = StringArray
    geoh5_type = list


class ReferenceMapConversion(ArrayConversion):
    """
    Convert between :obj:`omf.data.MappedData.legends` to attributes
    of :obj:`geoh5py.data.referenced_data`.
    """

    geoh5_type = ReferencedData

    def collect_attributes(self, element, workspace, **kwargs):
        if isinstance(element, MappedData):
            kwargs = self.collect_omf_attributes(element, **kwargs)
        else:
            kwargs = self.collect_h5_attributes(element, workspace, **kwargs)
        return kwargs

    @staticmethod
    def collect_omf_attributes(element, **kwargs) -> dict:
        if not element.legends:
            return kwargs

        value_map = {
            count + 1: str(val) for count, val in enumerate(element.legends[0].values)
        }
        color_map = np.vstack(
            [
                np.r_[count + 1, val, 1.0]
                for count, val in enumerate(element.legends[1].values)
            ]
        )
        kwargs["value_map"] = value_map
        kwargs["type"] = "referenced"
        kwargs["color_map"] = color_map

        return kwargs

    @staticmethod
    def collect_h5_attributes(element, workspace, **kwargs) -> dict:
        with fetch_h5_handle(workspace):
            labels = list(element.value_map.map.values())
            ind = 0
            if "Unknown" in labels:
                ind = 1
                labels.remove("Unknown")

            kwargs.update(
                {
                    "legends": [
                        Legend(values=StringArray(array=labels)),
                        Legend(
                            values=ColorArray(
                                array=element.entity_type.color_map.values[1:-1, ind:]
                                .astype(int)
                                .reshape((3, -1))
                                .T.tolist()
                            )
                        ),
                    ]
                }
            )
        return kwargs


class MappedDataConversion(DataConversion):
    """
    Conversion from :obj:`omf.data.MappedData` to :obj:`geoh5py.data.referenced_data`
    """

    omf_type = MappedData
    geoh5_type = ReferencedData
    _attribute_map = DataConversion._attribute_map.copy()
    _attribute_map.update(
        {"indices": IndicesConversion, "legends": ReferenceMapConversion}
    )


class ColormapConversion(ArrayConversion):
    """
    Conversion between :obj:`omf.data.ColorMap` and
    :obj:`geoh5py.share.entity_type.color_map` attribute.
    """

    omf_type = ScalarColormap
    geoh5_type = Data
    _attribute_map: dict = {"colormap": "color_map"}

    def collect_attributes(self, element, workspace, **kwargs):
        if isinstance(element, UidModel):
            kwargs = self.collect_omf_attributes(element, **kwargs)
        else:
            kwargs = self.collect_h5_attributes(element, workspace, **kwargs)
        return kwargs

    @staticmethod
    def collect_omf_attributes(element, **kwargs) -> dict:
        colormap = getattr(element, "colormap", None)
        if colormap is None:
            return kwargs

        colors = np.vstack(colormap.gradient.array)
        values = np.linspace(colormap.limits[0], colormap.limits[1], colors.shape[0])

        kwargs["color_map"] = np.c_[values, colors, np.ones_like(values)]
        return kwargs

    @staticmethod
    def collect_h5_attributes(element, workspace, **kwargs) -> dict:
        with fetch_h5_handle(workspace):

            if getattr(element.entity_type, "color_map", None) is not None:
                cmap = element.entity_type.color_map
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

                kwargs.update(
                    {"colormap": ScalarColormap(limits=limits, gradient=color_array)}
                )

        return kwargs


class ScalarDataConversion(DataConversion):
    """
    Base conversion for numerical data.
    """

    omf_type = ScalarData
    geoh5_type = FloatData
    _attribute_map = DataConversion._attribute_map.copy()
    _attribute_map.update({"array": ArrayConversion, "colormap": ColormapConversion})


class BaseGeometryConversion(BaseConversion):
    """
    Base geometry operations.
    """

    def from_omf(self, element, **kwargs):
        """Convert omf element to geoh5py attributes."""
        kwargs = self.collect_attributes(element, self.geoh5)
        return kwargs

    def from_geoh5(self, entity, **kwargs):
        """Convert geoh5 entity to omf attributes"""
        kwargs = self.collect_attributes(entity, self.geoh5)
        return kwargs

    def collect_attributes(self, element, workspace, **kwargs):
        if hasattr(element, "geometry"):
            for key, alias in self._attribute_map.items():
                kwargs[alias] = np.vstack(getattr(element.geometry, key))
        else:
            with fetch_h5_handle(workspace):
                geometry = self.omf_type(
                    **{
                        key: getattr(element, alias)
                        for key, alias in self._attribute_map.items()
                    }
                )
                kwargs.update({"geometry": geometry})

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
    Conversion between :obj:`omf.surface.SurfaceGeometry` and
    :obj:`geoh5py.objects.Surface` `vertices` and `cells`
    """

    omf_type = SurfaceGeometry
    geoh5_type = Surface
    _attribute_map: dict = {"vertices": "vertices", "triangles": "cells"}


class SurfaceGridGeometryConversion(BaseGeometryConversion):
    """
    Conversion between :obj:`omf.surface.SurfaceGridGeometry` and
    :obj:`geoh5py.objects.Grid2D` attributes
    """

    omf_type = SurfaceGridGeometry
    geoh5_type = Grid2D
    _attribute_map: dict = {
        "u": "u",
        "v": "v",
    }

    def collect_attributes(self, element, workspace, **kwargs):
        if isinstance(element, SurfaceElement):
            kwargs = self.collect_omf_attributes(element, **kwargs)
        else:
            kwargs = self.collect_h5_attributes(element, workspace, **kwargs)
        return kwargs

    @classmethod
    def collect_omf_attributes(cls, element, **kwargs) -> dict:
        """Convert attributes from omf to geoh5."""
        if element.geometry.axis_v[-1] != 0:
            raise OMFtoGeoh5NotImplemented(
                f"{SurfaceGridGeometry} with 3D rotation axes."
            )

        for key, alias in cls._attribute_map.items():
            tensor = getattr(element.geometry, f"tensor_{key}")
            if len(np.unique(tensor)) > 1:
                raise OMFtoGeoh5NotImplemented(
                    f"{SurfaceGridGeometry} with variable cell sizes along the {key} axis."
                )

            kwargs.update(
                {f"{alias}_cell_size": tensor[0], f"{alias}_count": len(tensor)}
            )

        azimuth = (
            450
            - np.rad2deg(
                np.arctan2(element.geometry.axis_v[1], element.geometry.axis_v[0])
            )
        ) % 360

        if azimuth != 0:
            kwargs.update({"rotation": azimuth})

        if element.geometry.axis_u[-1] != 0:
            dip = np.rad2deg(
                np.arcsin(
                    element.geometry.axis_u[-1]
                    / np.linalg.norm(element.geometry.axis_u)
                )
            )
            kwargs.update({"dip": dip})

        if element.geometry.offset_w is not None:
            warnings.warn(
                str(OMFtoGeoh5NotImplemented("Warped Grid2D with 'offset_w'."))
            )
        return kwargs

    @classmethod
    def collect_h5_attributes(cls, entity, workspace, **kwargs) -> dict:
        with fetch_h5_handle(workspace):
            geometry = {}
            for key, alias in cls._attribute_map.items():
                cell_size, count = getattr(entity, f"{alias}_cell_size"), getattr(
                    entity, f"{alias}_count"
                )
                tensor = np.ones(count) * np.abs(cell_size)
                geometry.update({f"tensor_{key}": tensor})

            if entity.rotation is not None or entity.dip is not None:
                dip = np.deg2rad(getattr(entity, "dip", 0.0))
                azm = np.deg2rad(getattr(entity, "rotation", 0.0))
                rot = rotation_opt(azm, dip)

                geometry["axis_u"] = rot.dot(np.c_[1.0, 0.0, 0.0].T).flatten()
                geometry["axis_v"] = rot.dot(np.c_[0.0, 1.0, 0.0].T).flatten()

            geometry.update(
                {
                    "origin": np.r_[
                        entity.origin["x"], entity.origin["y"], entity.origin["z"]
                    ]
                }
            )
            kwargs.update({"geometry": geometry})
        return kwargs


class VolumeGridGeometryConversion(BaseGeometryConversion):
    """
    Conversion between :obj:`omf.volume.VolumeGridGeometry` and
    :obj:`geoh5py.objects.BlockModel` attributes.
    """

    omf_type = VolumeGridGeometry
    geoh5_type = BlockModel
    _attribute_map: dict = {"u": "u", "v": "v", "w": "z"}

    def __init__(
        self, obj: UidModel | Entity, geoh5: str | Path | Workspace, parent=None
    ):
        super().__init__(obj, geoh5, parent=parent)

    def collect_attributes(self, element, workspace, **kwargs):
        if isinstance(element, VolumeElement):
            kwargs = self.collect_omf_attributes(element, **kwargs)
        else:
            kwargs = self.collect_h5_attributes(element, workspace, **kwargs)
        return kwargs

    @classmethod
    def collect_omf_attributes(cls, element, **kwargs) -> dict:
        if not np.allclose(np.cross(element.geometry.axis_w, [0, 0, 1]), [0, 0, 0]):
            raise OMFtoGeoh5NotImplemented(
                f"{VolumeGridGeometry} with 3D rotation axes."
            )

        for key, alias in cls._attribute_map.items():
            tensor = getattr(element.geometry, f"tensor_{key}")
            cell_delimiter = np.r_[0, np.cumsum(tensor)]
            kwargs.update({f"{alias}_cell_delimiters": cell_delimiter})

        azimuth = (
            450
            - np.rad2deg(
                np.arctan2(element.geometry.axis_v[1], element.geometry.axis_v[0])
            )
        ) % 360

        if azimuth != 0:
            kwargs.update({"rotation": azimuth})

        kwargs.update({"origin": np.r_[element.geometry.origin]})

        return kwargs

    @classmethod
    def collect_h5_attributes(cls, entity, workspace, **kwargs) -> dict:
        with fetch_h5_handle(workspace):
            geometry = {}
            offsets = []
            for key, alias in cls._attribute_map.items():
                cell_delimiter = getattr(entity, f"{alias}_cell_delimiters")
                tensor = np.diff(cell_delimiter)
                offsets.append(np.sum(tensor[tensor < 0]))
                geometry.update({f"tensor_{key}": np.abs(tensor)})

            if entity.rotation is not None:

                azm = np.deg2rad(getattr(entity, "rotation", 0.0))
                rot = rotation_opt(azm, 0.0)

                geometry["axis_u"] = rot.dot(np.c_[1.0, 0.0, 0.0].T).flatten()
                geometry["axis_v"] = rot.dot(np.c_[0.0, 1.0, 0.0].T).flatten()

            geometry.update(
                {
                    "origin": np.r_[
                        entity.origin["x"] + offsets[0],
                        entity.origin["y"] + offsets[1],
                        entity.origin["z"] + offsets[2],
                    ]
                }
            )
            kwargs.update({"geometry": geometry})
        return kwargs


class PointsConversion(ElementConversion):
    """
    Conversion between :obj:`omf.pointset.PointSetElement` and
    :obj:`geoh5py.objects.Points`.
    """

    omf_type = PointSetElement
    geoh5_type = Points
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = PointSetGeometryConversion


class CurveConversion(ElementConversion):
    """
    Conversion between :obj:`omf.lineset.LineSetElement` and
    :obj:`geoh5py.objects.Curve`.
    """

    omf_type = LineSetElement
    geoh5_type: Curve
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = LineSetGeometryConversion


class SurfaceConversion(ElementConversion):
    """
    Conversion between :obj:`omf.lineset.SurfaceElement` and
    :obj:`geoh5py.objects.Surface`.
    """

    omf_type = SurfaceElement
    geoh5_type: Surface
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = SurfaceGeometryConversion


class SurfaceGridConversion(ElementConversion):
    """
    Conversion between :obj:`omf.lineset.SurfaceElement` and
    :obj:`geoh5py.objects.Grid2D`.
    """

    omf_type = SurfaceElement
    geoh5_type: Grid2D
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = SurfaceGridGeometryConversion


class VolumeConversion(ElementConversion):
    """
    Conversion between :obj:`omf.volume.VolumeElement` and
    :obj:`geoh5py.objects.BlockModel`.
    """

    omf_type = VolumeElement
    geoh5_type: BlockModel
    _attribute_map = ElementConversion._attribute_map.copy()
    _attribute_map["geometry"] = VolumeGridGeometryConversion


def rotation_opt(azimuth: float, dip: float):
    """
    Construct a 3D rotation matrix from azimuth and dip angles in radian.

    :param azimuth: Horizontal clockwise angle from North.
    :param dip: Vertical angle from horizontal (positive down).
    """
    r_x = np.r_[
        np.c_[1, 0, 0],
        np.c_[0, np.cos(dip), -np.sin(dip)],
        np.c_[0, np.sin(dip), np.cos(dip)],
    ]
    r_z = np.r_[
        np.c_[np.cos(azimuth), -np.sin(azimuth), 0],
        np.c_[np.sin(azimuth), np.cos(azimuth), 0],
        np.c_[0, 0, 1],
    ]
    return r_z.dot(r_x)


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


def block_model_reordering(entity: BlockModel | VolumeElement, values: np.ndarray):
    """
    Re-ordering of values between :obj:`omf.volume.VolumeElement` and
    :obj:`geoh5py.objects.BlockModel`.

    :param entity: Input object with values.
    :param values: Array of values to be re-ordered.

    :return values: Array of values after re-ordering
    """
    if isinstance(entity, VolumeElement):
        values = values.reshape(
            (
                entity.geometry.tensor_u.shape[0],
                entity.geometry.tensor_v.shape[0],
                entity.geometry.tensor_w.shape[0],
            ),
            order="C",
        )
        values = values.transpose((2, 0, 1))[::-1, :, :].flatten(order="F")

    else:
        values = values.reshape(
            (
                entity.shape[2],
                entity.shape[0],
                entity.shape[1],
            ),
            order="F",
        )[::-1, :, :]
        values = values.transpose((1, 2, 0)).flatten()

    return values


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

_CONVERSION_MAP: dict = {
    BlockModel: VolumeConversion,
    Curve: CurveConversion,
    FloatData: ScalarDataConversion,
    Grid2D: SurfaceGridConversion,
    Int2Array: ArrayConversion,
    IntegerData: ScalarDataConversion,
    LineSetElement: CurveConversion,
    MappedData: MappedDataConversion,
    Points: PointsConversion,
    PointSetElement: PointsConversion,
    Project: ProjectConversion,
    ReferencedData: MappedDataConversion,
    RootGroup: ProjectConversion,
    ScalarArray: ArrayConversion,
    ScalarColormap: ColormapConversion,
    ScalarData: ScalarDataConversion,
    Surface: SurfaceConversion,
    SurfaceElement: SurfaceConversion,
    Vector3Array: ArrayConversion,
    VolumeElement: VolumeConversion,
}

_VALUE_MAP: dict = {
    BlockModel: block_model_reordering,
    VolumeElement: block_model_reordering,
}
