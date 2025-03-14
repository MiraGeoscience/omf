"""pointset.py: PointSet element and geometry"""

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2017 Global Mining Standards and Guidelines Group             '
#                                                                              '
#  This file is part of mira-omf package.                                      '
#                                                                              '
#  mira-omf is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                 '
#                                                                              '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import properties

from .base import ProjectElement, ProjectElementGeometry
from .data import Vector3Array
from .texture import ImageTexture


class PointSetGeometry(ProjectElementGeometry):
    """Contains spatial information of a point set"""

    vertices = properties.Instance(
        "Spatial coordinates of points relative to point set origin", Vector3Array
    )

    _valid_locations = ("vertices",)

    def location_length(self, location):
        """Return correct data length based on location"""
        return self.num_nodes

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return len(self.vertices)

    @property
    def num_cells(self):
        """Number of cell centers (same as nodes)"""
        return self.num_nodes


class PointSetElement(ProjectElement):
    """Contains mesh, data, textures, and options of a point set"""

    geometry = properties.Instance(
        "Structure of the point set element", instance_class=PointSetGeometry
    )
    textures = properties.List(
        "Images mapped on the element",
        prop=ImageTexture,
        required=False,
        default=list,
    )
    subtype = properties.StringChoice(
        "Category of PointSet",
        choices=("point", "collar", "blasthole"),
        default="point",
    )
