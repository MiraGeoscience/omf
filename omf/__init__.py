"""omf: API library for Open Mining Format file interchange format"""

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2022-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of mira-omf package.                                      '
#                                                                              '
#  mira-omf is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                 '
#                                                                              '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2017 Global Mining Standards and Guidelines Group             '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import logging
import sys

from .base import Project
from .data import (
    ColorArray,
    ColorData,
    DateTimeArray,
    DateTimeColormap,
    DateTimeData,
    Legend,
    MappedData,
    ScalarArray,
    ScalarColormap,
    ScalarData,
    StringArray,
    StringData,
    Vector2Array,
    Vector2Data,
    Vector3Array,
    Vector3Data,
)
from .fileio import GeoH5Writer, OMFReader, OMFWriter
from .lineset import LineSetElement, LineSetGeometry
from .pointset import PointSetElement, PointSetGeometry
from .surface import SurfaceElement, SurfaceGeometry, SurfaceGridGeometry
from .texture import ImageTexture
from .volume import VolumeElement, VolumeGridGeometry


try:
    from ._version import __version__
except ModuleNotFoundError:
    from datetime import datetime

    __date_str = datetime.today().strftime("%Y%m%d")
    __version__ = "0.0.0.dev0+" + __date_str

__author__ = "Global Mining Standards and Guidelines Group, Mira Geoscience Ltd."
__license__ = "MIT License"
__copyright__ = (
    "Copyright 2017 Global Mining Standards and Guidelines Group, "
    "Copyright 2022-2025 Mira Geoscience Ltd."
)


def _create_logger():
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    ok_handler = logging.StreamHandler(sys.stdout)
    ok_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(__package__)
    logger.setLevel(logging.INFO)
    logger.addHandler(ok_handler)
    logger.addHandler(error_handler)

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    error_handler.setFormatter(formatter)
    ok_handler.setFormatter(formatter)

    class OkFilter(logging.Filter):
        def filter(self, record):
            return record.levelno < logging.ERROR

    ok_handler.addFilter(OkFilter())


_create_logger()
