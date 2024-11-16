"""omf: API library for Open Mining Format file interchange format"""

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


__version__ = "3.3.0-beta.1"
__author__ = "Global Mining Standards and Guidelines Group"
__license__ = "MIT License"
__copyright__ = "Copyright 2017 Global Mining Standards and Guidelines Group"


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
