from os import path

from geoh5py.workspace import Workspace

from omf.fileio import OMFReader, OMFWriter
from omf.fileio.geoh5 import GeoH5Reader, GeoH5Writer

dir = r"C:\Users\dominiquef\Documents\GIT\mira\geoapps\assets"
file = "FlinFlon.geoh5"

# with Workspace(path.join(dir, file)) as ws:
#     omf_entity =
project = GeoH5Reader(path.join(dir, file)).project

OMFWriter(project, path.join(dir, "FlinFlon.omf"))
# reader = OMFReader(path.join(dir, file))
# proj = reader.get_project()
#
# out = GeoH5Writer(proj, path.join(dir, "out.geoh5"))
#
# project = omf.fileio.geoh5.GeoH5Reader(file).project
#     omf_vol = project.elements[0]
