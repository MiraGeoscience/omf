import sys
from pathlib import Path

from omf.fileio import OMFReader
from omf.fileio.geoh5 import GeoH5Writer


def run():
    omf_filepath = Path(sys.argv[1])
    if len(sys.argv) < 2:
        output_filepath = omf_filepath.with_suffix(".geoh5")
    else:
        output_filepath = Path(sys.argv[2])
    reader = OMFReader(omf_filepath.absolute())
    GeoH5Writer(reader.get_project(), output_filepath)


if __name__ == "__main__":
    run()
