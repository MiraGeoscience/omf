import sys
from pathlib import Path

from omf.fileio import OMFWriter
from omf.fileio.geoh5 import GeoH5Reader


def run():
    geoh5_filepath = Path(sys.argv[1])
    if len(sys.argv) < 3:
        output_filepath = geoh5_filepath.with_suffix(".omf")
    else:
        output_filepath = Path(sys.argv[2])
    reader = GeoH5Reader(geoh5_filepath)
    OMFWriter(reader(), output_filepath)


if __name__ == "__main__":
    run()
