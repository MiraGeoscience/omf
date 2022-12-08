import logging
import sys
from pathlib import Path

from omf.fileio import OMFReader
from omf.fileio.geoh5 import GeoH5Writer

_logger = logging.getLogger(__package__)


def run():
    omf_filepath = Path(sys.argv[1])
    if len(sys.argv) < 3:
        output_filepath = omf_filepath.with_suffix(".geoh5")
    else:
        output_filepath = Path(sys.argv[2])
        print(output_filepath.suffix)
        if not output_filepath.suffix:
            output_filepath = output_filepath.with_suffix(".geoh5")
    if output_filepath.exists():
        _logger.error(
            f"Cowardly refuses to overwrite existing file '{output_filepath}'."
        )
        exit(1)

    reader = OMFReader(str(omf_filepath.absolute()))
    GeoH5Writer(reader.get_project(), output_filepath)
    _logger.info(f"geoh5 file created: {output_filepath}")


if __name__ == "__main__":
    run()
