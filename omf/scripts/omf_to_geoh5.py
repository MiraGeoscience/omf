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
            "Cowardly refuses to overwrite existing file '%s'.", output_filepath
        )
        sys.exit(1)

    reader = OMFReader(str(omf_filepath.absolute()))
    GeoH5Writer(reader.get_project(), output_filepath)
    _logger.info("geoh5 file created: %s", output_filepath)


if __name__ == "__main__":
    run()
