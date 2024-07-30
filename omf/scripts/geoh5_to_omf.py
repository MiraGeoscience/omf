import argparse
import logging
import sys
from pathlib import Path

from omf.fileio import OMFWriter
from omf.fileio.geoh5 import GeoH5Reader


_logger = logging.getLogger(__package__ + "." + Path(__file__).stem)


def main():
    parser = argparse.ArgumentParser(
        prog="geoh5_to_omf",
        description="Converts a geoh5 file to a new OMF file.",
    )
    parser.add_argument("geoh5_file", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=False, default=None)
    args = parser.parse_args()

    geoh5_filepath = args.geoh5_file
    if args.out is None:
        output_filepath = geoh5_filepath.with_suffix(".omf")
    else:
        output_filepath = args.out
        if not output_filepath.suffix:
            output_filepath = output_filepath.with_suffix(".omf")
    if output_filepath.exists():
        _logger.error(
            "Cowardly refuses to overwrite existing file '%s'.", output_filepath
        )
        sys.exit(1)

    reader = GeoH5Reader(geoh5_filepath)
    OMFWriter(reader(), str(output_filepath.absolute()))
    _logger.info("OMF file created: %s", output_filepath)


if __name__ == "__main__":
    main()  # pragma: no cover
