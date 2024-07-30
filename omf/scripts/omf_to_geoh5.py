import argparse
import logging
import sys
from pathlib import Path

from omf.fileio import OMFReader
from omf.fileio.geoh5 import GeoH5Writer


_logger = logging.getLogger(__package__ + "." + Path(__file__).stem)


def main():
    parser = argparse.ArgumentParser(
        prog="omf_to_geoh5",
        description="Converts an OMF file to a new geoh5 file.",
    )
    parser.add_argument("omf_file", type=Path, help="Path to the OMF file to convert.")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=False,
        default=None,
        help=(
            "Path to the output geoh5 file. If not specified, create the output file "
            "at the same location as the input file, but with the geoh5 extension."
        ),
    )
    parser.add_argument(
        "--gzip",
        type=int,
        choices=range(0, 10),
        default=5,
        help="Gzip compression level (0-9) for h5 data.",
    )
    args = parser.parse_args()

    omf_filepath = args.omf_file
    if args.out is None:
        output_filepath = omf_filepath.with_suffix(".geoh5")
    else:
        output_filepath = args.out
        if not output_filepath.suffix:
            output_filepath = output_filepath.with_suffix(".geoh5")
    if output_filepath.exists():
        _logger.error(
            "Cowardly refuses to overwrite existing file '%s'.", output_filepath
        )
        sys.exit(1)

    reader = OMFReader(str(omf_filepath.absolute()))
    GeoH5Writer(reader.get_project(), output_filepath, compression=args.gzip)
    _logger.info("geoh5 file created: %s", output_filepath)


if __name__ == "__main__":
    main()  # pragma: no cover
