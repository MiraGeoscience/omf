import logging
import sys


def _create_logger():
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    ok_handler = logging.StreamHandler(sys.stdout)
    ok_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(__package__)
    logger.setLevel(logging.INFO)
    logger.addHandler(ok_handler)
    logger.addHandler(error_handler)

    class OkFilter(logging.Filter):
        def filter(self, record):
            return record.levelno < logging.ERROR

    ok_handler.addFilter(OkFilter())


_create_logger()
