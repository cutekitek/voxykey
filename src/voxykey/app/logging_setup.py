import logging
import os


def setup_logging() -> None:
    level_name = os.getenv("VOXYKEY_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger(__name__).debug("Logging configured level=%s", logging.getLevelName(level))
