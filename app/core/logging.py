"""Logging configuration for LocalRAG."""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application-wide structured logging.

    Args:
        log_level: Logging level as a string (e.g. DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
