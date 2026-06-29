"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path

from src.config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT


def setup_logging(
    log_level: int = LOG_LEVEL,
    log_format: str = LOG_FORMAT,
    date_format: str = LOG_DATE_FORMAT,
    log_file: Path | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_format: Format string for log messages.
        date_format: Format string for timestamps.
        log_file: Optional path to log file. If None, logs to stdout only.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )
