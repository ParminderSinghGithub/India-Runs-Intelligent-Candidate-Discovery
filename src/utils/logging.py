"""Logging configuration and stage-timing utilities."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from src.config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT


def setup_logging(
    log_level: int = LOG_LEVEL,
    log_format: str = LOG_FORMAT,
    date_format: str = LOG_DATE_FORMAT,
    log_file: Optional[Path] = None,
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


@contextmanager
def stage_log(
    logger: logging.Logger,
    stage_name: str,
    *,
    count_label: str = "",
) -> Generator[None, None, None]:
    """Context manager that logs START, END, and elapsed time for a pipeline stage.

    Usage::

        with stage_log(logger, "Embedding generation", count_label="candidates"):
            embeddings = engine.embed(docs)

    Args:
        logger: Logger instance to use.
        stage_name: Human-readable name of the pipeline stage.
        count_label: Optional label appended to the END log (e.g., "candidates").

    Yields:
        None — simply wraps the block.
    """
    logger.info("[%s] START", stage_name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if count_label:
            logger.info("[%s] END -- elapsed %.2fs (%s)", stage_name, elapsed, count_label)
        else:
            logger.info("[%s] END -- elapsed %.2fs", stage_name, elapsed)
