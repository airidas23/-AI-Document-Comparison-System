"""Centralized logging setup for the project."""
from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: int = logging.INFO, logfile: Optional[str] = None) -> None:
    """Configure root logger with console (and optional file) handlers."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers)


logger = logging.getLogger("docdiff")
