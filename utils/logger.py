"""
utils/logger.py
Centralized logging configuration for the application.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config import LOG_DIR, LOG_FORMAT, LOG_LEVEL

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger instance. Loggers are cached so repeated
    calls with the same name return the same instance (avoids duplicate
    handlers).

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(LOG_LEVEL)
        logger.addHandler(console_handler)

        # File handler
        log_file = Path(LOG_DIR) / "landcover_app.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(LOG_LEVEL)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
