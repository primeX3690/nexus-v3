"""
Thin wrapper around stdlib logging so every module gets consistent
formatting without pulling in a logging framework dependency.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger