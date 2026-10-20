# mccl/src/mccl/utils/logging.py
"""Logger centralisé avec format homogène."""
from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

# Cache des loggers déjà configurés pour éviter les handlers en double
_CONFIGURED: set = set()


def get_logger(
    name: str = "mccl",
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> logging.Logger:
    """Renvoie un logger configuré (idempotent).

    :param name: nom du logger (généralement __name__)
    :param level: niveau de log
    :param fmt: format des messages
    :param datefmt: format de la date
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt or _DEFAULT_FORMAT,
        datefmt=datefmt or _DEFAULT_DATEFMT,
    ))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    _CONFIGURED.add(name)
    return logger


def set_level(level: int, name: str = "mccl") -> None:
    """Change le niveau d'un logger."""
    logging.getLogger(name).setLevel(level)


def silence(name: str = "mccl") -> None:
    """Silence un logger."""
    logging.getLogger(name).setLevel(logging.CRITICAL)