"""Single place to configure logging for the package and CLI."""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once; safe to call repeatedly."""
    global _CONFIGURED
    level = logging.DEBUG if verbose else logging.INFO
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
