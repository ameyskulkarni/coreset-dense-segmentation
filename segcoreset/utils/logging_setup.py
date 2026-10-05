"""Plain python logging config, shared by every script."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging to stdout with a `HH:MM:SS [LEVEL] logger: message` format.

    Called once at the top of every script's `main()`. Uses `logging.basicConfig`, so it is a
    no-op if the root logger already has handlers.

    Args:
        level: Minimum level to emit (default `logging.INFO`).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
