# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

import logging
import sys
from typing import TextIO


def configure_logging(
    level: str | int = "INFO",
    *,
    stream: TextIO | None = None,
    format_string: str | None = None,
) -> None:
    """Configure root logging once (idempotent for repeated calls in tests)."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = format_string or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers[0].setLevel(level)
