# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Shared experiment contract for Hydra-driven runs."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    def log_start(self) -> None:
        logger.info("Starting experiment name=%s sweep=%s", self.cfg.name, self.cfg.sweep)
