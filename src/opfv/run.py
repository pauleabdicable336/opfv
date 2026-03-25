# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from opfv.experiments.real_kuairec import KuairecTunePhiExperiment
from opfv.experiments.synthetic_ope import SyntheticOPEExperiment
from opfv.experiments.synthetic_opl import SyntheticOPLEperiment
from opfv.logging_config import configure_logging

logger = logging.getLogger(__name__)

_CONF_DIR = str(Path(__file__).resolve().parent / "conf")

# Experiment YAML `name:` values that are synthetic OPE but do not start with ``synthetic_ope``.
_SYNTHETIC_OPE_ALIASES: frozenset[str] = frozenset({"quick_synthetic_ope"})


def _dispatch(cfg: DictConfig) -> None:
    name = str(cfg.name)
    if name.startswith("synthetic_opl"):
        SyntheticOPLEperiment(cfg).run()
    elif name.startswith("real_kuairec"):
        KuairecTunePhiExperiment(cfg).run()
    elif name.startswith("synthetic_ope") or name in _SYNTHETIC_OPE_ALIASES:
        SyntheticOPEExperiment(cfg).run()
    else:
        raise ValueError(f"Unknown experiment name={name!r}")


@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    _dispatch(cfg)


if __name__ == "__main__":
    main()
