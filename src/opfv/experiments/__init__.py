"""Hydra-driven experiment entrypoints (synthetic and real)."""

from opfv.experiments.real_kuairec import KuairecTunePhiExperiment
from opfv.experiments.synthetic_ope import SyntheticOPEExperiment
from opfv.experiments.synthetic_opl import SyntheticOPLEperiment

__all__ = [
    "KuairecTunePhiExperiment",
    "SyntheticOPEExperiment",
    "SyntheticOPLEperiment",
]
