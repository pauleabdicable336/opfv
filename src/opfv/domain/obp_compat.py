# Modifications Copyright (c) 2026 Tatsuhiro Shimizu
"""Small helpers that older OBP versions exposed from ``obp.utils`` (API drift)."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state


def check_array(*, array: np.ndarray, name: str, expected_dim: int) -> None:
    arr = np.asarray(array)
    if arr.ndim != expected_dim:
        raise ValueError(f"Expected `{name}` to have dim {expected_dim}, got {arr.ndim}")


def softmax(x: np.ndarray | float) -> np.ndarray | float:
    if isinstance(x, np.ndarray) and x.ndim == 2:
        b = np.max(x, axis=1)[:, np.newaxis]
        numerator = np.exp(x - b)
        denominator = np.sum(numerator, axis=1)[:, np.newaxis]
        return numerator / denominator
    b = float(np.max(x))
    ex = np.exp(np.asarray(x, dtype=float) - b)
    return ex / float(np.sum(ex))


def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions
