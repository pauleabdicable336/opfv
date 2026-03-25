# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
#
# Modifications Copyright (c) 2026 Tatsuhiro Shimizu


import datetime
import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.utils import check_random_state

from opfv.synthetic_fopl.settings import SyntheticFOPLSettings


def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: np.ndarray) -> np.ndarray:
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


@dataclass
class RegBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.context.shape[0] == self.action.shape[0] == self.reward.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class GradientBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class Prognosticatordataset(torch.utils.data.Dataset):
    context: np.ndarray
    time: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    # q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.time.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            # == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.time[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            # self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class OPFVDataset(torch.utils.data.Dataset):
    context: np.ndarray
    time: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.time.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.time[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]


logger = logging.getLogger(__name__)

fromtimestamp_vec = np.vectorize(datetime.datetime.fromtimestamp)


def show_hyperparameters(
    settings: SyntheticFOPLSettings,
    time_at_evaluation_start: int | None = None,
    time_at_evaluation_end: int | None = None,
    flag_show_time_at_evaluation: bool = True,
    time_at_evaluation_list: list | None = None,
) -> None:
    st = settings
    logger.info("################# START hyperparameters (F-OPL) #################")
    logger.info("n_seeds=%s num_train=%s num_test=%s", st.n_seeds, st.num_train, st.num_test)
    logger.info("|C_r|=%s lambda=%s", st.num_time_structure_for_logged_data, st.lambda_ratio)
    logger.info(
        "OPL epochs=%s batch=%s num_time_learn=%s", st.max_iter, st.batch_size, st.num_time_learn
    )
    logger.info("Prognosticator phi=%s", st.phi_scalar_func_list)
    logger.info(
        "t_oldest=%s t_now=%s t_future=%s",
        datetime.datetime.fromtimestamp(st.t_oldest),
        datetime.datetime.fromtimestamp(st.t_now),
        datetime.datetime.fromtimestamp(st.t_future),
    )
    if flag_show_time_at_evaluation and time_at_evaluation_start is not None:
        logger.info(
            "eval window start=%s end=%s",
            datetime.datetime.fromtimestamp(time_at_evaluation_start),
            datetime.datetime.fromtimestamp(time_at_evaluation_end),
        )
    logger.info("|A|=%s d_x=%s beta=%s eps=%s", st.n_actions, st.dim_context, st.beta, st.eps)
    logger.info("num_train_list=%s lambda_list=%s", st.num_train_list, st.lambda_ratio_list)
    if not flag_show_time_at_evaluation and time_at_evaluation_list is not None:
        logger.info("time_at_evaluation_list=%s", fromtimestamp_vec(time_at_evaluation_list))
    logger.info("################# END hyperparameters #################")
