# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from opfv.domain.regression_model_time import RegressionModelTimeStructure


def calculate_hat_f_train_and_eval(
    phi_scalar_func_for_OPFV: Callable[..., float],
    val_bandit_data: dict,
    dataset,
    time_at_eval_vec: np.ndarray,
    round: int,
) -> tuple[np.ndarray, np.ndarray]:
    phi_vector_func = np.vectorize(phi_scalar_func_for_OPFV)
    time_structure = phi_vector_func(val_bandit_data["time"])

    reg_model_time_structure = RegressionModelTimeStructure(
        n_actions=dataset.n_actions,
        action_context=val_bandit_data["action_context"],
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=12345 + round
        ),
    )

    hat_g_x_phi_t_a = reg_model_time_structure.fit_predict(
        context=val_bandit_data["context"],
        time_structure=time_structure,
        action=val_bandit_data["action"],
        reward=val_bandit_data["reward"],
        n_folds=2,
        random_state=12345 + round,
    )

    hat_g_x_phi_t_a = np.squeeze(hat_g_x_phi_t_a, axis=2)
    hat_f_x_t_a = hat_g_x_phi_t_a

    hat_g_x_phi_t_a_at_eval = reg_model_time_structure.predict(
        context=val_bandit_data["context"],
        time_structure=phi_vector_func(time_at_eval_vec),
    )
    hat_g_x_phi_t_a_at_eval = np.squeeze(hat_g_x_phi_t_a_at_eval, axis=2)
    hat_f_x_t_a_at_eval = hat_g_x_phi_t_a_at_eval

    return hat_f_x_t_a, hat_f_x_t_a_at_eval
