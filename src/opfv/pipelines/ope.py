# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import OffPolicyEvaluation, RegressionModel
from sklearn.ensemble import RandomForestRegressor

from opfv.domain.estimators import OPFV, Prognosticator, fourier_scalar, fourier_vec
from opfv.domain.policy import gen_eps_greedy
from opfv.domain.synthetic_bandit import unix_time_to_time_structure_n_tree
from opfv.pipelines.ope_support import calculate_hat_f_train_and_eval


def run_ope(
    dataset,
    round: int,
    time_at_evaluation: float,
    estimated_policy_value_list: list,
    val_bandit_data: dict,
    action_dist_val: np.ndarray,
    num_episodes_for_Prognosticator: int,
    num_time_structure_from_t_now_to_time_at_evaluation: int,
    num_true_time_structure_for_OPFV_reward: int,
    phi_scalar_func_list: Sequence[Callable],
    phi_vector_func_list: Sequence[Callable],
    num_features_for_Prognosticator: int,
    num_features_for_Prognosticator_list: Iterable[int],
    *,
    num_true_time_structure_for_OPFV_for_context: int | None = None,
    eps: float = 0.2,
    flag_calulate_robust_OPFV: bool = False,
    flag_Prognosticator_optimality: bool = True,
    true_policy_value: float | None = None,
    flag_include_DM: bool = True,
    flag_calculate_data_driven_OPFV: bool = False,
    candidate_num_time_structure_list: Sequence[int] | None = None,
) -> None:
    def phi_scalar_func_for_OPFV(unix_time: float) -> float:
        return unix_time_to_time_structure_n_tree(
            unix_time, num_true_time_structure_for_OPFV_reward
        )

    if flag_calulate_robust_OPFV:

        def phi_scalar_func_for_OPFV_for_context(unix_time: float) -> float:
            assert num_true_time_structure_for_OPFV_for_context is not None
            return unix_time_to_time_structure_n_tree(
                unix_time, num_true_time_structure_for_OPFV_for_context
            )

    assert candidate_num_time_structure_list is not None or not flag_calculate_data_driven_OPFV

    def finest_time_structure(unix_time: float) -> float:
        assert candidate_num_time_structure_list is not None
        return unix_time_to_time_structure_n_tree(unix_time, candidate_num_time_structure_list[-1])

    reg_model = RegressionModel(
        n_actions=dataset.n_actions,
        action_context=val_bandit_data["action_context"],
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=12345 + round
        ),
    )

    estimated_rewards = reg_model.fit_predict(
        context=val_bandit_data["context"],
        action=val_bandit_data["action"],
        reward=val_bandit_data["reward"],
        n_folds=2,
        random_state=12345 + round,
    )

    if flag_include_DM:
        ope_estimators = [
            IPS(estimator_name="IPS"),
            DR(estimator_name="DR"),
            DM(estimator_name="DM"),
        ]
    else:
        ope_estimators = [
            IPS(estimator_name="IPS"),
            DR(estimator_name="DR"),
        ]

    ope = OffPolicyEvaluation(
        bandit_feedback=val_bandit_data,
        ope_estimators=ope_estimators,
    )

    estimated_policy_values = ope.estimate_policy_values(
        action_dist=action_dist_val[:, :, np.newaxis],
        estimated_rewards_by_reg_model=estimated_rewards,
    )

    sorted_indices = np.argsort(val_bandit_data["time"])

    action_sorted_Prognosticator = val_bandit_data["action"][sorted_indices]
    reward_sorted_Prognosticator = val_bandit_data["reward"][sorted_indices]
    pscore_sorted_Prognosticator = val_bandit_data["pscore"][sorted_indices]
    action_dist_val_sorted_Prognosticator = action_dist_val[sorted_indices]

    nf_list = list(num_features_for_Prognosticator_list)

    if flag_Prognosticator_optimality:
        estimated_mse_list: list[float] = []
        candidate_estimated_policy_value_list: list[float] = []

        for i in range(len(phi_scalar_func_list)):
            for num_features_for_Prognosticator_ in nf_list:
                candidate_estimated_policy_value = Prognosticator(
                    num_episodes=num_episodes_for_Prognosticator,
                    phi_scalar_func=phi_scalar_func_list[i],
                    phi_vector_func=phi_vector_func_list[i],
                    reward=reward_sorted_Prognosticator,
                    action=action_sorted_Prognosticator,
                    pscore=pscore_sorted_Prognosticator,
                    action_dist=action_dist_val_sorted_Prognosticator,
                    num_episodes_after_logged_data=num_time_structure_from_t_now_to_time_at_evaluation,
                    num_features_for_Prognosticator=num_features_for_Prognosticator_,
                )
                candidate_estimated_policy_value_list.append(
                    float(candidate_estimated_policy_value)
                )
                assert true_policy_value is not None
                estimated_mse = (candidate_estimated_policy_value - true_policy_value) ** 2
                estimated_mse_list.append(float(estimated_mse))

        min_index = min(range(len(estimated_mse_list)), key=lambda j: estimated_mse_list[j])
        estimated_policy_values["Prognosticator"] = candidate_estimated_policy_value_list[min_index]
    else:
        estimated_policy_values["Prognosticator"] = Prognosticator(
            num_episodes=num_episodes_for_Prognosticator,
            phi_scalar_func=fourier_scalar,
            phi_vector_func=fourier_vec,
            reward=reward_sorted_Prognosticator,
            action=action_sorted_Prognosticator,
            pscore=pscore_sorted_Prognosticator,
            action_dist=action_dist_val_sorted_Prognosticator,
            num_episodes_after_logged_data=num_time_structure_from_t_now_to_time_at_evaluation,
            num_features_for_Prognosticator=num_features_for_Prognosticator,
        )

    n = val_bandit_data["action"].shape[0]
    time_at_eval_vec = np.full(n, time_at_evaluation)

    hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
        phi_scalar_func_for_OPFV, val_bandit_data, dataset, time_at_eval_vec, round
    )

    _, _, hat_f_x_t_a_at_eval_true = dataset.synthesize_expected_reward(
        contexts=val_bandit_data["context"], times=time_at_eval_vec
    )

    action_dist_val_at_eval = gen_eps_greedy(
        expected_reward=hat_f_x_t_a_at_eval_true,
        is_optimal=True,
        eps=eps,
    )

    estimated_policy_values["OPFV"] = OPFV(
        phi_scalar_func=phi_scalar_func_for_OPFV,
        phi_scalar_func_for_context=None,
        time_at_eval=time_at_evaluation,
        estimated_rewards_by_reg_model=hat_f_x_t_a,
        estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
        reward=val_bandit_data["reward"],
        action=val_bandit_data["action"],
        time=val_bandit_data["time"],
        pscore=val_bandit_data["pscore"],
        action_dist=action_dist_val,
        action_dist_at_eval=action_dist_val_at_eval,
        flag_robust_to_non_stationary_context=False,
        flag_use_true_P_phi_t_for_reward=False,
        P_phi_t_true_for_reward=None,
        flag_use_true_P_phi_t_for_context=False,
        P_phi_t_true_for_context=None,
        flag_use_true_P_phi_t_for_context_reward=False,
        P_phi_t_true_for_context_reward=None,
    ).mean()

    if flag_calulate_robust_OPFV:
        estimated_policy_values["robust OPFV"] = OPFV(
            phi_scalar_func=phi_scalar_func_for_OPFV,
            phi_scalar_func_for_context=phi_scalar_func_for_OPFV_for_context,
            time_at_eval=time_at_evaluation,
            estimated_rewards_by_reg_model=hat_f_x_t_a,
            estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
            reward=val_bandit_data["reward"],
            action=val_bandit_data["action"],
            time=val_bandit_data["time"],
            pscore=val_bandit_data["pscore"],
            action_dist=action_dist_val,
            action_dist_at_eval=action_dist_val_at_eval,
            flag_robust_to_non_stationary_context=True,
            flag_use_true_P_phi_t_for_reward=False,
            P_phi_t_true_for_reward=None,
            flag_use_true_P_phi_t_for_context=False,
            P_phi_t_true_for_context=None,
            flag_use_true_P_phi_t_for_context_reward=False,
            P_phi_t_true_for_context_reward=None,
        ).mean()

    if flag_calculate_data_driven_OPFV:
        assert candidate_num_time_structure_list is not None
        hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
            finest_time_structure, val_bandit_data, dataset, time_at_eval_vec, round
        )

        estimated_value_with_finest_time_structure = OPFV(
            phi_scalar_func=finest_time_structure,
            phi_scalar_func_for_context=None,
            time_at_eval=time_at_evaluation,
            estimated_rewards_by_reg_model=hat_f_x_t_a,
            estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
            reward=val_bandit_data["reward"],
            action=val_bandit_data["action"],
            time=val_bandit_data["time"],
            pscore=val_bandit_data["pscore"],
            action_dist=action_dist_val,
            action_dist_at_eval=action_dist_val_at_eval,
            flag_robust_to_non_stationary_context=False,
            flag_use_true_P_phi_t_for_reward=False,
            P_phi_t_true_for_reward=None,
            flag_use_true_P_phi_t_for_context=False,
            P_phi_t_true_for_context=None,
            flag_use_true_P_phi_t_for_context_reward=False,
            P_phi_t_true_for_context_reward=None,
        ).mean()
        estimated_mse_list_dd: list[float] = []
        candidate_estimated_value_list: list[float] = []
        for candidate_num_time_structure in candidate_num_time_structure_list:

            def _make_phi(k: int):
                def candidate_phi_scalar_func(unix_time: float) -> float:
                    return unix_time_to_time_structure_n_tree(unix_time, k)

                return candidate_phi_scalar_func

            candidate_phi_scalar_func = _make_phi(int(candidate_num_time_structure))

            hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
                candidate_phi_scalar_func,
                val_bandit_data,
                dataset,
                time_at_eval_vec,
                round,
            )
            candidate_value_round_rewards = OPFV(
                phi_scalar_func=candidate_phi_scalar_func,
                phi_scalar_func_for_context=None,
                time_at_eval=time_at_evaluation,
                estimated_rewards_by_reg_model=hat_f_x_t_a,
                estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
                reward=val_bandit_data["reward"],
                action=val_bandit_data["action"],
                time=val_bandit_data["time"],
                pscore=val_bandit_data["pscore"],
                action_dist=action_dist_val,
                action_dist_at_eval=action_dist_val_at_eval,
                flag_robust_to_non_stationary_context=False,
                flag_use_true_P_phi_t_for_reward=False,
                P_phi_t_true_for_reward=None,
                flag_use_true_P_phi_t_for_context=False,
                P_phi_t_true_for_context=None,
                flag_use_true_P_phi_t_for_context_reward=False,
                P_phi_t_true_for_context_reward=None,
            )
            candidate_estimated_value_list.append(float(candidate_value_round_rewards.mean()))

            est_squared_bias = (
                candidate_value_round_rewards.mean() - estimated_value_with_finest_time_structure
            ) ** 2
            est_var = np.var(candidate_value_round_rewards, ddof=1) / len(
                candidate_value_round_rewards
            )
            estimated_mse_list_dd.append(float(est_squared_bias + est_var))

        min_index_dd = min(
            range(len(estimated_mse_list_dd)), key=lambda j: estimated_mse_list_dd[j]
        )

        estimated_policy_values["data-driven OPFV"] = candidate_estimated_value_list[min_index_dd]

    estimated_policy_values["V_t"] = true_policy_value

    estimated_policy_value_list.append(estimated_policy_values)
