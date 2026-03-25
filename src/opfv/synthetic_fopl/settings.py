# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

"""Explicit settings for synthetic F-OPL (Hydra → frozen dataclass; no global ``conf`` module)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from opfv.domain.synthetic_bandit import obtain_num_time_structure, unix_time_to_day_of_week
from opfv.experiments.timestamps import opl_unix_times
from opfv.registry import resolve_phi_pairs_from_config


@dataclass(frozen=True, slots=True)
class SyntheticFOPLSettings:
    max_iter: int
    batch_size: int
    num_time_learn: int
    n_seeds: int
    num_train: int
    num_train_list: tuple[int, ...]
    lambda_ratio_list: tuple[float, ...]
    num_time_at_evaluation: int
    candidate_num_time_structure_list: tuple[int, ...]
    candidate_num_time_structure_list_for_OPFV: tuple[int, ...]
    reward_std: float
    num_test: int
    alpha_ratio_list: tuple[float, ...]
    alpha_ratio_and_lambda_ratio_list: tuple[Any, ...]
    num_overlaps: int
    t_oldest: int
    t_now: int
    time_at_evaluation_start: int
    t_future: int
    num_time_structure_for_logged_data: int
    num_cycles_in_evaluation_period: int
    num_true_time_structure_for_OPFV_reward: int
    num_episodes_for_Prognosticator: int
    phi_scalar_func_list: tuple[Callable[..., object], ...]
    phi_vector_func_list: tuple[Callable[..., object], ...]
    num_features_for_Prognosticator: int
    num_features_for_Prognosticator_list: tuple[int, ...]
    flag_Prognosticator_optimality: bool
    sample_non_stationary_context: bool
    time_structure_func_for_context: Callable[..., object]
    num_time_structure_for_context: int
    p_1_coef: int
    p_2_coef: int
    beta_list: tuple[float, ...]
    n_actions_list: tuple[int, ...]
    dim_context_list: tuple[int, ...]
    num_time_learn_list: tuple[int, ...]
    n_users_list: tuple[Any, ...]
    beta: float
    alpha_ratio: float
    lambda_ratio: float
    n_actions: int
    dim_context: int
    n_users: Any
    flag_simple_reward: bool
    g_coef: int
    h_coef: int
    random_state: int
    eps: float
    flag_Prognosticator_with_multiple_feature_func: bool
    flag_include_DM: bool
    flag_calculate_data_driven_OPFV: bool
    flag_plot_loss: bool
    flag_plot_value: bool
    flag_include_behavior_policy: bool
    flag_include_best_policy: bool
    flag_include_RegBased: bool
    flag_include_IPS_PG: bool
    flag_include_DR_PG: bool
    flag_include_Prognosticator: bool
    markersize: int

    @classmethod
    def from_experiment_cfg(cls, cfg: Any) -> SyntheticFOPLSettings:
        s = cfg.synthetic
        times = opl_unix_times(s)
        phi_s, phi_v = resolve_phi_pairs_from_config(cfg.phi_pairs)
        return cls(
            max_iter=int(s.max_iter),
            batch_size=int(s.batch_size),
            num_time_learn=int(s.num_time_learn),
            n_seeds=int(s.n_seeds),
            num_train=int(s.num_train),
            num_train_list=tuple(int(x) for x in s.num_train_list),
            lambda_ratio_list=tuple(float(x) for x in s.lambda_ratio_list),
            num_time_at_evaluation=int(s.num_time_at_evaluation),
            candidate_num_time_structure_list=tuple(int(x) for x in s.candidate_num_time_structure_list),
            candidate_num_time_structure_list_for_OPFV=tuple(
                int(x) for x in s.candidate_num_time_structure_list_for_OPFV
            ),
            reward_std=float(s.reward_std),
            num_test=int(s.num_test),
            alpha_ratio_list=tuple(float(x) for x in s.alpha_ratio_list),
            alpha_ratio_and_lambda_ratio_list=tuple(s.alpha_ratio_and_lambda_ratio_list),
            num_overlaps=int(s.num_overlaps),
            t_oldest=int(times["t_oldest"]),
            t_now=int(times["t_now"]),
            time_at_evaluation_start=int(times["time_at_evaluation_start"]),
            t_future=int(times["t_future"]),
            num_time_structure_for_logged_data=int(s.num_time_structure_for_logged_data),
            num_cycles_in_evaluation_period=int(s.num_cycles_in_evaluation_period),
            num_true_time_structure_for_OPFV_reward=int(s.num_true_time_structure_for_OPFV_reward),
            num_episodes_for_Prognosticator=int(s.num_episodes_for_Prognosticator),
            phi_scalar_func_list=tuple(phi_s),
            phi_vector_func_list=tuple(phi_v),
            num_features_for_Prognosticator=int(s.num_features_for_Prognosticator),
            num_features_for_Prognosticator_list=tuple(
                int(x) for x in s.num_features_for_Prognosticator_list
            ),
            flag_Prognosticator_optimality=bool(s.flag_Prognosticator_optimality),
            sample_non_stationary_context=bool(s.sample_non_stationary_context),
            time_structure_func_for_context=unix_time_to_day_of_week,
            num_time_structure_for_context=obtain_num_time_structure(unix_time_to_day_of_week),
            p_1_coef=int(s.p_1_coef),
            p_2_coef=int(s.p_2_coef),
            beta_list=tuple(float(x) for x in s.beta_list),
            n_actions_list=tuple(int(x) for x in s.n_actions_list),
            dim_context_list=tuple(int(x) for x in s.dim_context_list),
            num_time_learn_list=tuple(int(x) for x in s.num_time_learn_list),
            n_users_list=tuple(s.n_users_list),
            beta=float(s.beta),
            alpha_ratio=float(s.alpha_ratio),
            lambda_ratio=float(s.lambda_ratio),
            n_actions=int(s.n_actions),
            dim_context=int(s.dim_context),
            n_users=s.n_users,
            flag_simple_reward=bool(s.flag_simple_reward),
            g_coef=int(s.g_coef),
            h_coef=int(s.h_coef),
            random_state=int(s.random_state),
            eps=float(s.eps),
            flag_Prognosticator_with_multiple_feature_func=bool(
                s.flag_Prognosticator_with_multiple_feature_func
            ),
            flag_include_DM=bool(s.flag_include_DM),
            flag_calculate_data_driven_OPFV=bool(s.flag_calculate_data_driven_OPFV),
            flag_plot_loss=bool(s.flag_plot_loss),
            flag_plot_value=bool(s.flag_plot_value),
            flag_include_behavior_policy=bool(s.flag_include_behavior_policy),
            flag_include_best_policy=bool(s.flag_include_best_policy),
            flag_include_RegBased=bool(s.flag_include_RegBased),
            flag_include_IPS_PG=bool(s.flag_include_IPS_PG),
            flag_include_DR_PG=bool(s.flag_include_DR_PG),
            flag_include_Prognosticator=bool(s.flag_include_Prognosticator),
            markersize=12,
        )
