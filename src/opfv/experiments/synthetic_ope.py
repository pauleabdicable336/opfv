# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pandas import DataFrame
from sklearn.utils import check_random_state
from tqdm import tqdm

from opfv.config_log import log_synthetic_ope_settings
from opfv.domain.policy import gen_eps_greedy
from opfv.domain.synthetic_bandit import SECONDS_PER_DAY, SyntheticBanditWithTimeDataset
from opfv.experiments.base import BaseExperiment
from opfv.experiments.timestamps import ope_unix_times
from opfv.pipelines.ope import run_ope
from opfv.registry import resolve_phi_pairs_from_config

logger = logging.getLogger(__name__)

NUM_DAYS_IN_ONE_CYCLE = 365


def _aggregate_block(
    estimated_policy_value_list: list,
    x_key: str,
    x_value: Any,
    policy_value: float,
    *,
    exclude_v_t_from_mean: bool = False,
) -> DataFrame:
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    result_df[x_key] = x_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0.0
    result_df["variance"] = 0.0
    sub = result_df[result_df["est"] != "V_t"] if exclude_v_t_from_mean else result_df
    sample_mean = DataFrame(sub.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (policy_value - mean_estimates) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (estimates - mean_estimates) ** 2
    return result_df


class SyntheticOPEExperiment(BaseExperiment):
    """Paper §4 synthetic F-OPE sweeps driven by Hydra config."""

    def run(self) -> None:
        cfg = self.cfg
        OmegaConf.set_struct(cfg, False)
        self.log_start()
        log_synthetic_ope_settings(cfg)

        sweep = cfg.sweep
        if sweep == "target_time":
            self._run_target_time(cfg)
        elif sweep == "lambda":
            self._run_lambda(cfg)
        elif sweep == "num_time_feature":
            self._run_num_time_feature(cfg)
        elif sweep == "n_trains":
            self._run_n_trains(cfg)
        else:
            raise ValueError(f"Unknown synthetic OPE sweep: {sweep}")

    def _output_dir(self, cfg: Any) -> Path:
        base = Path(cfg.paths.output_subdir) / str(cfg.name) / "df"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _resolve_times(self, synthetic: Any) -> dict[str, int]:
        return ope_unix_times(synthetic)

    def _run_target_time(self, cfg: Any) -> None:
        s = cfg.synthetic
        times = self._resolve_times(s)
        phi_s, phi_v = resolve_phi_pairs_from_config(cfg.phi_pairs)
        df_path = self._output_dir(cfg)
        t0 = time.perf_counter()

        time_at_evaluation_list: list[int] = []
        x_ticks_single: list[int] = []
        for i in range(int(s.num_time_at_evaluation)):
            t_at_eval_dt = datetime.datetime.fromtimestamp(times["t_now"]) + datetime.timedelta(
                days=((i + 1) * 365 // s.num_time_structure_for_logged_data)
            )
            t_at_eval = int(t_at_eval_dt.timestamp())
            time_at_evaluation_list.append(t_at_eval)
            x_ticks_single.append((i + 1) * 365 // s.num_time_structure_for_logged_data)

        logger.info(
            "time_at_evaluation_list=%s",
            [datetime.datetime.fromtimestamp(t).isoformat() for t in time_at_evaluation_list],
        )

        x_key = "time_at_evaluation"
        result_df_list: list[DataFrame] = []

        for h in range(int(s.n_seeds_all)):
            for i, t_end in enumerate(time_at_evaluation_list):
                dataset = SyntheticBanditWithTimeDataset(
                    n_actions=int(s.n_actions),
                    dim_context=int(s.dim_context),
                    n_users=s.n_users,
                    t_oldest=times["t_oldest"],
                    t_now=times["t_now"],
                    t_future=times["t_future"],
                    beta=float(s.beta),
                    reward_std=float(s.reward_std),
                    num_time_structure=int(s.num_time_structure_for_logged_data),
                    lambda_ratio=float(s.lambda_ratio),
                    flag_simple_reward=bool(s.flag_simple_reward),
                    g_coef=int(s.g_coef),
                    h_coef=int(s.h_coef),
                    random_state=int(s.random_state) + h * 10,
                )

                if i != 0:
                    t_start = time_at_evaluation_list[i - 1] + 1
                    t_eval_end = t_end
                else:
                    t_start = int(dataset.t_now) + 1
                    t_eval_end = t_end

                for sidx in range(int(s.n_seeds_for_time_eval_sampling)):
                    estimated_policy_value_list: list = []
                    random_ = check_random_state(sidx + h * 10)
                    time_at_evaluation = int(random_.uniform(t_start, t_eval_end, size=1)[0])

                    test_bandit_data = dataset.obtain_batch_bandit_feedback(
                        n_rounds=int(s.num_test),
                        evaluation_mode=True,
                        time_at_evaluation=time_at_evaluation,
                        random_state_for_sampling=sidx + i * 10 + h * 100,
                    )
                    action_dist_test = gen_eps_greedy(
                        expected_reward=test_bandit_data["expected_reward"],
                        is_optimal=True,
                        eps=float(s.eps),
                    )
                    policy_value = float(
                        dataset.calc_ground_truth_policy_value(
                            expected_reward=test_bandit_data["expected_reward"],
                            action_dist=action_dist_test,
                        )
                    )

                    for seed in tqdm(
                        range(int(s.n_seeds)),
                        desc=f"h={h} target_bin={x_ticks_single[i]} t_sample={sidx}",
                    ):
                        val_bandit_data = dataset.obtain_batch_bandit_feedback(
                            n_rounds=int(s.num_val),
                            evaluation_mode=False,
                            random_state_for_sampling=seed + sidx * 10 + h * 100,
                        )
                        action_dist_val = gen_eps_greedy(
                            expected_reward=val_bandit_data["expected_reward"],
                            is_optimal=True,
                            eps=float(s.eps),
                        )
                        days_after = (time_at_evaluation - dataset.t_now) // SECONDS_PER_DAY
                        days_per_ts = NUM_DAYS_IN_ONE_CYCLE / dataset.num_time_structure
                        num_ts_eval = int(np.ceil(days_after / days_per_ts).astype(int))
                        run_ope(
                            dataset=dataset,
                            round=seed + sidx * 10 + h * 100,
                            time_at_evaluation=time_at_evaluation,
                            estimated_policy_value_list=estimated_policy_value_list,
                            val_bandit_data=val_bandit_data,
                            action_dist_val=action_dist_val,
                            num_episodes_for_Prognosticator=int(s.num_episodes_for_Prognosticator),
                            num_time_structure_from_t_now_to_time_at_evaluation=num_ts_eval,
                            num_true_time_structure_for_OPFV_reward=int(
                                s.num_true_time_structure_for_OPFV_reward
                            ),
                            phi_scalar_func_list=phi_s,
                            phi_vector_func_list=phi_v,
                            num_features_for_Prognosticator=int(s.num_features_for_Prognosticator),
                            num_features_for_Prognosticator_list=s.num_features_for_Prognosticator_list,
                            num_true_time_structure_for_OPFV_for_context=None,
                            eps=float(s.eps),
                            flag_calulate_robust_OPFV=False,
                            flag_Prognosticator_optimality=bool(s.flag_Prognosticator_optimality),
                            true_policy_value=policy_value,
                            flag_include_DM=bool(s.flag_include_DM),
                            flag_calculate_data_driven_OPFV=bool(s.flag_calculate_data_driven_OPFV),
                            candidate_num_time_structure_list=list(
                                s.candidate_num_time_structure_list
                            ),
                        )

                    result_df_list.append(
                        _aggregate_block(
                            estimated_policy_value_list,
                            x_key,
                            x_ticks_single[i],
                            policy_value,
                            exclude_v_t_from_mean=False,
                        )
                    )

        out = pd.concat(result_df_list).reset_index(level=0)
        out.to_csv(df_path / "result_df.csv")
        logger.info("Wrote %s", df_path / "result_df.csv")
        logger.info("Done in %.2f min", (time.perf_counter() - t0) / 60)

    def _eval_window(self, synthetic: Any, times: dict[str, int]) -> tuple[int, int]:
        t_start = times["time_at_evaluation"]
        end_dt = (
            datetime.datetime.fromtimestamp(t_start)
            + datetime.timedelta(
                days=NUM_DAYS_IN_ONE_CYCLE * int(synthetic.num_cycles_in_evaluation_period)
            )
            - datetime.timedelta(seconds=1)
        )
        t_end = int(end_dt.timestamp())
        return t_start, t_end

    def _run_lambda(self, cfg: Any) -> None:
        s = cfg.synthetic
        times = self._resolve_times(s)
        phi_s, phi_v = resolve_phi_pairs_from_config(cfg.phi_pairs)
        df_path = self._output_dir(cfg)
        t0 = time.perf_counter()
        te_start, te_end = self._eval_window(s, times)
        result_df_list: list[DataFrame] = []

        for h in range(int(s.n_seeds_all)):
            for lambda_ratio in s.lambda_ratio_list:
                dataset = SyntheticBanditWithTimeDataset(
                    n_actions=int(s.n_actions),
                    dim_context=int(s.dim_context),
                    n_users=s.n_users,
                    t_oldest=times["t_oldest"],
                    t_now=times["t_now"],
                    t_future=times["t_future"],
                    beta=float(s.beta),
                    reward_std=float(s.reward_std),
                    num_time_structure=int(s.num_time_structure_for_logged_data),
                    lambda_ratio=float(lambda_ratio),
                    flag_simple_reward=bool(s.flag_simple_reward),
                    g_coef=int(s.g_coef),
                    h_coef=int(s.h_coef),
                    random_state=int(s.random_state) + h * 10,
                )
                for sidx in range(int(s.n_seeds_for_time_eval_sampling)):
                    estimated_policy_value_list: list = []
                    random_ = check_random_state(sidx + h * 10)
                    time_at_evaluation = int(random_.uniform(te_start, te_end, size=1)[0])
                    test_bandit_data = dataset.obtain_batch_bandit_feedback(
                        n_rounds=int(s.num_test),
                        evaluation_mode=True,
                        time_at_evaluation=time_at_evaluation,
                        random_state_for_sampling=sidx + h * 10,
                    )
                    action_dist_test = gen_eps_greedy(
                        expected_reward=test_bandit_data["expected_reward"],
                        is_optimal=True,
                        eps=float(s.eps),
                    )
                    policy_value = float(
                        dataset.calc_ground_truth_policy_value(
                            expected_reward=test_bandit_data["expected_reward"],
                            action_dist=action_dist_test,
                        )
                    )
                    for seed in tqdm(
                        range(int(s.n_seeds)),
                        desc=f"h={h} lambda={lambda_ratio} t_sample={sidx}",
                    ):
                        val_bandit_data = dataset.obtain_batch_bandit_feedback(
                            n_rounds=int(s.num_val),
                            evaluation_mode=False,
                            random_state_for_sampling=seed
                            + int(lambda_ratio * 10) * 10
                            + sidx * 100
                            + h * 1000,
                        )
                        action_dist_val = gen_eps_greedy(
                            expected_reward=val_bandit_data["expected_reward"],
                            is_optimal=True,
                            eps=float(s.eps),
                        )
                        days_after = (time_at_evaluation - dataset.t_now) // SECONDS_PER_DAY
                        days_per_ts = NUM_DAYS_IN_ONE_CYCLE / dataset.num_time_structure
                        num_ts_eval = int(np.ceil(days_after / days_per_ts).astype(int))
                        run_ope(
                            dataset=dataset,
                            round=seed + int(lambda_ratio * 10) * 10 + sidx * 100 + h * 1000,
                            time_at_evaluation=time_at_evaluation,
                            estimated_policy_value_list=estimated_policy_value_list,
                            val_bandit_data=val_bandit_data,
                            action_dist_val=action_dist_val,
                            num_episodes_for_Prognosticator=int(s.num_episodes_for_Prognosticator),
                            num_time_structure_from_t_now_to_time_at_evaluation=num_ts_eval,
                            num_true_time_structure_for_OPFV_reward=int(
                                s.num_true_time_structure_for_OPFV_reward
                            ),
                            phi_scalar_func_list=phi_s,
                            phi_vector_func_list=phi_v,
                            num_features_for_Prognosticator=int(s.num_features_for_Prognosticator),
                            num_features_for_Prognosticator_list=s.num_features_for_Prognosticator_list,
                            eps=float(s.eps),
                            flag_Prognosticator_optimality=bool(s.flag_Prognosticator_optimality),
                            true_policy_value=policy_value,
                            flag_include_DM=bool(s.flag_include_DM),
                            flag_calculate_data_driven_OPFV=bool(s.flag_calculate_data_driven_OPFV),
                            candidate_num_time_structure_list=list(
                                s.candidate_num_time_structure_list
                            ),
                        )
                    result_df_list.append(
                        _aggregate_block(
                            estimated_policy_value_list,
                            "lambda_ratio",
                            float(lambda_ratio),
                            policy_value,
                            exclude_v_t_from_mean=False,
                        )
                    )

        pd.concat(result_df_list).reset_index(level=0).to_csv(df_path / "result_df.csv")
        logger.info("Wrote %s", df_path / "result_df.csv")
        logger.info("Done in %.2f min", (time.perf_counter() - t0) / 60)

    def _run_num_time_feature(self, cfg: Any) -> None:
        s = cfg.synthetic
        times = self._resolve_times(s)
        phi_s, phi_v = resolve_phi_pairs_from_config(cfg.phi_pairs)
        df_path = self._output_dir(cfg)
        t0 = time.perf_counter()
        te_start, te_end = self._eval_window(s, times)
        cand_opfv = list(s.candidate_num_time_structure_list_for_OPFV)
        num_epi = int(s.num_episodes_for_Prognosticator)
        logger.info(
            "num_time_structure_for_logged_data=%s |C_OPFV| sweep=%s",
            s.num_time_structure_for_logged_data,
            cand_opfv,
        )
        result_df_list: list[DataFrame] = []

        for h in range(int(s.n_seeds_all)):
            for i, num_true in enumerate(cand_opfv):
                dataset = SyntheticBanditWithTimeDataset(
                    n_actions=int(s.n_actions),
                    dim_context=int(s.dim_context),
                    n_users=s.n_users,
                    t_now=times["t_now"],
                    t_oldest=times["t_oldest"],
                    t_future=times["t_future"],
                    beta=float(s.beta),
                    reward_std=float(s.reward_std),
                    num_time_structure=int(s.num_time_structure_for_logged_data),
                    lambda_ratio=float(s.lambda_ratio),
                    flag_simple_reward=bool(s.flag_simple_reward),
                    g_coef=int(s.g_coef),
                    h_coef=int(s.h_coef),
                    random_state=int(s.random_state) + h * 10,
                )
                for sidx in range(int(s.n_seeds_for_time_eval_sampling)):
                    estimated_policy_value_list: list = []
                    random_ = check_random_state(sidx + h * 10)
                    time_at_evaluation = int(random_.uniform(te_start, te_end, size=1)[0])
                    test_bandit_data = dataset.obtain_batch_bandit_feedback(
                        n_rounds=int(s.num_test),
                        evaluation_mode=True,
                        time_at_evaluation=time_at_evaluation,
                        random_state_for_sampling=sidx + h * 10,
                    )
                    action_dist_test = gen_eps_greedy(
                        expected_reward=test_bandit_data["expected_reward"],
                        is_optimal=True,
                        eps=float(s.eps),
                    )
                    policy_value = float(
                        dataset.calc_ground_truth_policy_value(
                            expected_reward=test_bandit_data["expected_reward"],
                            action_dist=action_dist_test,
                        )
                    )
                    for seed in tqdm(
                        range(int(s.n_seeds)),
                        desc=f"h={h} |C_OPFV|={num_true} t_sample={sidx}",
                    ):
                        val_bandit_data = dataset.obtain_batch_bandit_feedback(
                            n_rounds=int(s.num_val),
                            evaluation_mode=False,
                            random_state_for_sampling=seed + h * 10,
                        )
                        action_dist_val = gen_eps_greedy(
                            expected_reward=val_bandit_data["expected_reward"],
                            is_optimal=True,
                            eps=float(s.eps),
                        )
                        days_after = (time_at_evaluation - dataset.t_now) // SECONDS_PER_DAY
                        days_per_ts = NUM_DAYS_IN_ONE_CYCLE / dataset.num_time_structure
                        num_ts_eval = int(np.ceil(days_after / days_per_ts).astype(int))
                        run_ope(
                            dataset=dataset,
                            round=seed + h * 10,
                            time_at_evaluation=time_at_evaluation,
                            estimated_policy_value_list=estimated_policy_value_list,
                            val_bandit_data=val_bandit_data,
                            action_dist_val=action_dist_val,
                            num_true_time_structure_for_OPFV_reward=int(num_true),
                            num_episodes_for_Prognosticator=num_epi,
                            num_time_structure_from_t_now_to_time_at_evaluation=num_ts_eval,
                            phi_scalar_func_list=phi_s,
                            phi_vector_func_list=phi_v,
                            num_features_for_Prognosticator=int(s.num_features_for_Prognosticator),
                            num_features_for_Prognosticator_list=s.num_features_for_Prognosticator_list,
                            eps=float(s.eps),
                            flag_Prognosticator_optimality=bool(s.flag_Prognosticator_optimality),
                            true_policy_value=policy_value,
                            flag_include_DM=bool(s.flag_include_DM),
                            flag_calculate_data_driven_OPFV=bool(s.flag_calculate_data_driven_OPFV),
                            candidate_num_time_structure_list=cand_opfv,
                        )
                    result_df_list.append(
                        _aggregate_block(
                            estimated_policy_value_list,
                            "num_time_structure_for_OPFV",
                            int(num_true),
                            policy_value,
                            exclude_v_t_from_mean=False,
                        )
                    )

        pd.concat(result_df_list).reset_index(level=0).to_csv(df_path / "result_df.csv")
        logger.info("Wrote %s", df_path / "result_df.csv")
        logger.info("Done in %.2f min", (time.perf_counter() - t0) / 60)

    def _run_n_trains(self, cfg: Any) -> None:
        s = cfg.synthetic
        times = self._resolve_times(s)
        phi_s, phi_v = resolve_phi_pairs_from_config(cfg.phi_pairs)
        df_path = self._output_dir(cfg)
        t0 = time.perf_counter()
        te_start, te_end = self._eval_window(s, times)
        result_df_list: list[DataFrame] = []

        for h in range(int(s.n_seeds_all)):
            for n_rounds in s.n_rounds_list:
                dataset = SyntheticBanditWithTimeDataset(
                    n_actions=int(s.n_actions),
                    dim_context=int(s.dim_context),
                    n_users=s.n_users,
                    t_oldest=times["t_oldest"],
                    t_now=times["t_now"],
                    t_future=times["t_future"],
                    beta=float(s.beta),
                    reward_std=float(s.reward_std),
                    num_time_structure=int(s.num_time_structure_for_logged_data),
                    lambda_ratio=float(s.lambda_ratio),
                    flag_simple_reward=bool(s.flag_simple_reward),
                    g_coef=int(s.g_coef),
                    h_coef=int(s.h_coef),
                    random_state=int(s.random_state) + h * 10,
                )
                for sidx in range(int(s.n_seeds_for_time_eval_sampling)):
                    estimated_policy_value_list: list = []
                    random_ = check_random_state(sidx + h * 10)
                    time_at_evaluation = int(random_.uniform(te_start, te_end, size=1)[0])
                    test_bandit_data = dataset.obtain_batch_bandit_feedback(
                        n_rounds=int(s.num_test),
                        evaluation_mode=True,
                        time_at_evaluation=time_at_evaluation,
                        random_state_for_sampling=sidx + h * 10,
                    )
                    action_dist_test = gen_eps_greedy(
                        expected_reward=test_bandit_data["expected_reward"],
                        is_optimal=True,
                        eps=float(s.eps),
                    )
                    policy_value = float(
                        dataset.calc_ground_truth_policy_value(
                            expected_reward=test_bandit_data["expected_reward"],
                            action_dist=action_dist_test,
                        )
                    )
                    for seed in tqdm(
                        range(int(s.n_seeds)),
                        desc=f"h={h} n_rounds={n_rounds} t_sample={sidx}",
                    ):
                        val_bandit_data = dataset.obtain_batch_bandit_feedback(
                            n_rounds=int(n_rounds),
                            evaluation_mode=False,
                            random_state_for_sampling=seed + sidx * 10 + int(n_rounds) + h * 100,
                        )
                        action_dist_val = gen_eps_greedy(
                            expected_reward=val_bandit_data["expected_reward"],
                            is_optimal=True,
                            eps=float(s.eps),
                        )
                        days_after = (time_at_evaluation - dataset.t_now) // SECONDS_PER_DAY
                        days_per_ts = NUM_DAYS_IN_ONE_CYCLE / dataset.num_time_structure
                        num_ts_eval = int(np.ceil(days_after / days_per_ts).astype(int))
                        run_ope(
                            dataset=dataset,
                            round=seed + sidx * 10 + h * 100,
                            time_at_evaluation=time_at_evaluation,
                            estimated_policy_value_list=estimated_policy_value_list,
                            val_bandit_data=val_bandit_data,
                            action_dist_val=action_dist_val,
                            num_true_time_structure_for_OPFV_reward=int(
                                s.num_true_time_structure_for_OPFV_reward
                            ),
                            num_episodes_for_Prognosticator=int(s.num_episodes_for_Prognosticator),
                            num_time_structure_from_t_now_to_time_at_evaluation=num_ts_eval,
                            phi_scalar_func_list=phi_s,
                            phi_vector_func_list=phi_v,
                            num_features_for_Prognosticator=int(s.num_features_for_Prognosticator),
                            num_features_for_Prognosticator_list=s.num_features_for_Prognosticator_list,
                            eps=float(s.eps),
                            flag_Prognosticator_optimality=bool(s.flag_Prognosticator_optimality),
                            true_policy_value=policy_value,
                            flag_include_DM=bool(s.flag_include_DM),
                            flag_calculate_data_driven_OPFV=bool(s.flag_calculate_data_driven_OPFV),
                            candidate_num_time_structure_list=list(
                                s.candidate_num_time_structure_list
                            ),
                        )
                    result_df_list.append(
                        _aggregate_block(
                            estimated_policy_value_list,
                            "n_rounds",
                            int(n_rounds),
                            policy_value,
                            exclude_v_t_from_mean=True,
                        )
                    )

        pd.concat(result_df_list).reset_index(level=0).to_csv(df_path / "result_df.csv")
        logger.info("Wrote %s", df_path / "result_df.csv")
        logger.info("Done in %.2f min", (time.perf_counter() - t0) / 60)
