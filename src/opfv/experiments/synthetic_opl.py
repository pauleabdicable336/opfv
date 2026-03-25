# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from pandas import DataFrame
from sklearn.utils import check_random_state
from tqdm import tqdm

from opfv.config_log import log_synthetic_opl_settings
from opfv.domain.synthetic_bandit import SyntheticBanditWithTimeDataset
from opfv.experiments.base import BaseExperiment
from opfv.experiments.timestamps import opl_unix_times
from opfv.synthetic_fopl.opl import OPL
from opfv.synthetic_fopl.settings import SyntheticFOPLSettings

logger = logging.getLogger(__name__)

NUM_DAYS_IN_ONE_CYCLE = 365


class SyntheticOPLEperiment(BaseExperiment):
    """Paper §4 synthetic F-OPL sweeps (Hydra + :func:`~opfv.synthetic_fopl.opl.OPL` + settings)."""

    def run(self) -> None:
        cfg = self.cfg
        self.log_start()
        log_synthetic_opl_settings(cfg)
        settings = SyntheticFOPLSettings.from_experiment_cfg(cfg)
        sweep = cfg.sweep
        if sweep == "time_at_eval":
            self._run_time_at_eval(cfg, settings)
        elif sweep == "n_trains":
            self._run_n_trains(cfg, settings)
        elif sweep == "lambda":
            self._run_lambda(cfg, settings)
        elif sweep == "num_time_feature":
            self._run_num_time_feature(cfg, settings)
        else:
            raise ValueError(f"Unknown synthetic OPL sweep: {sweep}")

    def _out(self, cfg: Any) -> Path:
        p = Path(cfg.paths.output_subdir) / str(cfg.name) / "df"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _eval_window(self, s: Any, times: dict[str, int]) -> tuple[int, int]:
        """Match legacy F-OPL: end = start + full cycle days (no -1s)."""
        t_start = times["time_at_evaluation_start"]
        end_dt = datetime.datetime.fromtimestamp(t_start) + datetime.timedelta(
            days=NUM_DAYS_IN_ONE_CYCLE * int(s.num_cycles_in_evaluation_period)
        )
        return t_start, int(end_dt.timestamp())

    def _run_time_at_eval(self, cfg: Any, settings: SyntheticFOPLSettings) -> None:
        s = cfg.synthetic
        times = opl_unix_times(s)
        df_path = self._out(cfg)
        t0 = time.perf_counter()
        torch.manual_seed(int(s.random_state))

        time_at_evaluation_list: list[int] = []
        x_ticks_single: list[int] = []
        for i in range(int(s.num_time_at_evaluation)):
            t_dt = datetime.datetime.fromtimestamp(times["t_now"]) + datetime.timedelta(
                days=((i + 1) * NUM_DAYS_IN_ONE_CYCLE // int(s.num_time_structure_for_logged_data))
            )
            time_at_evaluation_list.append(int(t_dt.timestamp()))
            x_ticks_single.append((i + 1) * 365 // int(s.num_time_structure_for_logged_data))

        x = "time_at_evaluation"
        result_df_list: list[DataFrame] = []

        for i in tqdm(range(len(time_at_evaluation_list))):
            test_policy_value_list: list = []
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
                num_time_structure_for_context=int(settings.num_time_structure_for_context),
                lambda_ratio=float(s.lambda_ratio),
                alpha_ratio=float(s.alpha_ratio),
                flag_simple_reward=bool(s.flag_simple_reward),
                g_coef=int(s.g_coef),
                h_coef=int(s.h_coef),
                p_1_coef=int(s.p_1_coef),
                p_2_coef=int(s.p_2_coef),
                random_state=int(s.random_state),
            )

            t_end = time_at_evaluation_list[i]
            if i != 0:
                t_start_win = time_at_evaluation_list[i - 1] + 1
            else:
                t_start_win = int(dataset.t_now) + 1

            random_ = check_random_state(int(s.random_state) + i)
            time_at_evaluation_vec = random_.uniform(
                t_start_win, t_end, size=int(s.num_test)
            ).astype(int)

            dataset_test = dataset.obtain_batch_bandit_feedback(
                n_rounds=int(s.num_test),
                evaluation_mode=True,
                time_at_evaluation_vec=time_at_evaluation_vec,
                random_state_for_sampling=int(s.random_state) + i,
            )

            for seed in tqdm(range(int(s.n_seeds)), desc=f"{x}={x_ticks_single[i]}"):
                dataset_train = dataset.obtain_batch_bandit_feedback(
                    n_rounds=int(s.num_train),
                    evaluation_mode=False,
                    random_state_for_sampling=seed,
                )
                true_value_of_learned_policies, pi_0_value = OPL(
                    dataset,
                    dataset_test,
                    dataset_train,
                    t_start_win,
                    t_end,
                    settings=settings,
                    round=seed,
                    flag_plot_loss=bool(s.flag_plot_loss),
                    flag_plot_value=bool(s.flag_plot_value),
                    num_time_structure_for_OPFV_reward=int(
                        s.num_true_time_structure_for_OPFV_reward
                    ),
                    n_actions=int(s.n_actions),
                    dim_context=int(s.dim_context),
                    max_iter=int(s.max_iter),
                    batch_size=int(s.batch_size),
                    num_time_learn=int(s.num_time_learn),
                )
                test_policy_value_list.append(true_value_of_learned_policies)

            result_df = (
                DataFrame(test_policy_value_list)
                .stack()
                .reset_index(1)
                .rename(columns={"level_1": "method", 0: "value"})
            )
            result_df[x] = x_ticks_single[i]
            result_df["pi_0_value"] = pi_0_value
            result_df["rel_value"] = result_df["value"] / pi_0_value
            result_df_list.append(result_df)

        pd.concat(result_df_list).reset_index(level=0).to_csv(df_path / "result_df_data.csv")
        logger.info("Wrote %s", df_path / "result_df_data.csv")
        logger.info("Done in %.2f min", (time.perf_counter() - t0) / 60)

    def _run_lambda(self, cfg: Any, settings: SyntheticFOPLSettings) -> None:
        s = cfg.synthetic
        times = opl_unix_times(s)
        df_path = self._out(cfg)
        t0 = time.perf_counter()
        torch.manual_seed(int(s.random_state))
        te_start, te_end = self._eval_window(s, times)
        x = "lambda_ratio"
        result_df_list: list[DataFrame] = []

        for lambda_ratio in tqdm(s.lambda_ratio_list, desc="lambda_ratio"):
            test_policy_value_list: list = []
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
                num_time_structure_for_context=int(settings.num_time_structure_for_context),
                lambda_ratio=float(lambda_ratio),
                alpha_ratio=float(s.alpha_ratio),
                flag_simple_reward=bool(s.flag_simple_reward),
                g_coef=int(s.g_coef),
                h_coef=int(s.h_coef),
                p_1_coef=int(s.p_1_coef),
                p_2_coef=int(s.p_2_coef),
                random_state=int(s.random_state),
            )
            random_ = check_random_state(int(s.random_state))
            time_at_evaluation_vec = random_.uniform(te_start, te_end, size=int(s.num_test)).astype(
                int
            )
            dataset_test = dataset.obtain_batch_bandit_feedback(
                n_rounds=int(s.num_test),
                evaluation_mode=True,
                time_at_evaluation_vec=time_at_evaluation_vec,
                random_state_for_sampling=int(s.random_state),
            )
            for seed in tqdm(range(int(s.n_seeds)), desc=f"{x}={lambda_ratio}"):
                dataset_train = dataset.obtain_batch_bandit_feedback(
                    n_rounds=int(s.num_train),
                    evaluation_mode=False,
                    random_state_for_sampling=seed,
                )
                true_value_of_learned_policies, pi_0_value = OPL(
                    dataset,
                    dataset_test,
                    dataset_train,
                    te_start,
                    te_end,
                    settings=settings,
                    round=seed,
                    flag_plot_loss=bool(s.flag_plot_loss),
                    flag_plot_value=bool(s.flag_plot_value),
                    num_time_structure_for_OPFV_reward=int(
                        s.num_true_time_structure_for_OPFV_reward
                    ),
                    max_iter=int(s.max_iter),
                    batch_size=int(s.batch_size),
                    num_time_learn=int(s.num_time_learn),
                )
                test_policy_value_list.append(true_value_of_learned_policies)
            result_df = (
                DataFrame(test_policy_value_list)
                .stack()
                .reset_index(1)
                .rename(columns={"level_1": "method", 0: "value"})
            )
            result_df[x] = float(lambda_ratio)
            result_df["pi_0_value"] = pi_0_value
            result_df["rel_value"] = result_df["value"] / pi_0_value
            result_df_list.append(result_df)

        pd.concat(result_df_list).reset_index(level=0).to_csv(df_path / "result_df_data.csv")
        logger.info("Wrote %s", df_path / "result_df_data.csv")
        logger.info("Done in %.2f min", (time.perf_counter() - t0) / 60)

    def _run_n_trains(self, cfg: Any, settings: SyntheticFOPLSettings) -> None:
        s = cfg.synthetic
        times = opl_unix_times(s)
        df_path = self._out(cfg)
        t0 = time.perf_counter()
        torch.manual_seed(int(s.random_state))
        te_start, te_end = self._eval_window(s, times)
        x = "num_train"
        result_df_list: list[DataFrame] = []

        for num_train in tqdm(s.num_train_list, desc="num_train"):
            test_policy_value_list: list = []
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
                num_time_structure_for_context=int(settings.num_time_structure_for_context),
                lambda_ratio=float(s.lambda_ratio),
                alpha_ratio=float(s.alpha_ratio),
                flag_simple_reward=bool(s.flag_simple_reward),
                g_coef=int(s.g_coef),
                h_coef=int(s.h_coef),
                p_1_coef=int(s.p_1_coef),
                p_2_coef=int(s.p_2_coef),
                random_state=int(s.random_state),
            )
            random_ = check_random_state(int(s.random_state))
            time_at_evaluation_vec = random_.uniform(te_start, te_end, size=int(s.num_test)).astype(
                int
            )
            dataset_test = dataset.obtain_batch_bandit_feedback(
                n_rounds=int(s.num_test),
                evaluation_mode=True,
                time_at_evaluation_vec=time_at_evaluation_vec,
                random_state_for_sampling=int(s.random_state),
            )
            for seed in tqdm(range(int(s.n_seeds)), desc=f"{x}={num_train}"):
                dataset_train = dataset.obtain_batch_bandit_feedback(
                    n_rounds=int(num_train),
                    evaluation_mode=False,
                    random_state_for_sampling=seed,
                )
                true_value_of_learned_policies, pi_0_value = OPL(
                    dataset,
                    dataset_test,
                    dataset_train,
                    te_start,
                    te_end,
                    settings=settings,
                    round=seed,
                    flag_plot_loss=bool(s.flag_plot_loss),
                    flag_plot_value=bool(s.flag_plot_value),
                    num_time_structure_for_OPFV_reward=int(
                        s.num_true_time_structure_for_OPFV_reward
                    ),
                    n_actions=int(s.n_actions),
                    dim_context=int(s.dim_context),
                    max_iter=int(s.max_iter),
                    batch_size=int(s.batch_size),
                    num_time_learn=int(s.num_time_learn),
                )
                test_policy_value_list.append(true_value_of_learned_policies)
            result_df = (
                DataFrame(test_policy_value_list)
                .stack()
                .reset_index(1)
                .rename(columns={"level_1": "method", 0: "value"})
            )
            result_df[x] = int(num_train)
            result_df["pi_0_value"] = pi_0_value
            result_df["rel_value"] = result_df["value"] / pi_0_value
            result_df_list.append(result_df)

        pd.concat(result_df_list).reset_index(level=0).to_csv(df_path / "result_df_data.csv")
        logger.info("Wrote %s", df_path / "result_df_data.csv")
        logger.info("Done in %.2f min", (time.perf_counter() - t0) / 60)

    def _run_num_time_feature(self, cfg: Any, settings: SyntheticFOPLSettings) -> None:
        s = cfg.synthetic
        times = opl_unix_times(s)
        df_path = self._out(cfg)
        t0 = time.perf_counter()
        torch.manual_seed(int(s.random_state))
        te_start, te_end = self._eval_window(s, times)
        x = "num_time_structure_for_OPFV"
        result_df_list: list[DataFrame] = []

        for num_time_structure_for_OPFV in tqdm(
            s.candidate_num_time_structure_list_for_OPFV,
            desc="num_time_structure_for_OPFV",
        ):
            test_policy_value_list: list = []
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
                num_time_structure_for_context=int(settings.num_time_structure_for_context),
                lambda_ratio=float(s.lambda_ratio),
                alpha_ratio=float(s.alpha_ratio),
                flag_simple_reward=bool(s.flag_simple_reward),
                g_coef=int(s.g_coef),
                h_coef=int(s.h_coef),
                p_1_coef=int(s.p_1_coef),
                p_2_coef=int(s.p_2_coef),
                random_state=int(s.random_state),
            )
            random_ = check_random_state(int(s.random_state))
            time_at_evaluation_vec = random_.uniform(te_start, te_end, size=int(s.num_test)).astype(
                int
            )
            dataset_test = dataset.obtain_batch_bandit_feedback(
                n_rounds=int(s.num_test),
                evaluation_mode=True,
                time_at_evaluation_vec=time_at_evaluation_vec,
                random_state_for_sampling=int(s.random_state),
            )
            for seed in tqdm(
                range(int(s.n_seeds)),
                desc=f"{x}={num_time_structure_for_OPFV}",
            ):
                dataset_train = dataset.obtain_batch_bandit_feedback(
                    n_rounds=int(s.num_train),
                    evaluation_mode=False,
                    random_state_for_sampling=seed,
                )
                true_value_of_learned_policies, pi_0_value = OPL(
                    dataset,
                    dataset_test,
                    dataset_train,
                    te_start,
                    te_end,
                    settings=settings,
                    round=seed,
                    flag_plot_loss=bool(s.flag_plot_loss),
                    flag_plot_value=bool(s.flag_plot_value),
                    num_time_structure_for_OPFV_reward=int(num_time_structure_for_OPFV),
                    n_actions=int(s.n_actions),
                    dim_context=int(s.dim_context),
                    max_iter=int(s.max_iter),
                    batch_size=int(s.batch_size),
                    num_time_learn=int(s.num_time_learn),
                )
                test_policy_value_list.append(true_value_of_learned_policies)
            result_df = (
                DataFrame(test_policy_value_list)
                .stack()
                .reset_index(1)
                .rename(columns={"level_1": "method", 0: "value"})
            )
            result_df[x] = int(num_time_structure_for_OPFV)
            result_df["pi_0_value"] = pi_0_value
            result_df["rel_value"] = result_df["value"] / pi_0_value
            result_df_list.append(result_df)

        pd.concat(result_df_list).reset_index(level=0).to_csv(df_path / "result_df_data.csv")
        logger.info("Wrote %s", df_path / "result_df_data.csv")
        logger.info("Done in %.2f min", (time.perf_counter() - t0) / 60)
