# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from opfv.experiments.base import BaseExperiment
from opfv.kuairec_fopl import conf as kuairec_conf
from opfv.kuairec_fopl import opl as kuairec_opl
from opfv.kuairec_fopl import preprocess as kuairec_preprocess

logger = logging.getLogger(__name__)


def _apply_hydra_kuairec_defaults(conf_mod: Any, k: Any) -> None:
    """Apply Hydra ``kuairec`` fields onto the KuaiRec defaults module."""
    conf_mod.n_seeds = int(k.n_seeds)
    conf_mod.random_state = int(k.random_state)
    conf_mod.n_actions = int(k.n_actions)
    conf_mod.dim_context = int(k.dim_context)
    conf_mod.dim_action_context = int(k.dim_action_context)
    conf_mod.max_iter = int(k.max_iter)
    conf_mod.batch_size = int(k.batch_size)
    conf_mod.num_time_learn = int(k.num_time_learn)
    conf_mod.learning_rate = float(k.learning_rate)
    conf_mod.solver = str(k.solver)


class KuairecTunePhiExperiment(BaseExperiment):
    """KuaiRec OPFV-φ tuning via :mod:`opfv.kuairec_fopl`."""

    def run(self) -> None:
        self.log_start()
        k = self.cfg.kuairec
        root = Path(str(k.root)).expanduser()
        data_dir = root / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"KuaiRec data directory not found: {data_dir}. "
                "Set kuairec.root or export KUAIREC_ROOT."
            )

        _apply_hydra_kuairec_defaults(kuairec_conf, k)

        df_path = Path(self.cfg.paths.output_subdir) / str(self.cfg.name) / "df"
        df_path.mkdir(parents=True, exist_ok=True)

        t_whole = time.perf_counter()
        logger.info("Loading KuaiRec tables from %s", data_dir)
        big_matrix = pd.read_csv(data_dir / "big_matrix.csv")
        small_matrix = pd.read_csv(data_dir / "small_matrix.csv")
        social_network = pd.read_csv(data_dir / "social_network.csv")
        social_network["friend_list"] = social_network["friend_list"].map(eval)
        item_categories = pd.read_csv(data_dir / "item_categories.csv")
        item_categories["feat"] = item_categories["feat"].map(eval)
        user_features = pd.read_csv(data_dir / "user_features.csv")
        item_daily_features = pd.read_csv(data_dir / "item_daily_features.csv")
        logger.info("All KuaiRec CSVs loaded.")

        torch.manual_seed(int(kuairec_conf.random_state))

        test_policy_value_list_DM_all_results: list = []
        test_policy_value_list_IPS_all_results: list = []
        test_policy_value_list_SNIPS_all_results: list = []
        test_policy_value_list_SNDR_all_results: list = []

        for rnd in tqdm(range(int(kuairec_conf.n_seeds)), desc="kuairec_seed"):
            logger.info("Round %s / %s", rnd + 1, kuairec_conf.n_seeds)
            t_pre = time.perf_counter()
            dataset, dataset_train, dataset_test = kuairec_preprocess.pre_process(
                small_matrix,
                big_matrix,
                item_categories,
                item_daily_features,
                user_features,
                social_network,
                random_state=int(kuairec_conf.random_state) + rnd,
                n_actions=int(kuairec_conf.n_actions),
                dim_context=int(kuairec_conf.dim_context),
                dim_action_context=int(kuairec_conf.dim_action_context),
            )
            logger.info("Preprocess done in %.3f min", (time.perf_counter() - t_pre) / 60)

            t_opl = time.perf_counter()
            pi_opfv_tuned = kuairec_opl.OPL_OPFV_tune_phi(
                dataset=dataset,
                dataset_test=dataset_test,
                dataset_train=dataset_train,
                time_test=dataset_test["time"],
                round=int(kuairec_conf.random_state) + rnd,
                num_time_structure_for_OPFV_reward=int(
                    kuairec_conf.num_time_structure_for_OPFV_reward
                ),
                phi_scalar_func_for_OPFV=kuairec_conf.phi_scalar_func_for_OPFV,
                n_actions=dataset["n_actions"],
                dim_context=dataset["dim_context"],
                max_iter=int(kuairec_conf.max_iter),
                batch_size=int(kuairec_conf.batch_size),
                num_time_learn=int(kuairec_conf.num_time_learn),
            )
            logger.info("OPL_OPFV_tune_phi done in %.3f min", (time.perf_counter() - t_opl) / 60)

            kuairec_opl.evaluate_OPL_algorithm(
                dataset_test=dataset_test,
                pi_opfv_tuned=pi_opfv_tuned,
                test_policy_value_list_DM_all_results=test_policy_value_list_DM_all_results,
                test_policy_value_list_IPS_all_results=test_policy_value_list_IPS_all_results,
                test_policy_value_list_SNIPS_all_results=test_policy_value_list_SNIPS_all_results,
                test_policy_value_list_SNDR_all_results=test_policy_value_list_SNDR_all_results,
                round=rnd,
            )

            pd.DataFrame(test_policy_value_list_DM_all_results).to_csv(
                df_path / "result_df_DM_opfv_tuned.csv"
            )
            pd.DataFrame(test_policy_value_list_IPS_all_results).to_csv(
                df_path / "result_df_IPS_opfv_tuned.csv"
            )
            pd.DataFrame(test_policy_value_list_SNIPS_all_results).to_csv(
                df_path / "result_df_SNIPS_opfv_tuned.csv"
            )
            pd.DataFrame(test_policy_value_list_SNDR_all_results).to_csv(
                df_path / "result_df_SNDR_opfv_tuned.csv"
            )

        logger.info("Total wall time %.3f min", (time.perf_counter() - t_whole) / 60)
        logger.info("Wrote results under %s", df_path)
