# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

"""Log experiment hyperparameters (replaces print-based show_hyperparameters)."""

from __future__ import annotations

import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_synthetic_ope_settings(cfg: Any) -> None:
    """Log key synthetic F-OPE settings from a Hydra/OmegaConf node."""
    s = cfg.synthetic
    logger.info("### Seeds and sample sizes")
    logger.info(
        "n_seeds=%s n_seeds_all=%s n_seeds_for_time_eval=%s num_val=%s num_test=%s",
        s.n_seeds,
        s.n_seeds_all,
        s.n_seeds_for_time_eval_sampling,
        s.num_val,
        s.num_test,
    )
    logger.info("### Time structure")
    logger.info(
        "|C_r|=%s lambda_ratio=%s phi_pairs=%s",
        s.num_time_structure_for_logged_data,
        s.lambda_ratio,
        list(cfg.phi_pairs),
    )
    logger.info(
        "### Time horizon %s / %s / %s",
        s.t_oldest_iso,
        s.t_now_iso,
        s.t_future_iso,
    )
    logger.info("### Bandit DGP")
    logger.info(
        "|A|=%s d_x=%s n_users=%s beta=%s eps=%s",
        s.n_actions,
        s.dim_context,
        s.n_users,
        s.beta,
        s.eps,
    )
    logger.info("### OPE flags")
    logger.info(
        "include_DM=%s data_driven_OPFV=%s prognosticator_opt=%s",
        s.flag_include_DM,
        s.flag_calculate_data_driven_OPFV,
        s.flag_Prognosticator_optimality,
    )
    if getattr(cfg, "time_at_evaluation_list", None) is not None:
        tal = list(cfg.time_at_evaluation_list)
        logger.info(
            "time_at_evaluation_list=%s",
            [datetime.datetime.fromtimestamp(int(t)).isoformat() for t in tal],
        )


def log_synthetic_opl_settings(cfg: Any) -> None:
    s = cfg.synthetic
    logger.info("### Synthetic F-OPL")
    logger.info(
        "n_actions=%s dim_context=%s max_iter=%s batch_size=%s num_time_learn=%s",
        s.n_actions,
        s.dim_context,
        s.max_iter,
        s.batch_size,
        s.num_time_learn,
    )
    logger.info("phi_pairs=%s", list(cfg.phi_pairs))
    logger.info("t_oldest=%s t_now=%s t_future=%s", s.t_oldest, s.t_now, s.t_future)
