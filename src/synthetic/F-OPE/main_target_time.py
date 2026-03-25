import datetime
import time
from logging import getLogger
from pathlib import Path

import conf
import numpy as np
import pandas as pd
from ope import run_ope
from pandas import DataFrame
from policy import gen_eps_greedy
from sklearn.utils import check_random_state
from synthetic_time import SECONDS_PER_DAY, SyntheticBanditWithTimeDataset
from tqdm import tqdm
from utils import show_hyperparameters

logger = getLogger(__name__)
logger.info(f"The current working directory is {Path().cwd()}")

# log path
log_path = Path("./varying_target_time_data")
df_path = log_path / "df"
df_path.mkdir(exist_ok=True, parents=True)
start_time = time.time()

x = "time_at_evaluation"

time_at_evaluation_list = []
x_ticks_list_single = []

xlabel = "target time (days later)"
for i in range(conf.num_time_at_evaluation):
    t_at_eval_datetime = datetime.datetime.fromtimestamp(
        conf.t_now
    ) + datetime.timedelta(
        days=((i + 1) * 365 // conf.num_time_structure_for_logged_data)
    )
    t_at_eval = int(datetime.datetime.timestamp(t_at_eval_datetime))
    time_at_evaluation_list.append(t_at_eval)
    x_ticks_list_single.append((i + 1) * 365 // conf.num_time_structure_for_logged_data)


x_ticks_list = []

for i in range(len(x_ticks_list_single)):
    if i != 0:
        x_ticks_list.append(
            f"{x_ticks_list_single[i - 1] + 1}~{x_ticks_list_single[i]}"
        )
    else:
        x_ticks_list.append(f"1~{x_ticks_list_single[i]}")


# Show hyperparameters
show_hyperparameters(
    time_at_evaluation_start=None,
    time_at_evaluation_end=None,
    flag_show_time_at_evaluation=False,
    time_at_evaluation_list=time_at_evaluation_list,
)


V_t_list = []

result_df_list = []

for h in range(conf.n_seeds_all):
    for i in range(len(time_at_evaluation_list)):
        dataset = SyntheticBanditWithTimeDataset(
            n_actions=conf.n_actions,
            dim_context=conf.dim_context,
            n_users=conf.n_users,
            t_oldest=conf.t_oldest,
            t_now=conf.t_now,
            t_future=conf.t_future,
            beta=conf.beta,
            reward_std=conf.reward_std,
            num_time_structure=conf.num_time_structure_for_logged_data,
            lambda_ratio=conf.lambda_ratio,
            flag_simple_reward=conf.flag_simple_reward,
            g_coef=conf.g_coef,
            h_coef=conf.h_coef,
            random_state=conf.random_state + h * 10,
        )

        if i != 0:
            time_at_evaluation_start = time_at_evaluation_list[i - 1] + 1
            time_at_evaluation_end = time_at_evaluation_list[i]
        else:
            time_at_evaluation_start = dataset.t_now + 1
            time_at_evaluation_end = time_at_evaluation_list[i]

        for s in range(conf.n_seeds_for_time_eval_sampling):
            estimated_policy_value_list = []

            # Obtain random state
            random_ = check_random_state(s + h * 10)
            # Sample the time at evaluation from given distribution (uniform)
            time_at_evaluation = random_.uniform(
                time_at_evaluation_start, time_at_evaluation_end, size=1
            ).astype(int)

            ### test bandit data is used to approximate the ground-truth policy value
            test_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=conf.num_test,
                evaluation_mode=True,
                time_at_evaluation=time_at_evaluation,
                random_state_for_sampling=s + i * 10 + h * 100,
            )

            # Generate an evaluation policy via the epsilon-greedy rule
            action_dist_test = gen_eps_greedy(
                expected_reward=test_bandit_data["expected_reward"],
                is_optimal=True,
                eps=conf.eps,
            )

            # actulal policy value
            policy_value = dataset.calc_ground_truth_policy_value(
                expected_reward=test_bandit_data["expected_reward"],
                action_dist=action_dist_test,
            )

            V_t_list.append(policy_value)

            for _ in tqdm(
                range(conf.n_seeds),
                desc=f"h = {h}, {xlabel} = {x_ticks_list[i]}, n_seeds_for_time_eval_sampling = {s}",
            ):
                ## generate validation data
                val_bandit_data = dataset.obtain_batch_bandit_feedback(
                    n_rounds=conf.num_val,
                    evaluation_mode=False,
                    random_state_for_sampling=_ + s * 10 + h * 100,
                )
                ## make decisions on validation data
                action_dist_val = gen_eps_greedy(
                    expected_reward=val_bandit_data["expected_reward"],
                    is_optimal=True,
                    eps=conf.eps,
                )

                # Calculate the number of days in one cycle of given time structure function \phi(t)
                NUM_DAYS_IN_ONE_CYCLE = 365

                days_after_logged_data = (
                    time_at_evaluation - dataset.t_now
                ) // SECONDS_PER_DAY

                days_per_time_structure = (
                    NUM_DAYS_IN_ONE_CYCLE / dataset.num_time_structure
                )

                num_time_structure_from_t_now_to_time_at_evaluation = np.ceil(
                    days_after_logged_data / days_per_time_structure
                ).astype(int)

                run_ope(
                    dataset=dataset,
                    round=_ + s * 10 + h * 100,
                    time_at_evaluation=time_at_evaluation,
                    estimated_policy_value_list=estimated_policy_value_list,
                    val_bandit_data=val_bandit_data,
                    action_dist_val=action_dist_val,
                    num_true_time_structure_for_OPFV_reward=conf.num_true_time_structure_for_OPFV_reward,
                    num_true_time_structure_for_OPFV_for_context=None,
                    num_episodes_for_Prognosticator=conf.num_episodes_for_Prognosticator,
                    num_time_structure_from_t_now_to_time_at_evaluation=num_time_structure_from_t_now_to_time_at_evaluation,
                    eps=conf.eps,
                    true_policy_value=policy_value,
                    flag_Prognosticator_optimality=conf.flag_Prognosticator_optimality,
                    num_features_for_Prognosticator_list=conf.num_features_for_Prognosticator_list,
                    flag_include_DM=conf.flag_include_DM,
                    flag_calculate_data_driven_OPFV=conf.flag_calculate_data_driven_OPFV,
                    candidate_num_time_structure_list=conf.candidate_num_time_structure_list,
                )

            ## summarize results
            result_df = (
                DataFrame(DataFrame(estimated_policy_value_list).stack())
                .reset_index(1)
                .rename(columns={"level_1": "est", 0: "value"})
            )
            result_df[x] = x_ticks_list_single[i]
            result_df["se"] = (result_df.value - policy_value) ** 2
            result_df["bias"] = 0
            result_df["variance"] = 0
            sample_mean = DataFrame(
                result_df.groupby(["est"]).mean().value
            ).reset_index()
            for est_ in sample_mean["est"]:
                estimates = result_df.loc[result_df["est"] == est_, "value"].values
                mean_estimates = sample_mean.loc[
                    sample_mean["est"] == est_, "value"
                ].values
                mean_estimates = np.ones_like(estimates) * mean_estimates
                result_df.loc[result_df["est"] == est_, "bias"] = (
                    policy_value - mean_estimates
                ) ** 2
                result_df.loc[result_df["est"] == est_, "variance"] = (
                    estimates - mean_estimates
                ) ** 2
            result_df_list.append(result_df)


# aggregate all results
result_df = pd.concat(result_df_list).reset_index(level=0)
result_df.to_csv(df_path / "result_df.csv")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"execution time: {elapsed_time / 60} mins")
