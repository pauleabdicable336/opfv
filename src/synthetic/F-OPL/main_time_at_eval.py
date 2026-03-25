import warnings

warnings.filterwarnings("ignore")
import datetime
import time
from logging import getLogger
from pathlib import Path

import conf
import pandas as pd
import torch
from opl import OPL
from pandas import DataFrame
from sklearn.utils import check_random_state
from synthetic_time import SyntheticBanditWithTimeDataset
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
xlabel = "target time (days later)"

time_at_evaluation_list = []
x_ticks_list_single = []

NUM_DAYS_IN_ONE_CYCLE = 365

for i in range(conf.num_time_at_evaluation):
    t_at_evaluation_datetime = datetime.datetime.fromtimestamp(
        conf.t_now
    ) + datetime.timedelta(
        days=(
            (i + 1) * NUM_DAYS_IN_ONE_CYCLE // conf.num_time_structure_for_logged_data
        )
    )
    t_at_evaluation = int(datetime.datetime.timestamp(t_at_evaluation_datetime))
    time_at_evaluation_list.append(t_at_evaluation)
    x_ticks_list_single.append((i + 1) * 365 // conf.num_time_structure_for_logged_data)

x_ticks_list = []

for i in range(len(x_ticks_list_single)):
    if i != 0:
        x_ticks_list.append(
            f"{x_ticks_list_single[i - 1] + 1}~{x_ticks_list_single[i]}"
        )
    else:
        x_ticks_list.append(f"1~{x_ticks_list_single[i]}")

# Set seed
torch.manual_seed(conf.random_state)

result_df_list = []


# Show hyperparameters
show_hyperparameters(
    time_at_evaluation_start=None,
    time_at_evaluation_end=None,
    flag_show_time_at_evaluation=False,
    time_at_evaluation_list=time_at_evaluation_list,
)

result_df_list = []
for i in tqdm(range(len(time_at_evaluation_list))):
    test_policy_value_list = []

    dataset = SyntheticBanditWithTimeDataset(
        n_actions=conf.n_actions,  # Number of Actions |A|
        dim_context=conf.dim_context,  # Dimension of the context d_x
        n_users=conf.n_users,  # number of users
        t_oldest=conf.t_oldest,  # time when we start collecting the logged data
        t_now=conf.t_now,  # time when we finish collecting the logged data
        t_future=conf.t_future,  # Future time
        beta=conf.beta,  # optimality of the behavior policy
        reward_std=conf.reward_std,  # standard deviation of reward
        num_time_structure=conf.num_time_structure_for_logged_data,  # the true number of time structure for reward
        num_time_structure_for_context=conf.num_time_structure_for_context,
        lambda_ratio=conf.lambda_ratio,  # strength of the influence of the time structure for reward
        alpha_ratio=conf.alpha_ratio,  # strength of the influence of the time structure for context
        flag_simple_reward=conf.flag_simple_reward,  # if expected reward function is simple or not
        sample_non_stationary_context=False,  # if the context is non-stationary or not
        g_coef=conf.g_coef,  # parameter for generating g(x, phi(t), a)
        h_coef=conf.h_coef,  # parameter for generating h(x, t, a)
        p_1_coef=conf.p_1_coef,  # parameter for generating the part of non-staitonary context affected by time structure for context
        p_2_coef=conf.p_2_coef,  # parameter for generating the part of non-staitonary context not affected by time structure for context
        random_state=conf.random_state,  # random state
    )

    time_at_evaluation_start = time_at_evaluation_list[i]
    time_at_evaluation_end = time_at_evaluation_list[i]

    if i != 0:
        time_at_evaluation_start = time_at_evaluation_list[i - 1] + 1
        time_at_evaluation_end = time_at_evaluation_list[i]
    else:
        time_at_evaluation_start = dataset.t_now + 1
        time_at_evaluation_end = time_at_evaluation_list[i]

    random_ = check_random_state(conf.random_state + i)

    # Sample the time at evaluation from given distribution (uniform)
    time_at_evaluation_vec = random_.uniform(
        time_at_evaluation_start, time_at_evaluation_end, size=conf.num_test
    ).astype(int)

    ### test bandit data is used to approximate the ground-truth policy value
    dataset_test = dataset.obtain_batch_bandit_feedback(
        n_rounds=conf.num_test,
        evaluation_mode=True,
        time_at_evaluation_vec=time_at_evaluation_vec,
        random_state_for_sampling=conf.random_state + i,
    )

    for _ in tqdm(range(conf.n_seeds), desc=f"{x}={x_ticks_list[i]}"):
        ## generate training data
        dataset_train = dataset.obtain_batch_bandit_feedback(
            n_rounds=conf.num_train, evaluation_mode=False, random_state_for_sampling=_
        )

        true_value_of_learned_policies, pi_0_value = OPL(
            dataset=dataset,
            dataset_test=dataset_test,
            dataset_train=dataset_train,
            time_at_evaluation_start=time_at_evaluation_start,
            time_at_evaluation_end=time_at_evaluation_end,
            round=_,
            flag_plot_loss=conf.flag_plot_loss,
            flag_plot_value=conf.flag_plot_value,
            num_time_structure_for_OPFV_reward=conf.num_true_time_structure_for_OPFV_reward,
            n_actions=conf.n_actions,
            dim_context=conf.dim_context,
            max_iter=conf.max_iter,
            batch_size=conf.batch_size,
            num_time_learn=conf.num_time_learn,
        )

        test_policy_value_list.append(true_value_of_learned_policies)

    ## summarize results
    result_df = (
        DataFrame(test_policy_value_list)
        .stack()
        .reset_index(1)
        .rename(columns={"level_1": "method", 0: "value"})
    )
    result_df[f"{x}"] = x_ticks_list_single[i]
    result_df["pi_0_value"] = pi_0_value
    result_df["rel_value"] = result_df["value"] / pi_0_value
    result_df_list.append(result_df)
result_df_data = pd.concat(result_df_list).reset_index(level=0)
result_df_data.to_csv(df_path / "result_df_data.csv")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"execution time: {elapsed_time / 60} mins")
