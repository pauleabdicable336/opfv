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
log_path = Path("./varying_num_time_feature_data")
df_path = log_path / "df"
df_path.mkdir(exist_ok=True, parents=True)
start_time = time.time()

x = "num_time_structure_for_OPFV"
xlabel = "number of time features for OPFV"
xticklabels = conf.candidate_num_time_structure_list_for_OPFV


# Set seed
torch.manual_seed(conf.random_state)

result_df_list = []

# Test Data
# Obtain the unix time when we start the evaluation of a target policy
time_at_evaluation_start = conf.time_at_evaluation_start

# Calculate the number of days in one cycle of given time structure function \phi(t)
NUM_DAYS_IN_ONE_CYCLE = 365

# Determine the unix time when we end the evaluation of a target policy
time_at_evaluation_end_datetime = datetime.datetime.fromtimestamp(
    time_at_evaluation_start
) + datetime.timedelta(
    days=NUM_DAYS_IN_ONE_CYCLE * conf.num_cycles_in_evaluation_period
)
time_at_evaluation_end = int(
    datetime.datetime.timestamp(time_at_evaluation_end_datetime)
)


# Show hyperparameters
show_hyperparameters(
    time_at_evaluation_start=time_at_evaluation_start,
    time_at_evaluation_end=time_at_evaluation_end,
    flag_show_time_at_evaluation=True,
    time_at_evaluation_list=None,
)

result_df_list = []
for num_time_structure_for_OPFV in tqdm(
    conf.candidate_num_time_structure_list_for_OPFV
):
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

    random_ = check_random_state(conf.random_state)
    # Sample the time at evaluation from given distribution (uniform)
    time_at_evaluation_vec = random_.uniform(
        time_at_evaluation_start, time_at_evaluation_end, size=conf.num_test
    ).astype(int)

    ### test bandit data is used to approximate the ground-truth policy value
    dataset_test = dataset.obtain_batch_bandit_feedback(
        n_rounds=conf.num_test,
        evaluation_mode=True,
        time_at_evaluation_vec=time_at_evaluation_vec,
        random_state_for_sampling=conf.random_state,
    )

    for _ in tqdm(range(conf.n_seeds), desc=f"{x}={num_time_structure_for_OPFV}"):
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
            num_time_structure_for_OPFV_reward=num_time_structure_for_OPFV,
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
    result_df[f"{x}"] = num_time_structure_for_OPFV
    result_df["pi_0_value"] = pi_0_value
    result_df["rel_value"] = result_df["value"] / pi_0_value
    result_df_list.append(result_df)
result_df_data = pd.concat(result_df_list).reset_index(level=0)
result_df_data.to_csv(df_path / "result_df_data.csv")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"execution time: {elapsed_time / 60} mins")
