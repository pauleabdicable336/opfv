import time
import warnings

import pandas as pd
import torch
from opl import OPL_OPFV_tune_phi, evaluate_OPL_algorithm
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")
from logging import getLogger
from pathlib import Path

import conf
from preprocess import pre_process
from tqdm import tqdm

time_whole_execution_start = time.time()
logger = getLogger(__name__)
logger.info(f"The current working directory is {Path().cwd()}")

# log path
log_path = Path("./real_data")
df_path = log_path / "df"
df_path.mkdir(exist_ok=True, parents=True)
# If you are running locally, make sure you are in the directory of KuaiRec.
rootpath = "../../../../KuaiRec/"

# Read the CSV files
print("Loading big matrix...")
big_matrix = pd.read_csv(rootpath + "data/big_matrix.csv")
print("Loading small matrix...")
small_matrix = pd.read_csv(rootpath + "data/small_matrix.csv")

print("Loading social network...")
social_network = pd.read_csv(rootpath + "data/social_network.csv")
social_network["friend_list"] = social_network["friend_list"].map(eval)

print("Loading item features...")
item_categories = pd.read_csv(rootpath + "data/item_categories.csv")
item_categories["feat"] = item_categories["feat"].map(eval)

print("Loading user features...")
user_features = pd.read_csv(rootpath + "data/user_features.csv")

print("Loading items' daily features...")
item_daily_features = pd.read_csv(rootpath + "data/item_daily_features.csv")

print("All data loaded.")
# Set seed
torch.manual_seed(conf.random_state)
random_ = check_random_state(conf.random_state)

test_policy_value_list_DM_all_results = []
test_policy_value_list_IPS_all_results = []
test_policy_value_list_SNIPS_all_results = []
test_policy_value_list_SNDR_all_results = []
pi_learned_list_all_results = []

for _ in tqdm(range(conf.n_seeds)):
    print(
        f"\n############################################### START of ROUND {_ + 1}/{conf.n_seeds} ###############################################"
    )

    ### Preprocess ###
    time_pre_process_start = time.time()
    dataset, dataset_train, dataset_test = pre_process(
        small_matrix,
        big_matrix,
        item_categories,
        item_daily_features,
        user_features,
        social_network,
        random_state=conf.random_state + _,
        n_actions=conf.n_actions,
        dim_context=conf.dim_context,
        dim_action_context=conf.dim_action_context,
    )
    time_pre_process_end = time.time()

    elapsed_time = time_pre_process_end - time_pre_process_start
    print(f"Execution time for preprocessing = {elapsed_time / 60:.3f} mins")

    ### OPL ###
    start_time = time.time()

    pi_opfv_tuned = OPL_OPFV_tune_phi(
        dataset=dataset,
        dataset_test=dataset_test,
        dataset_train=dataset_train,
        time_test=dataset_test["time"],
        round=conf.random_state + _,
        num_time_structure_for_OPFV_reward=conf.num_time_structure_for_OPFV_reward,
        phi_scalar_func_for_OPFV=conf.phi_scalar_func_for_OPFV,
        n_actions=dataset["n_actions"],
        dim_context=dataset["dim_context"],
        max_iter=conf.max_iter,
        batch_size=conf.batch_size,
        num_time_learn=conf.num_time_learn,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time for OPL: {elapsed_time / 60:.3f} mins")

    ### Evaluate the learned policy ###
    (
        test_policy_value_list_DM,
        test_policy_value_list_IPS,
        test_policy_value_list_SNIPS,
        test_policy_value_list_SNDR,
    ) = evaluate_OPL_algorithm(
        dataset_test=dataset_test,
        # pi_reg=pi_reg,
        # pi_ips=pi_ips,
        # pi_dr=pi_dr,
        # pi_prognosticator_DM=pi_prognosticator_DM,
        # pi_prognosticator_IPS=pi_prognosticator_IPS,
        # pi_prognosticator_SNIPS=pi_prognosticator_SNIPS,
        # pi_prognosticator_SNDR=pi_prognosticator_SNDR,
        # pi_opfv=pi_opfv,
        pi_opfv_tuned=pi_opfv_tuned,
        test_policy_value_list_DM_all_results=test_policy_value_list_DM_all_results,
        test_policy_value_list_IPS_all_results=test_policy_value_list_IPS_all_results,
        test_policy_value_list_SNIPS_all_results=test_policy_value_list_SNIPS_all_results,
        test_policy_value_list_SNDR_all_results=test_policy_value_list_SNDR_all_results,
        round=_,
    )

    ### Write the result tables to the CSV files
    # result_pi_learned = pi_learned_list_all_results.copy()
    result_df_DM = test_policy_value_list_DM_all_results.copy()
    result_df_IPS = test_policy_value_list_IPS_all_results.copy()
    result_df_SNIPS = test_policy_value_list_SNIPS_all_results.copy()
    result_df_SNDR = test_policy_value_list_SNDR_all_results.copy()

    result_df_DM = pd.DataFrame(result_df_DM)
    result_df_IPS = pd.DataFrame(result_df_IPS)
    result_df_SNIPS = pd.DataFrame(result_df_SNIPS)
    result_df_SNDR = pd.DataFrame(result_df_SNDR)

    result_df_DM.to_csv(df_path / "result_df_DM_opfv_tuned.csv")
    result_df_IPS.to_csv(df_path / "result_df_IPS_opfv_tuned.csv")
    result_df_SNIPS.to_csv(df_path / "result_df_SNIPS_opfv_tuned.csv")
    result_df_SNDR.to_csv(df_path / "result_df_SNDR_opfv_tuned.csv")

    print(
        f"############################################### END of ROUND {_ + 1}/{conf.n_seeds} ###############################################\n\n\n"
    )

time_whole_execution_end = time.time()

time_whole_execution = time_whole_execution_end - time_whole_execution_start

print(f"Execution time = {time_whole_execution / 60:.3f} mins")
### Write the result tables to the CSV files
# result_pi_learned = pi_learned_list_all_results.copy()
result_df_DM = test_policy_value_list_DM_all_results.copy()
result_df_IPS = test_policy_value_list_IPS_all_results.copy()
result_df_SNIPS = test_policy_value_list_SNIPS_all_results.copy()
result_df_SNDR = test_policy_value_list_SNDR_all_results.copy()

result_df_DM = pd.DataFrame(result_df_DM)
result_df_IPS = pd.DataFrame(result_df_IPS)
result_df_SNIPS = pd.DataFrame(result_df_SNIPS)
result_df_SNDR = pd.DataFrame(result_df_SNDR)

result_df_DM.to_csv(df_path / "result_df_DM_opfv_tuned.csv")
result_df_IPS.to_csv(df_path / "result_df_IPS_opfv_tuned.csv")
result_df_SNIPS.to_csv(df_path / "result_df_SNIPS_opfv_tuned.csv")
result_df_SNDR.to_csv(df_path / "result_df_SNDR_opfv_tuned.csv")
