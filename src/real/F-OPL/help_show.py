import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from logging import getLogger
from pathlib import Path

from utils import add_median_rank, create_df_mean_std_rank

logger = getLogger(__name__)
logger.info(f"The current working directory is {Path().cwd()}")

# log path
log_path = Path("./real_data")
df_path = log_path / "df"
df_path.mkdir(exist_ok=True, parents=True)
### Create DataFrame from the stored csv file
result_df_DM = pd.read_csv(df_path / "result_df_DM.csv", index_col=0)
result_df_IPS = pd.read_csv(df_path / "result_df_IPS.csv", index_col=0)
result_df_SNIPS = pd.read_csv(df_path / "result_df_SNIPS.csv", index_col=0)
result_df_SNDR = pd.read_csv(df_path / "result_df_SNDR.csv", index_col=0)
result_df_DM_opfv_tuned = pd.read_csv(
    df_path / "result_df_DM_opfv_tuned.csv", index_col=0
)
result_df_IPS_opfv_tuned = pd.read_csv(
    df_path / "result_df_IPS_opfv_tuned.csv", index_col=0
)
result_df_SNIPS_opfv_tuned = pd.read_csv(
    df_path / "result_df_SNIPS_opfv_tuned.csv", index_col=0
)
result_df_SNDR_opfv_tuned = pd.read_csv(
    df_path / "result_df_SNDR_opfv_tuned.csv", index_col=0
)
# Merge tables
result_df_DM = pd.merge(left=result_df_DM, right=result_df_DM_opfv_tuned)
result_df_IPS = pd.merge(left=result_df_IPS, right=result_df_IPS_opfv_tuned)
result_df_SNIPS = pd.merge(left=result_df_SNIPS, right=result_df_SNIPS_opfv_tuned)
result_df_SNDR = pd.merge(left=result_df_SNDR, right=result_df_SNDR_opfv_tuned)

### Mean of three results
result_df_3_mean = (result_df_DM + result_df_SNIPS + result_df_SNDR) / 3


### Create two DataFrames to calcualte mean, standard deviation, and mean rank of the policy value for each OPL algorithm
(
    result_df_DM_stats,
    result_df_IPS_stats,
    result_df_SNIPS_stats,
    result_df_SNDR_stats,
    result_df_3_mean_stats,
    rank_df_DM_mean,
    rank_df_IPS_mean,
    rank_df_SNIPS_mean,
    rank_df_SNDR_mean,
    rank_df_3_mean_mean,
) = create_df_mean_std_rank(
    result_df_DM, result_df_IPS, result_df_SNIPS, result_df_SNDR, result_df_3_mean
)


### Change the name of the index
new_index_list = [
    "Behavior Policy",
    "RegBased",
    "IPS-PG",
    "DR-PG",
    "Prognosticator",
    r"OPFV-PG (w/o tuned $\phi$)",
    r"OPFV-PG (w/ tuned $\phi$)",
]
overall_result_df_list = [
    result_df_DM_stats,
    result_df_IPS_stats,
    result_df_SNIPS_stats,
    result_df_SNDR_stats,
    result_df_3_mean_stats,
    rank_df_DM_mean,
    rank_df_IPS_mean,
    rank_df_SNIPS_mean,
    rank_df_SNDR_mean,
    rank_df_3_mean_mean,
]

for df in overall_result_df_list:
    df.set_index(pd.Index(new_index_list), inplace=True, drop=True)


### Merge two DataFrames on mean value and mean rank
df_stats_rank_DM = pd.merge(
    left=result_df_DM_stats, right=rank_df_DM_mean, left_index=True, right_index=True
)
df_stats_rank_IPS = pd.merge(
    left=result_df_IPS_stats, right=rank_df_IPS_mean, left_index=True, right_index=True
)
df_stats_rank_SNIPS = pd.merge(
    left=result_df_SNIPS_stats,
    right=rank_df_SNIPS_mean,
    left_index=True,
    right_index=True,
)
df_stats_rank_SNDR = pd.merge(
    left=result_df_SNDR_stats,
    right=rank_df_SNDR_mean,
    left_index=True,
    right_index=True,
)
df_stats_rank_3_mean = pd.merge(
    left=result_df_3_mean_stats,
    right=rank_df_3_mean_mean,
    left_index=True,
    right_index=True,
)


result_df_list = [result_df_DM, result_df_SNIPS, result_df_SNDR, result_df_3_mean]
df_stats_rank_list = [
    df_stats_rank_DM,
    df_stats_rank_SNIPS,
    df_stats_rank_SNDR,
    df_stats_rank_3_mean,
]

add_median_rank(result_df_list, df_stats_rank_list)

col_list = ["mean", "std", "Median", "Mean Rank", "Median of Rank"]
df_stats_rank_DM = df_stats_rank_DM[col_list]
df_stats_rank_SNIPS = df_stats_rank_SNIPS[col_list]
df_stats_rank_SNDR = df_stats_rank_SNDR[col_list]
df_stats_rank_3_mean = df_stats_rank_3_mean[col_list]


overall_result_df_list = [
    df_stats_rank_DM,
    df_stats_rank_SNIPS,
    df_stats_rank_SNDR,
    df_stats_rank_3_mean,
]
overall_result_df_list_name = [
    "df_stats_rank_DM",
    "df_stats_rank_SNIPS",
    "df_stats_rank_SNDR",
    "df_stats_rank_3_mean",
]
output_name_list = [
    "DM Evaluation",
    "SNIPS Evaluation",
    "SNDR Evaluation",
    "Mean DM, SNIPS, SNDR Evaluation",
]
with open("README.md", "a") as f:
    for i, df_result in enumerate(overall_result_df_list):
        f.write("\n\n")
        f.write(f"{output_name_list[i]}")
        f.write('\n\n<div align="center">\n\n')

        # Specify the number of digits for formatting
        desired_digits = 4  # Adjust as needed
        f.write(df_result.to_markdown(floatfmt=f".{desired_digits}f"))
        f.write("\n\n</div>\n")
df_stats_rank_3_mean.round(2)
# Convert DataFrame to LaTeX table

# Create the Dataframes for converting them to the latex form
precision = 4

# Mean Value
result_df_DM_stats_latex = pd.DataFrame()
result_df_IPS_stats_latex = pd.DataFrame()
result_df_SNIPS_stats_latex = pd.DataFrame()
result_df_SNDR_stats_latex = pd.DataFrame()
result_df_DM_stats_latex["Mean Value"] = (
    result_df_DM_stats["mean"].round(precision).astype(str)
    + r"$\pm$"
    + result_df_DM_stats["std"].round(precision).astype(str)
)
result_df_SNIPS_stats_latex["Mean Value"] = (
    result_df_SNIPS_stats["mean"].round(precision).astype(str)
    + r"$\pm$"
    + result_df_SNIPS_stats["std"].round(precision).astype(str)
)
result_df_SNDR_stats_latex["Mean Value"] = (
    result_df_SNDR_stats["mean"].round(precision).astype(str)
    + r"$\pm$"
    + result_df_SNDR_stats["std"].round(precision).astype(str)
)

# Mean Rank
rank_df_DM_mean_latex = pd.DataFrame()
rank_df_IPS_mean_latex = pd.DataFrame()
rank_df_SNIPS_mean_latex = pd.DataFrame()
rank_df_SNDR_mean_latex = pd.DataFrame()
rank_df_DM_mean_latex["Mean Rank"] = rank_df_DM_mean["Mean Rank"].round(precision)
rank_df_SNIPS_mean_latex["Mean Rank"] = rank_df_SNIPS_mean["Mean Rank"].round(precision)
rank_df_SNDR_mean_latex["Mean Rank"] = rank_df_SNDR_mean["Mean Rank"].round(precision)

# Mean Value and Mean Rank
df_stats_rank_DM_latex = pd.merge(
    left=result_df_DM_stats_latex,
    right=rank_df_DM_mean,
    left_index=True,
    right_index=True,
)
df_stats_rank_SNIPS_latex = pd.merge(
    left=result_df_SNIPS_stats_latex,
    right=rank_df_SNIPS_mean,
    left_index=True,
    right_index=True,
)
df_stats_rank_SNDR_latex = pd.merge(
    left=result_df_SNDR_stats_latex,
    right=rank_df_SNDR_mean,
    left_index=True,
    right_index=True,
)


precision = 2

# Mean/Median Values and Mean/Median ranks
DM_latex = pd.DataFrame()
SNIPS_latex = pd.DataFrame()
SNDR_latex = pd.DataFrame()
three_latex = pd.DataFrame()

DM_latex["Mean/Median Values"] = (
    df_stats_rank_DM["mean"].round(precision).astype(str)
    + "/"
    + df_stats_rank_DM["Median"].round(precision).astype(str)
)
DM_latex["Mean/Median Ranks"] = (
    df_stats_rank_DM["Mean Rank"].round(precision).astype(str)
    + "/"
    + df_stats_rank_DM["Median of Rank"].round(precision).astype(str)
)
SNIPS_latex["Mean/Median Values"] = (
    df_stats_rank_SNIPS["mean"].round(precision).astype(str)
    + "/"
    + df_stats_rank_SNIPS["Median"].round(precision).astype(str)
)
SNIPS_latex["Mean/Median Ranks"] = (
    df_stats_rank_SNIPS["Mean Rank"].round(precision).astype(str)
    + "/"
    + df_stats_rank_SNIPS["Median of Rank"].round(precision).astype(str)
)
SNDR_latex["Mean/Median Values"] = (
    df_stats_rank_SNDR["mean"].round(precision).astype(str)
    + "/"
    + df_stats_rank_SNDR["Median"].round(precision).astype(str)
)
SNDR_latex["Mean/Median Ranks"] = (
    df_stats_rank_SNDR["Mean Rank"].round(precision).astype(str)
    + "/"
    + df_stats_rank_SNDR["Median of Rank"].round(precision).astype(str)
)
three_latex["Mean/Median Values"] = (
    df_stats_rank_3_mean["mean"].round(precision).astype(str)
    + "/"
    + df_stats_rank_3_mean["Median"].round(precision).astype(str)
)
three_latex["Mean/Median Ranks"] = (
    df_stats_rank_3_mean["Mean Rank"].round(precision).astype(str)
    + "/"
    + df_stats_rank_3_mean["Median of Rank"].round(precision).astype(str)
)

DM_latex = DM_latex.drop("Behavior Policy")
SNIPS_latex = SNIPS_latex.drop("Behavior Policy")
SNDR_latex = SNDR_latex.drop("Behavior Policy")
three_latex = three_latex.drop("Behavior Policy")
big_latex = pd.concat(
    [DM_latex, SNIPS_latex, SNDR_latex],
    axis=1,
    keys=["DM Evaluation", "SNIPS Evaluation", "SNDR Evaluation"],
).T


three_latex = three_latex.T
latex_table = three_latex.to_latex(index=True, float_format="%.2f")

# # Display the LaTeX table
print(latex_table)
big_latex = pd.concat(
    [DM_latex, SNIPS_latex, SNDR_latex],
    axis=1,
    keys=["DM Evaluation", "SNIPS Evaluation", "SNDR Evaluation"],
).T
big_latex
precision_of_mean_std = 2

df_stats_rank_list = [df_stats_rank_DM, df_stats_rank_SNIPS, df_stats_rank_SNDR]
df_mean_std_list = [None, None, None]

for i in range(len(df_stats_rank_list)):
    df_mean_std_list[i] = df_stats_rank_list[i][["mean", "std"]].copy()
    df_mean_std_list[i].drop("Behavior Policy", inplace=True)
    df_mean_std_list[i] = df_mean_std_list[i].round(precision_of_mean_std)
    df_mean_std_list[i]["mean"] = df_mean_std_list[i]["mean"].astype(str)
    df_mean_std_list[i]["std"] = df_mean_std_list[i]["std"].astype(str)
    df_mean_std_list[i][r"Mean Value ($\pm$ SD)"] = (
        df_mean_std_list[i]["mean"] + "(" + r"$\pm$ " + df_mean_std_list[i]["std"] + ")"
    )
    df_mean_std_list[i].drop(["mean", "std"], axis=1, inplace=True)

big_latex_mean_std = pd.concat(df_mean_std_list, axis=1, keys=["DM", "SNIPS", "SNDR"]).T
big_latex_mean_std
latex_table = big_latex_mean_std.to_latex(index=True, float_format="%.2f")
print(latex_table)
