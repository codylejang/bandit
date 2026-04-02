import polars as pl
from analysis.experiment_config import AQUINO_BEH_PQT_DATA
# from experiment_config import AQUINO_BEH_PQT_DATA

import seaborn as sns
import matplotlib.pyplot as plt

DF = pl.read_parquet(AQUINO_BEH_PQT_DATA)
DF = DF[
    [
    "patientId","trialAccum","blockID","trialInBlock","outcome",
    "selectedVector",
    "select_qVals","reject_qVals","qValsLeft","qValsRight",
    "select_uVal","reject_uVal","uValLeft", "uValRight",             
    "select_nVal","reject_nVal", "nValLeft", "nValRight",
    ]
]

#################################################################################
# PREP DATA
DF = DF.with_columns(pl.col("patientId").cast(pl.Int64)).sort(["patientId","blockID"])

# compute differences
DF = DF.with_columns([
    (pl.col("selectedVector")==1).cast(pl.Int8).alias("chose_left"),    
    (pl.col("qValsLeft")-pl.col("qValsRight")).alias("dQ"),
    (pl.col("uValLeft")-pl.col("uValRight")).alias("dUnc"),
    (pl.col("nValLeft")-pl.col("nValRight")).alias("dNov"),

    (pl.col("select_qVals")-pl.col("reject_qVals")).alias("dQ_select"),

    pl.when(pl.col("select_qVals") > pl.col("reject_qVals")).then(1.0)
      .when(pl.col("select_qVals") < pl.col("reject_qVals")).then(0.0)
      .otherwise(None)
      .alias("chose_ev"),

    pl.when(pl.col("select_uVal") > pl.col("reject_uVal")).then(1.0)
      .when(pl.col("select_uVal") < pl.col("reject_uVal")).then(0.0)
      .otherwise(None)
      .alias("chose_uncertain"),

    pl.when(pl.col("select_nVal") > pl.col("reject_nVal")).then(1.0)
      .when(pl.col("select_nVal") < pl.col("reject_nVal")).then(0.0)
      .otherwise(None)
      .alias("chose_novel"),

])    

DF = DF.sort(["patientId","blockID"])
print(DF)

n_quantiles = 5;
DF = DF.group_by("patientId").map_groups(
    lambda g: g.with_columns([
        pl.col("dQ").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dQ_binned"),
        pl.col("dUnc").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dUnc_binned"),
        pl.col("dNov").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dNov_binned"),
        pl.col("dQ_select").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dQ_select_binned")
    ])
)
############################################################################
# FIGURE 1E: EFFECT OF TRIAL HORIZON ON PREFERENCE
# Proportion of trials in which patients chose the option 
# with higher EV (blue), uncertainty (black) or novelty (magenta), 
# as a function of trial number. 
# The dots and bars indicate the mean and s.e.m.
# n = 22 sessions.

df = DF[["patientId","blockID","trialInBlock",
         "chose_left","chose_ev","chose_uncertain","chose_novel"]]
df = df.sort(["patientId","blockID","trialInBlock"])

print(df)


# Long format 
df = (
    df
    .group_by(["patientId","trialInBlock"])
    .agg([
        pl.col("chose_ev").mean().alias("prob_chose_ev"),
        pl.col("chose_uncertain").mean().alias("prob_chose_uncertain"),
        pl.col("chose_novel").mean().alias("prob_chose_novel")
    ])
)
df = df.sort(["patientId","trialInBlock"])
print(df)

df_long = df.unpivot(
    on=["prob_chose_ev","prob_chose_uncertain","prob_chose_novel"],
    index =["patientId","trialInBlock"],
    variable_name="ChoiceVariable",
    value_name="p_chosen"
)
print(df_long)

df_long = (
    df_long
    .group_by(["ChoiceVariable","trialInBlock"])
    .agg([
        pl.col("p_chosen").mean().alias("mean_p"),
        (pl.col("p_chosen").std(ddof=1) / pl.len().sqrt()).alias("sem_p"),
    ])
    .sort(["ChoiceVariable","trialInBlock"])
)
print(df_long)


# PLOT
df_plot = df_long.to_pandas()
# df_plot["sem"] = df_plot["sem"].fillna(0)

label_map = {
    "prob_chose_ev": "EV",
    "prob_chose_uncertain": "Uncertainty",
    "prob_chose_novel": "Novelty",
}

fig, ax = plt.subplots(figsize=(6, 4))

for choice_var, subdf in df_plot.groupby("ChoiceVariable"):
    subdf = subdf.sort_values("trialInBlock")
    print(subdf)
    ax.errorbar(
        subdf["trialInBlock"],
        subdf["mean_p"],
        yerr=subdf["sem_p"],
        marker="o",
        capsize=4,
        label=label_map[choice_var],
    )

ax.set_xlabel("Trial number in block")
ax.set_ylabel("Proportion of higher variable chosen")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.show()

############################################################################
# FIGURE 1G:     as a function of task variables

df = DF[["patientId","blockID","dQ","dUnc","dNov","dQ_binned","dUnc_binned",
         "dNov_binned","chose_left"]]
df = df.sort(["patientId","blockID"])
print(df)

# Long format 
df_long = df.unpivot(
    on=["dQ_binned","dUnc_binned","dNov_binned"],
    index = ["patientId","blockID","chose_left"],
    variable_name="BanditVariable",
    value_name="dLRQuantile",
)

df_long = (
    df_long
    .group_by(["patientId","BanditVariable","dLRQuantile"])
    .agg((pl.col("chose_left").mean()).alias("prob_chose_left"))
).sort(["patientId","BanditVariable","dLRQuantile"])

print(df_long)

df_plot = (
    df_long
    .group_by(["BanditVariable", "dLRQuantile"])
    .agg([
        pl.col("prob_chose_left").mean(),
        (pl.col("prob_chose_left").std(ddof=1) / pl.len().sqrt()).alias("sem"),
    ])
    .sort(["BanditVariable", "dLRQuantile"])
)

print(df_plot)



# PLOT
df_plot = df_plot.to_pandas()
df_plot["sem"] = df_plot["sem"].fillna(0)

label_map = {
    "dQ_binned": "EV",
    "dUnc_binned": "Uncertainty",
    "dNov_binned": "Novelty",
}

fig, ax = plt.subplots(figsize=(6, 4))

for bandit_var, subdf in df_plot.groupby("BanditVariable"):
    subdf = subdf.sort_values("dLRQuantile")
    print(subdf)
    ax.errorbar(
        subdf["dLRQuantile"],
        subdf["prob_chose_left"],
        yerr=subdf["sem"],
        marker="o",
        capsize=4,
        label=label_map[bandit_var],
    )

ax.set_xlabel("EV difference quantile")
ax.set_ylabel("Proportion chosen")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.show()
###################################################################################
# FIGURE 1D: Uncertainty and novelty bias choices

df = DF[["patientId","blockID","dQ_select","chose_left",
         "chose_uncertain","chose_novel","dQ_select_binned"]]
print(df)

# Compute probabilities of choosing left, unvertain, and novel choice
# N = numSessions X numBins = 22 X 5 = 110
df = (
    df
    .group_by(["patientId","dQ_select_binned"])
    .agg(
        [(pl.col("chose_left").mean()).alias("prob_chose_left"),
         (pl.col("chose_uncertain").mean()).alias("prob_chose_uncertain"),
         (pl.col("chose_novel").mean()).alias("prob_chose_nov")
        ]
    )
    .sort(["patientId","dQ_select_binned"])
).drop_nulls(subset=["dQ_select_binned"])

print(df)
df.describe()

# Long format 
df_long = df.unpivot(
    on=["prob_chose_uncertain","prob_chose_left","prob_chose_nov"],
    index = ["patientId","dQ_select_binned"],
    variable_name="choice_type",
    value_name="p_chosen",
)

# Compute mean and std of probabilities (of choosing left, more uncertain or more novel ) 
# N = numQuantiles = 5 (x 3 because one for left, novelty and uncertainty)

df_plot = (
    df_long
    .group_by(["dQ_select_binned", "choice_type"])
    .agg([
        pl.col("p_chosen").mean().alias("mean_p"),
        (pl.col("p_chosen").std(ddof=1) / pl.len().sqrt()).alias("sem_p"),
    ])
    .sort(["choice_type", "dQ_select_binned"])
)
df_plot
# Plot 
df_plot = df_plot.to_pandas()
# df_plot["sem_p"] = df_plot["sem_p"].fillna(0)

label_map = {
    "prob_chose_left": "Left option",
    "prob_chose_uncertain": "Uncertain option",
    "prob_chose_nov": "New option",
}

fig, ax = plt.subplots(figsize=(6, 4))


for choice_type, subdf in df_plot.groupby("choice_type"):
    subdf = subdf.sort_values("dQ_select_binned")

    ax.errorbar(
        subdf["dQ_select_binned"],
        subdf["mean_p"],
        yerr=subdf["sem_p"],
        marker="o",
        capsize=4,
        label=label_map[choice_type],
    )

ax.set_xlabel("EV difference quantile")
ax.set_ylabel("Proportion chosen")

min_val,max_val = [.2,.8]

ax.set_ylim(min_val, max_val)
ax.legend(frameon=False)
plt.show()