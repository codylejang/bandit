

#########################################################################
# Experiment Config
from experiment_config import BANDIT_FEATURES, CHOICE_VARS, LAYERS, TRIALS_PER_BLOCK, LAYER_SIZES
#########################################################################
# INIT ----
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from dfutils import load_dataframe, pull_neuron_column, get_layer_columns,parse_neuron_id
from helpers import norm_list
import pandas as pd


df = load_dataframe(episode="max", verbose=True)
df = df.select(norm_list(["block", "trial", *BANDIT_FEATURES, *CHOICE_VARS, "reward"]))
df = df.with_columns([
    (1 - pl.col("choice_side")).alias("chose_left") 
])

df = df.with_columns([
    (1 - pl.col("choice_side")).alias("chose_left") 
])

df = df.with_columns([
    (pl.col("dQ").qcut(4).rank(method="dense") - 1)
        .cast(pl.UInt8)
        .alias("dQ_binned"),
    (pl.col("dUnc").qcut(4).rank(method="dense") - 1)
        .cast(pl.UInt8)
        .alias("dUnc_binned"),
    (pl.col("dNov").qcut(4).rank(method="dense") - 1)
        .cast(pl.UInt8)
        .alias("dNov_binned"),        
])



######################################################
# Fig 1D: Uncertainty and novelty bias choices
df = df.with_columns([
    (
        (
            (pl.col("chose_left") == 1) & (pl.col("dUnc") > 0)  # chose left and left is more uncertain
        ) | (
            (pl.col("chose_left") == 0) & (pl.col("dUnc") < 0)  # chose right and right is more uncertain
        )
    ).cast(pl.Int8).alias("chose_uncertain")  # 1=yes, 0=no
])

df = df.with_columns([
    (
        (
            (pl.col("chose_left") == 1) & (pl.col("dNov") > 0)  # chose left and left is more novel
        ) | (
            (pl.col("chose_left") == 0) & (pl.col("dNov") < 0)  # chose right and right is more novel
        )
    ).cast(pl.Int8).alias("chose_novel")  # 1=yes, 0=no
])



# group by: percent chosen for each behavior per dQ bin
agg = (
    df
    .group_by("dQ_binned")
    .agg([
        pl.col("chose_left").mean().alias("p_chose_left"),
        pl.col("chose_uncertain").mean().alias("p_chose_uncertain"),
        pl.col("chose_novel").mean().alias("p_chose_novel"),
    ])
    .sort("dQ_binned")
)

# Long format 
plot_df = agg.melt(
    id_vars="dQ_binned",
    variable_name="choice_type",
    value_name="p_chosen",
)

# Plot
plot_pd = plot_df.to_pandas()
sns.lineplot(data=plot_pd, x="dQ_binned", y="p_chosen", hue="choice_type", marker="o")
plt.xlabel("dQ quantile bin (0 = lowest)")
plt.ylabel("Percentage chosen")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()






#####################################################
## Figure 1G: Decisions as a function of task variables. 

df = df.with_columns([
    (pl.col("dQ").qcut(4).rank(method="dense") - 1)
        .cast(pl.UInt8)
        .alias("dQ_binned"),
    (pl.col("dUnc").qcut(4).rank(method="dense") - 1)
        .cast(pl.UInt8)
        .alias("dUnc_binned"),
    (pl.col("dNov").qcut(4).rank(method="dense") - 1)
        .cast(pl.UInt8)
        .alias("dNov_binned"),        
])

print(df)

prob_left_by_dQbin = (
    df.group_by("dQ_binned").agg((pl.col("chose_left").mean()).alias("p_choose_left"))
    .sort("dQ_binned")
)

prob_left_by_dUncbin = (
    df.group_by("dUnc_binned").agg((pl.col("chose_left").mean()).alias("p_choose_left"))
    .sort("dUnc_binned")
)

prob_left_by_dNovbin = (
    df.group_by("dNov_binned").agg((pl.col("chose_left").mean()).alias("p_choose_left"))
    .sort("dNov_binned")
)



plot_df = pl.concat([
    prob_left_by_dQbin.with_columns(pl.lit("dQ").alias("feature")).rename({"dQ_binned": "bin"}),
    prob_left_by_dUncbin.with_columns(pl.lit("dUnc").alias("feature")).rename({"dUnc_binned": "bin"}),
    prob_left_by_dNovbin.with_columns(pl.lit("dNov").alias("feature")).rename({"dNov_binned": "bin"}),
])


plot_pd = plot_df.to_pandas()
sns.lineplot(data=plot_pd, x="bin", y="p_choose_left", hue="feature", marker="o")
plt.xlabel("Left–right difference quantile")
plt.ylabel("P(choose left)")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

#####################################################
## Figure 1F: Effect of variable on choice
# Logistic regression coefficients for EV, uncertainty, novelty, and interactions with trial number (EV x t; uncertainty x t ; novelty x t; 
# Positive values indicate seeking behaviour
CHOICE_VARS =  ["chose_left","logit_diff"]
df_model = df.select(norm_list(["block", "trial", *BANDIT_FEATURES, *CHOICE_VARS]))
feature_cols = ["trial", "dQ", "dUnc", "dNov"]
target_col = "chose_left"

df_model = (
    df_model
    .select([*feature_cols, target_col])
    .drop_nulls()
    .with_columns([
        (pl.col("trial") * pl.col("dQ")).alias("trial_dQ"),
        (pl.col("trial") * pl.col("dUnc")).alias("trial_dUnc"),
        (pl.col("trial") * pl.col("dNov")).alias("trial_dNov"),
    ])
)

feature_cols_use = [*feature_cols, "trial_dQ", "trial_dUnc", "trial_dNov"]
X_df = df_model.select(feature_cols_use).to_pandas()
y = df_model.select(target_col).to_series().to_numpy()

# simple 90/10 split without sklearn
rng = np.random.default_rng(42)
idx = np.arange(len(y))
rng.shuffle(idx)
split = int(len(y) * 0.9)
train_idx, test_idx = idx[:split], idx[split:]

X_train = sm.add_constant(X_df.iloc[train_idx], has_constant="add")
X_test  = sm.add_constant(X_df.iloc[test_idx],  has_constant="add")
y_train, y_test = y[train_idx], y[test_idx]

# logistic regression
logit_mod = sm.Logit(y_train, X_train).fit(disp=False)

# predict and simple metrics
proba = logit_mod.predict(X_test)
pred = (proba >= 0.5).astype(int)
acc = (pred == y_test).mean()
g

# coefficient table
coef_table = pd.DataFrame({
    "term": logit_mod.params.index,
    "coef": logit_mod.params,
    "std_err": logit_mod.bse,
    "z": logit_mod.tvalues,
    "p_value": logit_mod.pvalues,
    "odds_ratio": np.exp(logit_mod.params)
}).round(4)

print(logit_mod.summary())  # full statsmodels summary
print(f"Accuracy: {acc:.3f}")
print(coef_table)


