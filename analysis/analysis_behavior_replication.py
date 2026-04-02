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
DF = DF.with_columns(
    pl.col("patientId").cast(pl.Int64)).sort(["patientId","blockID"])


###################################################################################
# FIGURE 1G
df = DF[["patientId","blockID","selectedVector",
         "qValsLeft","qValsRight",
         "uValLeft", "uValRight",          
         "nValLeft", "nValRight",
         ]]

print(df)


# compute differences
df = df.with_columns([
    (pl.col("selectedVector")==1).cast(pl.Int8).alias("select_left"),    
    (pl.col("qValsLeft")-pl.col("qValsRight")).alias("dQ"),
    (pl.col("uValLeft")-pl.col("uValRight")).alias("dUnc"),
    (pl.col("nValLeft")-pl.col("nValRight")).alias("dNov"),
])    


n_quantiles = 2;
df = df.group_by("patientId").map_groups(
    lambda g: g.with_columns([
        pl.col("dQ").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dQ_binned"),
        pl.col("dUnc").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dUnc_binned"),
        pl.col("dNov").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dNov_binned")
    ])
)

df = df.sort(["patientId","blockID"])
print(df)


# Long format 
df_long = df.unpivot(
    on=["dQ_binned","dUnc_binned","dNov_binned"],
    index = ["patientId","blockID","select_left"],
    variable_name="BanditVariable",
    value_name="dLRQuantile",
)

df_long = (
    df_long
    .group_by(["patientId","BanditVariable","dLRQuantile"])
    .agg((pl.col("select_left").mean()).alias("prob_select_left"))
).sort(["patientId","BanditVariable","dLRQuantile"])

print(df_long)

df_plot = (
    df_long
    .group_by(["BanditVariable", "dLRQuantile"])
    .agg([
        pl.col("prob_select_left").mean(),
        (pl.col("prob_select_left").std(ddof=1) / pl.len().sqrt()).alias("sem"),
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
        subdf["prob_select_left"],
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
# FIGURE 1D

df = DF[["patientId","blockID","selectedVector",
                     "select_qVals","reject_qVals",
                    "select_uVal","reject_uVal",            
                    "select_nVal","reject_nVal"]]

df.head()

df = df.with_columns([
    (pl.col("select_qVals")-pl.col("reject_qVals")).alias("dQ"),
    (pl.col("selectedVector")==1).cast(pl.Int8).alias("select_left"),

    pl.when(pl.col("select_uVal") > pl.col("reject_uVal")).then(1.0)
      .when(pl.col("select_uVal") < pl.col("reject_uVal")).then(0.0)
      .otherwise(None)
      .alias("select_uncertain"),

    pl.when(pl.col("select_nVal") > pl.col("reject_nVal")).then(1.0)
      .when(pl.col("select_nVal") < pl.col("reject_nVal")).then(0.0)
      .otherwise(None)
      .alias("select_novel"),
])

df = df.group_by("patientId").map_groups(
    lambda g: g.with_columns(
        pl.col("dQ").qcut(5, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dQ_binned")
    )
)

df = (
    df
    .group_by(["patientId","dQ_binned"])
    .agg(
        [(pl.col("select_left").mean()).alias("prob_select_left"),
         (pl.col("select_uncertain").mean()).alias("prob_select_unc"),
         (pl.col("select_novel").mean()).alias("prob_select_nov")
        ]
    )
    .sort(["patientId","dQ_binned"])
)

# Long format 
df_long = df.unpivot(
    on=["prob_select_unc","prob_select_left","prob_select_nov"],
    index = ["patientId","dQ_binned"],
    variable_name="choice_type",
    value_name="p_chosen",
)

df_plot = (
    df_long
    .group_by(["dQ_binned", "choice_type"])
    .agg([
        pl.col("p_chosen").mean().alias("mean_p"),
        (pl.col("p_chosen").std(ddof=1) / pl.len().sqrt()).alias("sem_p"),
    ])
    .sort(["choice_type", "dQ_binned"])
)

# Plot 

df_plot = df_plot.to_pandas()
df_plot["sem_p"] = df_plot["sem_p"].fillna(0)

label_map = {
    "prob_select_left": "Left option",
    "prob_select_unc": "Uncertain option",
    "prob_select_nov": "New option",
}

fig, ax = plt.subplots(figsize=(6, 4))

for choice_type, subdf in df_plot.groupby("choice_type"):
    subdf = subdf.sort_values("dQ_binned")

    ax.errorbar(
        subdf["dQ_binned"],
        subdf["mean_p"],
        yerr=subdf["sem_p"],
        marker="o",
        capsize=4,
        label=label_map[choice_type],
    )

ax.set_xlabel("EV difference quantile")
ax.set_ylabel("Proportion chosen")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.show()