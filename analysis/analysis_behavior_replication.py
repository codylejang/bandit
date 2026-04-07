import polars as pl
# from analysis.experiment_config import AQUINO_BEH_PQT_DATA
from experiment_config import AQUINO_BEH_PQT_DATA

import seaborn as sns
import matplotlib.pyplot as plt

run_df_inspection = False
run_fig_1D = False
run_fig_1E = False
run_fig_1G = False
run_fig_1F = True

DF = pl.read_parquet(AQUINO_BEH_PQT_DATA)
DF = DF[
    [
    "patientId","trialAccum","blockID","trialInBlock","outcome",
    "selectedVector",
    "select_qVals","reject_qVals","qValsLeft","qValsRight",
    "select_uVal","reject_uVal","uValLeft", "uValRight",             
    "select_nVal","reject_nVal", "nValLeft", "nValRight",
    "Q1Left", "Q1Right", "Q2Left", "Q2Right",
    "Q3Left", "Q3Right", "Q4Left", "Q4Right",
    "Q5Left", "Q5Right"
    ]
]
print(DF)
# DF = DF.filter(pl.col("trialInBlock")<=15)
#################################################################################
# PREP DATA
DF = (DF.with_columns(pl.col("patientId").cast(pl.Int64))
    .sort(["patientId","blockID"]))
    
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




# VALIDATE

tol = 1e-9

DF_check = DF.with_columns([
    (
        ((pl.col("select_qVals") - pl.col("qValsLeft")).abs() < tol) &
        ((pl.col("select_uVal") - pl.col("uValLeft")).abs() < tol) &
        ((pl.col("select_nVal") - pl.col("nValLeft")).abs() < tol)
    ).alias("selected_matches_left"),

    (
        ((pl.col("select_qVals") - pl.col("qValsRight")).abs() < tol) &
        ((pl.col("select_uVal") - pl.col("uValRight")).abs() < tol) &
        ((pl.col("select_nVal") - pl.col("nValRight")).abs() < tol)
    ).alias("selected_matches_right"),
])

print(
    DF_check.select([
        (((pl.col("selectedVector") == 1) == pl.col("selected_matches_left")).mean()).alias("agreement_left"),
        (((pl.col("selectedVector") != 1) == pl.col("selected_matches_right")).mean()).alias("agreement_right"),
    ])
)


############################################################################
# DATA DESCRIBE + VISUALIZE
if run_df_inspection:
    df = DF[["patientId","blockID","trialInBlock","chose_left",
            "select_qVals","reject_qVals","qValsLeft","qValsRight"]]
    df = df.sort(["patientId","blockID","trialInBlock"])


    print(df)


    # Long format 
    df = (
        df
        .group_by(["patientId","trialInBlock"])
        .agg([
            pl.col("chose_left").mean(),
            pl.col("select_qVals").mean(),
            pl.col("reject_qVals").mean(),
            pl.col("qValsRight").mean(),
            pl.col("qValsLeft").mean(),
        ])
    )
    df = df.sort(["patientId","trialInBlock"])
    print(df)

    df_long = df.unpivot(
        on=["chose_left","select_qVals","reject_qVals","qValsRight","qValsLeft"],
        index =["patientId","trialInBlock"],
        variable_name="Variable",
        value_name="value"
    )
    print(df_long)

    df_long = (
        df_long
        .group_by(["Variable","trialInBlock"])
        .agg([
            pl.col("value").mean().alias("mean_value"),
            (pl.col("value").std(ddof=1) / pl.len().sqrt()).alias("sem_value"),
        ])
        .sort(["Variable","trialInBlock"])
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

    for choice_var, subdf in df_plot.groupby("Variable"):
        subdf = subdf.sort_values("trialInBlock")
        # print(subdf)
        varname = subdf["Variable"].unique()
        print(varname)
        ax.errorbar(
            subdf["trialInBlock"],
            subdf["mean_value"],
            yerr=subdf["sem_value"],
            marker="o",
            capsize=4,
            label=varname,
        )

    ax.set_xlabel("Trial number in block")
    ax.set_ylabel("Proportion of higher variable chosen")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    plt.show()


############################################################################
# FIGURE 1E: EFFECT OF TRIAL HORIZON ON PREFERENCE
# Proportion of trials in which patients chose the option 
# with higher EV (blue), uncertainty (black) or novelty (magenta), 
# as a function of trial number. 
# The dots and bars indicate the mean and s.e.m.
# n = 22 sessions.
if run_fig_1E:

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
if run_fig_1G:
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


if run_fig_1D:
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




###################################################### 
# FIGURE 1F: # Logistic regression coefficients for EV, uncertainty, novelty, and interactions with trial number 
# (EV x t; uncertainty x t ; novelty x t)Positive values indicate seeking behaviour
###################################################### 

if run_fig_1F:

    # from analysis.experiment_config import BANDIT_FEATURES, TRIALS_PER_BLOCK
    # from analysis.helpers import norm_list

    from experiment_config import BANDIT_FEATURES, TRIALS_PER_BLOCK
    from helpers import norm_list

    import polars as pl
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import statsmodels.api as sm
    from dfutils import load_dataframe, pull_neuron_column, get_layer_columns,parse_neuron_id
    import pandas as pd


    CHOICE_VARS =  ["chose_left"]
    # df = 
    df = DF.select(norm_list(["patientId","blockID", "trialInBlock", *BANDIT_FEATURES, *CHOICE_VARS]))
    feature_cols = ["patientId","trialInBlock", "dQ", "dUnc", "dNov"]
    target_col = "chose_left"

    df = (
        df
        .select([*feature_cols, target_col])
        .drop_nulls()
        .with_columns([
            (pl.col("trialInBlock") * pl.col("dQ")).alias("trial_dQ"),
            (pl.col("trialInBlock") * pl.col("dUnc")).alias("trial_dUnc"),
            (pl.col("trialInBlock") * pl.col("dNov")).alias("trial_dNov"),
        ])
    )

    feature_cols_use = ["dQ", "dUnc", "dNov", "trial_dQ", "trial_dUnc", "trial_dNov"]
    df_coeffs = []  # collect one table per patient

    for ii in df["patientId"].unique().to_list():
        # print(ii)
        df_patient = df.filter(pl.col("patientId")==ii)
        
        print(df_patient)
   
        X_df = df_patient.select(feature_cols_use).to_pandas()
        y = df_patient.select(target_col).to_series().to_numpy()

        # simple 90/10 split without sklearn
        rng = np.random.default_rng(42)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        split = int(len(y) * 0.9)
        train_idx, test_idx = idx[:split], idx[split:]


        X = sm.add_constant(df_patient.select(feature_cols_use).to_pandas(), has_constant="add")
        y = df_patient.select(target_col).to_series().to_numpy()

        logit_mod = sm.Logit(y, X).fit(disp=False)
        # X_train = sm.add_constant(X_df.iloc[train_idx], has_constant="add")
        # X_test  = sm.add_constant(X_df.iloc[test_idx],  has_constant="add")
        # y_train, y_test = y[train_idx], y[test_idx]
        # logit_mod = sm.Logit(y_train, X_train).fit(disp=False)

        # predict and simple metrics
        # proba = logit_mod.predict(X_test)
        # pred = (proba >= 0.5).astype(int)
        # acc = (pred == y_test).mean()


        # coefficient table
        patient_coef = pd.DataFrame({
            "patientId": ii,
            "term": logit_mod.params.index,
            "coef": logit_mod.params,
            "std_err": logit_mod.bse,
            "z": logit_mod.tvalues,
            "p_value": logit_mod.pvalues,
            "odds_ratio": np.exp(logit_mod.params)
        }).round(4)

        df_coeffs.append(patient_coef)
        # print(f"Patient{ii} results:")
        # print(logit_mod.summary())  # full statsmodels summary
        # print(f"Accuracy: {acc:.3f}")
        # print(coef_table)
    print(df_coeffs)


    import seaborn as sns
    import matplotlib.pyplot as plt

    df_coeffs = pd.concat(df_coeffs, ignore_index=True)  # make a DataFrame

    df_plot = df_coeffs.loc[df_coeffs["term"] != "const"].copy()  # drop intercept
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=df_plot, x="term", y="coef", inner="box", cut=0)
    sns.stripplot(data=df_plot, x="term", y="coef", color="k", alpha=0.25, jitter=0.15, size=3)
    plt.axhline(0, color="gray", lw=1, ls="--")
    plt.ylabel("Logit coefficient")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# CHAT GPT's SLOP VERSION FOR FIGURE 1F
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# Assumes DF already contains:
# patientId, trialInBlock, dQ, dUnc, dNov, chose_left

# --------------------------------------------------
# Prepare design matrix
# --------------------------------------------------
df = (
    DF.select(["patientId", "trialInBlock", "dQ", "dUnc", "dNov", "chose_left"])
      .drop_nulls()
      .with_columns([
          (pl.col("trialInBlock") * pl.col("dQ")).alias("trial_dQ"),
          (pl.col("trialInBlock") * pl.col("dUnc")).alias("trial_dUnc"),
          (pl.col("trialInBlock") * pl.col("dNov")).alias("trial_dNov"),
      ])
      .sort(["patientId", "trialInBlock"])
)

feature_cols = ["dQ", "dUnc", "dNov", "trial_dQ", "trial_dUnc", "trial_dNov"]
term_order = ["dQ", "dUnc", "dNov", "trial_dQ", "trial_dUnc", "trial_dNov"]

label_map = {
    "dQ": "EV",
    "dUnc": "Uncertainty",
    "dNov": "Novelty",
    "trial_dQ": "EV:t",
    "trial_dUnc": "Unc:t",
    "trial_dNov": "Nov:t",
}

# --------------------------------------------------
# Fit one logistic regression per session
# --------------------------------------------------
coef_rows = []
failed_sessions = []

for sess in df["patientId"].unique().to_list():
    df_sess = df.filter(pl.col("patientId") == sess)

    X = df_sess.select(feature_cols).to_pandas()
    y = df_sess["chose_left"].to_numpy()

    X = sm.add_constant(X, has_constant="add")

    try:
        # Main fit
        res = sm.Logit(y, X).fit(disp=False, maxiter=200)
    except Exception:
        try:
            # Fallback if Logit has trouble converging
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        except Exception as e:
            failed_sessions.append((sess, str(e)))
            continue

    for term in ["const"] + feature_cols:
        coef_rows.append({
            "patientId": sess,
            "term": term,
            "coef": float(res.params[term]),
            "std_err": float(res.bse[term]),
            "z": float(res.tvalues[term]) if hasattr(res, "tvalues") else np.nan,
            "p_value": float(res.pvalues[term]),
        })

df_coeffs = pd.DataFrame(coef_rows)

if failed_sessions:
    print("Failed sessions:")
    for fs in failed_sessions:
        print(fs)

# --------------------------------------------------
# Summary stats across sessions
# --------------------------------------------------
df_plot = df_coeffs[df_coeffs["term"].isin(term_order)].copy()

summary_rows = []
for term in term_order:
    vals = df_plot.loc[df_plot["term"] == term, "coef"].to_numpy()
    mean_coef = np.mean(vals)
    sem_coef = stats.sem(vals, nan_policy="omit")
    t_stat, p_val = stats.ttest_1samp(vals, 0.0, nan_policy="omit")
    summary_rows.append({
        "term": term,
        "mean_coef": mean_coef,
        "sem_coef": sem_coef,
        "t_stat": t_stat,
        "p_val": p_val,
        "n_sessions": len(vals),
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary)

# --------------------------------------------------
# Plot like Fig. 1f:
# session dots + mean ± SEM
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))

xpos = np.arange(len(term_order))

# individual session dots
rng = np.random.default_rng(42)
for i, term in enumerate(term_order):
    vals = df_plot.loc[df_plot["term"] == term, "coef"].to_numpy()
    jitter = rng.normal(0, 0.04, size=len(vals))
    ax.scatter(
        np.full(len(vals), xpos[i]) + jitter,
        vals,
        s=28,
        alpha=0.7,
        color="black",
        zorder=2,
    )

# mean ± SEM
ax.errorbar(
    xpos,
    df_summary["mean_coef"],
    yerr=df_summary["sem_coef"],
    fmt="o",
    color="tab:red",
    capsize=5,
    markersize=7,
    linewidth=2,
    zorder=3,
)

# zero line
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

# significance stars
def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "NS"

ymin, ymax = df_plot["coef"].min(), df_plot["coef"].max()
yr = ymax - ymin if ymax > ymin else 1.0

for i, row in df_summary.iterrows():
    y = row["mean_coef"] + np.sign(row["mean_coef"] if row["mean_coef"] != 0 else 1) * (row["sem_coef"] + 0.08 * yr)
    ax.text(xpos[i], y, stars(row["p_val"]), ha="center", va="bottom", fontsize=10)

ax.set_xticks(xpos)
ax.set_xticklabels([label_map[t] for t in term_order])
ax.set_ylabel("Estimate")
ax.set_xlabel("")
fig.tight_layout()
plt.show()