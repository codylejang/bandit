# Analysis 

RUN_CHOICE_ANALYSIS = True
RUN_FEATURE_IMPACT_ANALYSIS = False
RUN_AUTOCORRELATION_ANALYSIS = False
RUN_TRANSITION_ANALYSIS = False # does not work yet (probably never wont), must set to False

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

#########################################################################
# Choice Analysis
if RUN_CHOICE_ANALYSIS:
    print("Running Choice Analysis...")
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.preprocessing import PolynomialFeatures
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Build data (polars -> pandas to keep column names)
df_choice = load_dataframe(episode="max", verbose=True)
df_choice = df_choice.select(norm_list(["block", "trial", *BANDIT_FEATURES, *CHOICE_VARS]))
feature_cols = ["trial", "dQ", "dUnc", "dNov"]
target_col = "choice_side"

df_model = (
    df_choice
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

# fit logistic regression (binomial logit)
logit_mod = sm.Logit(y_train, X_train).fit(disp=False)

# predict and simple metrics
proba = logit_mod.predict(X_test)
pred = (proba >= 0.5).astype(int)
acc = (pred == y_test).mean()

def roc_auc_score_binary(y_true, y_score):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    cum_pos = np.cumsum(y_true)
    cum_neg = np.cumsum(1 - y_true)
    tpr = cum_pos / cum_pos[-1] if cum_pos[-1] > 0 else np.zeros_like(cum_pos)
    fpr = cum_neg / cum_neg[-1] if cum_neg[-1] > 0 else np.zeros_like(cum_neg)
    return np.trapz(tpr, fpr)

auc = roc_auc_score_binary(y_test, proba)

# coefficient table with p-values
coef_table = pd.DataFrame({
    "term": logit_mod.params.index,
    "coef": logit_mod.params,
    "std_err": logit_mod.bse,
    "z": logit_mod.tvalues,
    "p_value": logit_mod.pvalues,
    "odds_ratio": np.exp(logit_mod.params)
}).round(4)

print(logit_mod.summary())  # full statsmodels summary
print(f"Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")
print(coef_table)

  

#########################################################################
# FX: FEATURE IMPACT
# Analyze the impact of a feature on neuron activity, in a given layer, using Spearman correlation
def feature_impact(df, feature, layer , plot=True) -> tuple[pl.DataFrame, np.ndarray]:
    # assert layer is valid
    assert layer in LAYERS, f"To analyze feature impact, Layer must be one of {LAYERS}"

    col_neur = get_layer_columns(df, layer)
    x = df.select(feature).to_numpy().flatten()
    y = df.select(col_neur).to_numpy()

    corrs = []
    pvals = []
    x_std = np.nanstd(x)
    for j in range(y.shape[1]):
        yj = y[:, j]
        if x_std == 0 or np.nanstd(yj) == 0:
            corr = 0.0
            pval = 1.0
        else:
            corr, pval = stats.spearmanr(x, yj)
            if np.isnan(corr) or np.isnan(pval):
                corr = 0.0
                pval = 1.0
        corrs.append(corr)
        pvals.append(pval)

    neur_ids = parse_neuron_id(col_neur)
    df_impact = pl.DataFrame(
        {
            "Neuron_id": pl.Series(neur_ids, dtype=pl.Int64),
            "Corr": pl.Series(corrs, dtype=pl.Float64),
            "Corr_pval": pl.Series(pvals, dtype=pl.Float64),
        }
    )

    # keep neuron if it has at least 1 significant correlation (p < 0.05) with any trial feature
    sig_neurons = df_impact.filter(pl.col("Corr_pval") < 0.05).select("Neuron_id").to_numpy()
    sig_neurons = np.unique(sig_neurons.flatten())
    df_impact = df_impact.filter(pl.col("Corr_pval") < 0.05)

    if plot:
        plt.figure(figsize=(7, 3)) 
        if df_impact.is_empty():
            print(f"No significant neurons for {feature} in layer {layer}.")
        else:
            sns.barplot(
                data=df_impact.to_pandas(),
                x="Neuron_id",
                y="Corr",
                color="steelblue",
                ci=None,
                edgecolor="black",
            )
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f'Spearman Correlation between Neuron Activity and {feature} ({layer})')
        plt.xlabel('Neuron ID')
        plt.ylabel('Spearman Correlation Coefficient')
        plt.show()

    return df_impact,sig_neurons


#########################################################################
# FX: Plot response curves for a given neuron and feature
def plot_response(df, layer, neuron_id, feature = BANDIT_FEATURES[0])-> None:
    df2plot = pull_neuron_column(df, layer, neuron_id, other_cols=feature)
    df_pd = df2plot.to_pandas()
    corr, pval = stats.spearmanr(df_pd[feature], df_pd["Activity"])

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.3)

    sns.scatterplot(data=df_pd, x=feature, y="Activity", ax=axs[0])
    sns.regplot(data=df_pd, x=feature, y="Activity", ax=axs[0], scatter=False, color='red')
    axs[0].set_title(f'Neuron {neuron_id} Activity vs {feature} ({layer})')
    axs[0].set_xlabel(feature)
    axs[0].set_ylabel('Neuron Activity')
    axs[0].text(0.025, 0.975, f"Spearman r={corr:.2f}\np={pval:.3f}", transform=axs[0].transAxes, verticalalignment='top')

    sns.histplot(data=df_pd, x="Activity", ax=axs[1], kde=True, color='gray')
    axs[1].set_title(f'Neuron {neuron_id} Activity Distribution ({layer})')
    axs[1].set_ylabel('Count')
    axs[1].set_xlabel('Neuron Activity')

    plt.show()

#########################################################################
# FX: Plot joint tuning response for a given neuron and two features
def plot_joint_response(df, layer, neuron_id, feature_x=BANDIT_FEATURES[0], feature_y=BANDIT_FEATURES[1]) -> None:
    df2plot = pull_neuron_column(df, layer, neuron_id, other_cols=[feature_x, feature_y])
    df2plot_pd = df2plot.to_pandas()

    # first 1d scatter plot for each feature
    for feature in [feature_x, feature_y]:
        plot_response(df, layer, neuron_id, feature=feature)
        

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df2plot_pd, x=feature_x, y=feature_y, hue="Activity", palette="viridis", edgecolor='black')
    plt.title(f'Neuron {neuron_id} Joint Response ({layer})')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

    # Lets do a binned heatmap as well
    import pandas as pd
    plt.figure(figsize=(7, 6))
    heatmap_data = df2plot_pd.pivot_table(index=pd.cut(df2plot_pd[feature_y], bins=10),
                                             columns=pd.cut(df2plot_pd[feature_x], bins=10),
                                             values='Activity',
                                             aggfunc='mean')
    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Mean Neuron Activity'})
    plt.title(f'Neuron {neuron_id} Joint Response Heatmap ({layer})')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()




#########################################################################
# FX: AUTOCORRELATION
def autocorr(df: pl.DataFrame,columns: list[str], max_lag: int = TRIALS_PER_BLOCK - 1, plot: bool = True)->pl.DataFrame:
    df = df.select(columns)
    total_nulls = df.null_count().select(pl.sum_horizontal(pl.all())).item()
    assert total_nulls == 0, "DataFrame contains NaN values. Please handle them before computing autocorrelation."      

    # Init data frame
    df_acorr = pl.DataFrame({
        "Column": pl.Series([], dtype=pl.Utf8),
        "Lag": pl.Series([], dtype=pl.Int64),
        "Corr": pl.Series([], dtype=pl.Float64)
    })


    for col in columns:
        x = df.select(col).to_numpy().flatten()
        x_std = np.std(x)
        if x_std == 0 or np.isnan(x_std):
            # Constant signal -> define autocorr as 1 at lag 0, 0 elsewhere
            lags = np.arange(-len(x) + 1, len(x))
            corr = np.zeros_like(lags, dtype=float)
            corr[lags == 0] = 1.0
        else:
            x = (x - np.mean(x)) / x_std
            corr = np.correlate(x, x, mode='full')
            corr = corr / np.max(np.abs(corr))

        lags = np.arange(-len(x) + 1, len(x))
        max_lag_eff = min(max_lag, len(x) - 1)
        mask = (lags >=0) & (lags <= max_lag_eff)
        
        df_col = pl.DataFrame({"Column": [col] * mask.sum(),
                               "Lag": lags[mask],
                               "Corr": corr[mask]})
        df_acorr = pl.concat([df_acorr, df_col])

    if plot:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_acorr.to_pandas(), x="Lag", y="Corr", hue="Column", marker="o")
        plt.title("Autocorrelation of Neuron Activity")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.axhline(0, color='gray', linestyle='--')
        plt.legend(title='Column')
        plt.show()
    return df_acorr

#
########################################################################
# ANZ: FEATURE IMPACT ON NEURON ACTIVITY
if RUN_FEATURE_IMPACT_ANALYSIS:

    df = load_dataframe(episode="max", verbose=True)
    
    for layer in LAYERS:
        print(f"\nRunning feature impact for layer: {layer}")

        df_dq, _ = feature_impact(df, "dQ", layer, plot=True)
        df_dunc, _ = feature_impact(df, "dUnc", layer, plot=True)
        df_dnov, _ = feature_impact(df, "dNov", layer, plot=True)

        # Population summary of tuning by layer
        df_dq_labeled = df_dq.select(["Neuron_id"]).with_columns([pl.lit("dQ").alias("Tuned_to")])
        df_dunc_labeled = df_dunc.select(["Neuron_id"]).with_columns([pl.lit("dUnc").alias("Tuned_to")])
        df_dnov_labeled = df_dnov.select(["Neuron_id"]).with_columns([pl.lit("dNov").alias("Tuned_to")])
        df_tuned = pl.concat([df_dq_labeled, df_dunc_labeled, df_dnov_labeled])

        if df_tuned.is_empty():
            print(f"No significant tuning found for layer {layer}.")
        else:
            df_tuning_count = (
                df_tuned.group_by("Tuned_to")
                .agg(pl.count("Neuron_id").alias("Count"))
                .sort("Tuned_to")
            )
            total_neurons = LAYER_SIZES.get(layer, len(get_layer_columns(df, layer)))
            df_tuning_percent = df_tuning_count.with_columns(
                (pl.col("Count") / total_neurons * 100).alias("Percent")
            )
            print(df_tuning_percent)

        # Response to single feature: find the single strongest tuned neuron for dQ
        if df_dq.is_empty():
            print(f"No significant dQ-tuned neurons found in layer {layer}.")
        else:
            get_strongest = pl.col("Corr").abs().arg_max()
            neuron_max_dq = df_dq.select(get_strongest).item()
            plot_response(df, layer, neuron_max_dq, "dQ")

        # Joint selectivity: find neuron with max joint selectivity for dQ and dUnc
        df_dq_dunc = df_dq.join(df_dunc, on=["Neuron_id"], suffix="_dunc")

        if df_dq_dunc.is_empty():
            print(f"No neurons with joint selectivity for dQ and dUnc in layer {layer}.")
        else:
            sum_selectivity = pl.col("Corr").abs() + pl.col("Corr_dunc").abs()
            df_dq_dunc = df_dq_dunc.with_columns([sum_selectivity.alias("Joint_Selectivity")])
            neuron_max_joint = (
                df_dq_dunc.sort("Joint_Selectivity", descending=True)
                .select(pl.col("Neuron_id").first())
                .item()
            )
            plot_joint_response(df, layer, neuron_max_joint, feature_x="dQ", feature_y="dUnc")






#########################################################################
# ANZ: AUTOCORRELATION ----
# Example: autocorrelation for a specific neuron
if RUN_AUTOCORRELATION_ANALYSIS:
    print("Running Autocorrelation Analysis...")
    eg_neuron = 10
    layers_to_analyze = LAYERS
    # eg_columns = [build_layer_column("h_stim", eg_neuron)]
    # eg_excorr = autocorr(df, columns=eg_columns, max_lag=TRIALS_PER_BLOCK-1, plot=True)  
    def autocorr_full_analysis(
        df: pl.DataFrame,
        layer: str,
        max_lag: int,
        degree: int = 2,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        # All neuron columns for a single layer
        neuron_cols = get_layer_columns(df, layer)

        # Autocorr for every neuron column
        df_acorr = autocorr(df, columns=neuron_cols, max_lag=max_lag, plot=False)

        rows = []
        for col in neuron_cols:
            df_col_all = df_acorr.filter((pl.col("Column") == col) & (pl.col("Lag") >= 0))
            lags_all = df_col_all.select("Lag").to_numpy().flatten()
            corr_all = df_col_all.select("Corr").to_numpy().flatten()
            if len(lags_all) < 3:
                # Not enough points to fit quadratic
                continue
            corr_all = np.nan_to_num(corr_all, nan=0.0, posinf=0.0, neginf=0.0)

            X = np.column_stack([lags_all ** k for k in range(1, degree + 1)])
            X = sm.add_constant(X)
            model = sm.OLS(corr_all, X).fit()

            params = np.nan_to_num(model.params, nan=0.0, posinf=0.0, neginf=0.0)
            pvals = np.nan_to_num(model.pvalues, nan=1.0, posinf=1.0, neginf=1.0)
            resid = np.nan_to_num(model.resid, nan=0.0, posinf=0.0, neginf=0.0)
            corr_lag0 = corr_all[lags_all == 0]
            corr_lag1 = corr_all[lags_all == 1]
            if len(corr_lag0) == 0 or len(corr_lag1) == 0:
                lag1_drop = float("nan")
            else:
                lag1_drop = float(corr_lag0[0] - corr_lag1[0])

            rows.append(
                (
                    col,
                    float(params[0]),
                    float(params[1]) if degree >= 1 else float("nan"),
                    float(params[2]) if degree >= 2 else float("nan"),
                    float(pvals[0]),
                    float(pvals[1]) if degree >= 1 else float("nan"),
                    float(pvals[2]) if degree >= 2 else float("nan"),
                    float(model.rsquared) if np.isfinite(model.rsquared) else 0.0,
                    float(model.rsquared_adj) if np.isfinite(model.rsquared_adj) else 0.0,
                    float(np.mean(resid ** 2)) if np.isfinite(np.mean(resid ** 2)) else 0.0,
                    float(np.std(resid)) if np.isfinite(np.std(resid)) else 0.0,
                    lag1_drop,
                )
            )

        df_metrics = pl.DataFrame(
            rows,
            schema=[
                "Column",
                "Beta0",
                "Beta1",
                "Beta2",
                "Pval_Beta0",
                "Pval_Beta1",
                "Pval_Beta2",
                "R2",
                "R2_adj",
                "MSE",
                "Resid_Std",
                "Lag1_Drop",
            ],
        ).with_columns(
            pl.col("Column").str.split("_").list.get(-1).cast(pl.Int64).alias("Neuron_id"),
            pl.when(pl.col("Lag1_Drop").is_nan()).then(None).otherwise(pl.col("Lag1_Drop")).alias("Lag1_Drop"),
        )

        return df_acorr, df_metrics


    def add_significance_flags(df_metrics: pl.DataFrame, alpha: float = 0.05) -> pl.DataFrame:
        return df_metrics.with_columns(
            [
                (pl.col("Pval_Beta1") < alpha).alias("Beta1_sig"),
                (pl.col("Pval_Beta2") < alpha).alias("Beta2_sig"),
            ]
        )

    def cluster_autocorr_metrics_dbscan(
        df_metrics: pl.DataFrame,
        eps: float | None = None,
        min_samples: int = 8,
    ) -> pl.DataFrame:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors

        if df_metrics.is_empty():
            return df_metrics.with_columns(pl.lit(-1).alias("Cluster"))

        feats = ["Beta0", "Beta1", "Beta2", "R2", "MSE", "Resid_Std"]
        df_feat = df_metrics.select(feats).to_pandas()
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        if df_feat.isna().any().any():
            n_cells = int(df_feat.isna().sum().sum())
            print(f"Imputing {n_cells} NaNs/Infs in clustering features with column medians.")
            df_feat = df_feat.fillna(df_feat.median())
            df_feat = df_feat.fillna(0.0)
        X = StandardScaler().fit_transform(df_feat.values)

        if eps is None:
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
            distances, _ = nbrs.kneighbors(X)
            k_dist = np.sort(distances[:, -1])
            eps = suggest_eps_from_k_distance(k_dist)
            if (not np.isfinite(eps)) or (eps <= 0):
                if len(k_dist) > 0:
                    eps = float(np.percentile(k_dist, 90))
                if (not np.isfinite(eps)) or (eps <= 0):
                    eps = 0.5
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        return df_metrics.with_columns(pl.Series("Cluster", labels))

    def plot_fit_examples(
        df_acorr,
        df_metrics,
        n_examples=4,
        mode="top_r2",
        beta1_sig: bool | None = None,
        beta2_sig: bool | None = None,
        title: str | None = None,
    ):
        if mode == "top_r2":
            dfm = df_metrics.sort("R2", descending=True).head(n_examples)
            title = title or "Autocorr examples with quadratic fit (top R2)"
        else:
            filt = pl.lit(True)
            if beta1_sig is not None:
                filt = filt & (pl.col("Beta1_sig") == beta1_sig)
            if beta2_sig is not None:
                filt = filt & (pl.col("Beta2_sig") == beta2_sig)
            dfm = df_metrics.filter(filt).sort("R2", descending=True).head(n_examples)
            title = title or f"Examples (Beta1_sig={beta1_sig}, Beta2_sig={beta2_sig})"

        if dfm.is_empty():
            plt.figure(figsize=(8, 5))
            plt.title(f"{title} — no examples")
            plt.xlabel("Lag")
            plt.ylabel("Correlation")
            plt.axhline(0, color="gray", linestyle="--")
            plt.show()
            return

        dfm = dfm.to_pandas()
        plt.figure(figsize=(8, 5))
        for _, row in dfm.iterrows():
            col = row["Column"]
            df_col = df_acorr.filter(pl.col("Column") == col).to_pandas()
            lags = df_col["Lag"].to_numpy()
            corr = df_col["Corr"].to_numpy()
            fit = row["Beta0"] + row["Beta1"] * lags + row["Beta2"] * (lags ** 2)
            plt.plot(lags, corr, marker="o", alpha=0.6, label=f"{col} data")
            plt.plot(lags, fit, linestyle="--", alpha=0.8, label=f"{col} fit")
        plt.title(title)
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.axhline(0, color="gray", linestyle="--")
        plt.legend()
        plt.show()

    def plot_fit_examples_grid(df_acorr, df_metrics, n_examples=4):
        combos = [
            (True, True, "Beta1 & Beta2 significant"),
            (True, False, "Beta1 only significant"),
            (False, True, "Beta2 only significant"),
            (False, False, "Neither significant"),
        ]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        axs = axs.flatten()

        for ax, (b1, b2, title) in zip(axs, combos):
            dfm = (
                df_metrics.filter((pl.col("Beta1_sig") == b1) & (pl.col("Beta2_sig") == b2))
                .sort("R2", descending=True)
                .head(n_examples)
            )

            if dfm.is_empty():
                ax.set_title(f"{title}\n(no examples)")
                ax.axhline(0, color="gray", linestyle="--")
                ax.set_xlabel("Lag")
                ax.set_ylabel("Correlation")
                continue

            dfm = dfm.to_pandas()
            for _, row in dfm.iterrows():
                col = row["Column"]
                df_col = df_acorr.filter(pl.col("Column") == col).to_pandas()
                lags = df_col["Lag"].to_numpy()
                corr = df_col["Corr"].to_numpy()
                fit = row["Beta0"] + row["Beta1"] * lags + row["Beta2"] * (lags ** 2)
                ax.plot(lags, corr, marker="o", alpha=0.6)
                ax.plot(lags, fit, linestyle="--", alpha=0.8)

            ax.set_title(title)
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")

        plt.suptitle("Autocorr example fits by significance combo", y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_coeff_scatter(df_metrics):
        dfm = df_metrics.to_pandas()
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=dfm, x="Beta1", y="Beta2", hue="Cluster", palette="tab10")
        plt.title("Coefficient space (Beta1 vs Beta2)")
        plt.xlabel("Beta1 (linear)")
        plt.ylabel("Beta2 (quadratic)")
        plt.axhline(0, color="gray", linestyle="--")
        plt.axvline(0, color="gray", linestyle="--")
        plt.show()

    def plot_metric_histograms(df_metrics):
        dfm = df_metrics.to_pandas()
        for col in ["Beta1", "Beta2", "R2", "MSE", "Resid_Std", "Pval_Beta1", "Pval_Beta2", "Lag1_Drop"]:
            plt.figure(figsize=(6, 4))
            sns.histplot(data=dfm, x=col, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.show()

    def suggest_eps_from_k_distance(k_dist: np.ndarray) -> float:
        if len(k_dist) < 3:
            return float("nan")
        x = np.linspace(0, 1, len(k_dist))
        y = (k_dist - np.min(k_dist)) / (np.max(k_dist) - np.min(k_dist) + 1e-12)
        # distance to line between endpoints (knee detection)
        line = y[0] + (y[-1] - y[0]) * x
        dist = y - line
        knee_idx = int(np.argmax(dist))
        return float(k_dist[knee_idx])

    def plot_k_distance(df_metrics: pl.DataFrame, k: int = 5):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        if df_metrics.is_empty():
            print("No metrics available for k-distance plot.")
            return

        feats = ["Beta0", "Beta1", "Beta2", "R2", "MSE", "Resid_Std"]
        df_feat = df_metrics.select(feats).to_pandas()
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        if df_feat.isna().any().any():
            n_cells = int(df_feat.isna().sum().sum())
            print(f"Imputing {n_cells} NaNs/Infs in k-distance features with column medians.")
            df_feat = df_feat.fillna(df_feat.median())
            df_feat = df_feat.fillna(0.0)
        X = StandardScaler().fit_transform(df_feat.values)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_dist = np.sort(distances[:, -1])

        eps_suggested = suggest_eps_from_k_distance(k_dist)

        plt.figure(figsize=(6, 4))
        plt.plot(k_dist)
        plt.title(f"k-distance plot (k={k})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{k}-NN distance")
        if not np.isnan(eps_suggested):
            plt.axhline(eps_suggested, color="red", linestyle="--", label=f"eps~{eps_suggested:.3f}")
            plt.legend()
        plt.show()
        print(f"Suggested eps from k-distance elbow: {eps_suggested:.3f}")

    def summarize_clusters(df_metrics: pl.DataFrame):
        if df_metrics.is_empty():
            print("No metrics available for cluster summary.")
            return

        summary = (
            df_metrics.group_by(["Cluster", "Beta1_sig", "Beta2_sig"])
            .agg(
                pl.count("Neuron_id").alias("Count"),
                pl.mean("R2").alias("Mean_R2"),
                pl.mean("Resid_Std").alias("Mean_Resid_Std"),
            )
            .sort(["Cluster", "Beta1_sig", "Beta2_sig"])
        )
        print("Cluster summary by significance flags:")
        print(summary)

    def add_layer_type(df_metrics: pl.DataFrame) -> pl.DataFrame:
        if df_metrics.is_empty():
            return df_metrics
        return df_metrics.with_columns(
            pl.col("Layer").str.split("_").list.get(0).alias("LayerType")
        )

    def summarize_layer_comparison(df_metrics: pl.DataFrame):
        if df_metrics.is_empty():
            print("No metrics available for layer comparison summary.")
            return
        summary = (
            df_metrics.group_by("LayerType")
            .agg(
                pl.count("Neuron_id").alias("Count"),
                pl.mean("Beta1").alias("Mean_Beta1"),
                pl.mean("Beta2").alias("Mean_Beta2"),
                pl.mean("R2").alias("Mean_R2"),
                pl.mean("MSE").alias("Mean_MSE"),
                pl.mean("Resid_Std").alias("Mean_Resid_Std"),
                pl.mean("Lag1_Drop").alias("Mean_Lag1_Drop"),
                pl.mean("Beta1_sig").alias("Frac_Beta1_sig"),
                pl.mean("Beta2_sig").alias("Frac_Beta2_sig"),
                pl.mean("Forgetful").alias("Frac_Forgetful"),
            )
            .sort("LayerType")
        )
        print("LayerType summary (means/fractions):")
        print(summary)

    def plot_layer_comparisons(df_metrics: pl.DataFrame):
        if df_metrics.is_empty():
            print("No metrics available for layer comparison plots.")
            return
        dfm = df_metrics.to_pandas()

        plt.figure(figsize=(7, 4))
        sns.boxplot(data=dfm, x="LayerType", y="Lag1_Drop")
        plt.title("Lag1_Drop by LayerType")
        plt.xlabel("LayerType")
        plt.ylabel("Lag1_Drop")
        plt.show()

        plt.figure(figsize=(7, 4))
        sns.boxplot(data=dfm, x="LayerType", y="R2")
        plt.title("R2 by LayerType")
        plt.xlabel("LayerType")
        plt.ylabel("R2")
        plt.show()

        plt.figure(figsize=(7, 4))
        sns.boxplot(data=dfm, x="LayerType", y="Resid_Std")
        plt.title("Resid_Std by LayerType")
        plt.xlabel("LayerType")
        plt.ylabel("Resid_Std")
        plt.show()

        plt.figure(figsize=(7, 4))
        sns.countplot(data=dfm, x="LayerType", hue="Cluster")
        plt.title("Cluster counts by LayerType")
        plt.xlabel("LayerType")
        plt.ylabel("Count")
        plt.show()

    def add_forgetfulness_flag(df_metrics: pl.DataFrame, quantile: float = 0.75) -> pl.DataFrame:
        if df_metrics.is_empty():
            return df_metrics.with_columns(pl.lit(False).alias("Forgetful"))
        df_non_null = df_metrics.filter(pl.col("Lag1_Drop").is_finite())
        if df_non_null.is_empty():
            return df_metrics.with_columns(pl.lit(False).alias("Forgetful"))
        threshold = df_non_null.select(pl.col("Lag1_Drop").quantile(quantile)).item()
        return df_metrics.with_columns(
            (pl.col("Lag1_Drop").is_finite() & (pl.col("Lag1_Drop") >= threshold)).alias("Forgetful")
        )

    def summarize_forgetfulness(df_metrics: pl.DataFrame):
        if df_metrics.is_empty():
            print("No metrics available for forgetfulness summary.")
            return
        summary = (
            df_metrics.group_by("Forgetful")
            .agg(
                pl.count("Neuron_id").alias("Count"),
                pl.mean("Lag1_Drop").alias("Mean_Lag1_Drop"),
                pl.median("Lag1_Drop").alias("Median_Lag1_Drop"),
            )
            .sort("Forgetful")
        )
        print("Forgetfulness summary (based on Lag1_Drop quantile):")
        print(summary)

    def cluster_centroids(df_metrics: pl.DataFrame) -> pl.DataFrame:
        if df_metrics.is_empty():
            print("No metrics available for cluster centroids.")
            return df_metrics
        centroids = (
            df_metrics.group_by("Cluster")
            .agg(
                pl.mean("Beta0").alias("Mean_Beta0"),
                pl.mean("Beta1").alias("Mean_Beta1"),
                pl.mean("Beta2").alias("Mean_Beta2"),
                pl.mean("R2").alias("Mean_R2"),
                pl.mean("MSE").alias("Mean_MSE"),
                pl.mean("Resid_Std").alias("Mean_Resid_Std"),
                pl.count("Neuron_id").alias("Count"),
            )
            .sort("Cluster")
        )
        print("Cluster centroids (means):")
        print(centroids)
        return centroids

    def plot_cluster_panels(df_metrics: pl.DataFrame):
        if df_metrics.is_empty():
            print("No metrics available for cluster panels.")
            return
        dfm = df_metrics.to_pandas()
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.scatterplot(data=dfm, x="Beta1", y="R2", hue="Cluster", palette="tab10", ax=axs[0])
        axs[0].set_title("Beta1 vs R2 by Cluster")
        axs[0].set_xlabel("Beta1 (linear)")
        axs[0].set_ylabel("R2")

        sns.scatterplot(data=dfm, x="Beta2", y="R2", hue="Cluster", palette="tab10", ax=axs[1])
        axs[1].set_title("Beta2 vs R2 by Cluster")
        axs[1].set_xlabel("Beta2 (quadratic)")
        axs[1].set_ylabel("R2")
        plt.tight_layout()
        plt.show()

    def plot_mse_vs_betas(df_metrics: pl.DataFrame):
        if df_metrics.is_empty():
            print("No metrics available for MSE vs Beta plots.")
            return
        dfm = df_metrics.to_pandas()
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.scatterplot(data=dfm, x="Beta1", y="MSE", hue="Cluster", palette="tab10", ax=axs[0])
        axs[0].set_title("MSE vs Beta1 by Cluster")
        axs[0].set_xlabel("Beta1 (linear)")
        axs[0].set_ylabel("MSE")

        sns.scatterplot(data=dfm, x="Beta2", y="MSE", hue="Cluster", palette="tab10", ax=axs[1])
        axs[1].set_title("MSE vs Beta2 by Cluster")
        axs[1].set_xlabel("Beta2 (quadratic)")
        axs[1].set_ylabel("MSE")
        plt.tight_layout()
        plt.show()

    def plot_forgetfulness_examples(df_acorr, df_metrics, n_examples=4, mode="most"):
        if df_metrics.is_empty():
            print("No metrics available for forgetfulness examples.")
            return

        if mode == "least":
            dfm = df_metrics.sort("Lag1_Drop").head(n_examples)
            title = "Least forgetful (small Lag1_Drop)"
        else:
            dfm = df_metrics.sort("Lag1_Drop", descending=True).head(n_examples)
            title = "Most forgetful (large Lag1_Drop)"

        if dfm.is_empty():
            plt.figure(figsize=(8, 5))
            plt.title(f"{title} — no examples")
            plt.xlabel("Lag")
            plt.ylabel("Correlation")
            plt.axhline(0, color="gray", linestyle="--")
            plt.show()
            return

        dfm = dfm.to_pandas()
        plt.figure(figsize=(8, 5))
        for _, row in dfm.iterrows():
            col = row["Column"]
            df_col = df_acorr.filter(pl.col("Column") == col).to_pandas()
            lags = df_col["Lag"].to_numpy()
            corr = df_col["Corr"].to_numpy()
            plt.plot(lags, corr, marker="o", alpha=0.6, label=f"{col}")
        plt.title(title)
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.axhline(0, color="gray", linestyle="--")
        plt.legend()
        plt.show()

    #####
    df = load_dataframe(episode="max", verbose=True)

    all_metrics = []
    all_acorr = []
    for layer in layers_to_analyze:
        print(f"\nAutocorr analysis for layer: {layer}")
        df_acorr, df_metrics = autocorr_full_analysis(
            df,
            layer=layer,
            max_lag=TRIALS_PER_BLOCK - 1,
            degree=2,
        )

        # tag layer for joining later
        df_metrics = df_metrics.with_columns(pl.lit(layer).alias("Layer"))
        df_acorr = df_acorr.with_columns(pl.lit(layer).alias("Layer"))
        all_metrics.append(df_metrics)
        all_acorr.append(df_acorr)

    df_metrics_all = pl.concat(all_metrics) if all_metrics else pl.DataFrame()
    df_acorr_all = pl.concat(all_acorr) if all_acorr else pl.DataFrame()

    # Global clustering across all layers
    df_metrics_all = add_significance_flags(df_metrics_all, alpha=0.05)
    df_metrics_all = add_forgetfulness_flag(df_metrics_all, quantile=0.75)
    df_metrics_all = cluster_autocorr_metrics_dbscan(df_metrics_all, eps=None, min_samples=8)
    df_metrics_all = add_layer_type(df_metrics_all)

    # Global summaries (across all layers)
    print(df_metrics_all.sort("R2", descending=True).head(10))
    plot_k_distance(df_metrics_all, k=5)
    summarize_clusters(df_metrics_all)
    summarize_forgetfulness(df_metrics_all)
    summarize_layer_comparison(df_metrics_all)
    cluster_centroids(df_metrics_all)
    plot_cluster_panels(df_metrics_all)
    plot_mse_vs_betas(df_metrics_all)
    plot_layer_comparisons(df_metrics_all)

    # Per-layer plots using global cluster labels
    for layer in layers_to_analyze:
        df_metrics = df_metrics_all.filter(pl.col("Layer") == layer)
        df_acorr = df_acorr_all.filter(pl.col("Layer") == layer)

        if df_metrics.is_empty() or df_acorr.is_empty():
            print(f"No data to plot for layer: {layer}")
            continue

        print(f"\nPlots for layer: {layer}")
        plot_fit_examples(df_acorr, df_metrics, n_examples=4, mode="top_r2")
        plot_fit_examples_grid(df_acorr, df_metrics, n_examples=4)
        plot_coeff_scatter(df_metrics)
        plot_metric_histograms(df_metrics)
        plot_cluster_panels(df_metrics)
        plot_mse_vs_betas(df_metrics)
        plot_forgetfulness_examples(df_acorr, df_metrics, n_examples=4, mode="most")
        plot_forgetfulness_examples(df_acorr, df_metrics, n_examples=4, mode="least")

    df_metrics_all.write_csv("df_metrics_all.csv")




