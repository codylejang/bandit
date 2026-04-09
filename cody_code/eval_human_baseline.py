"""
Human-human similarity baseline: is it chance?

For each non-repeating pair of subjects, computes the same similarity
metrics used in model-human evaluation. This establishes whether the
model-human composite score is meaningfully above the human-human baseline.

Approach:
  - Fit each subject's policy as P(left) = logistic(beta * dQ + intercept)
  - For pair (A, B): use A's fitted policy to predict B's trial-by-trial choices
  - Compute: choice agreement, NLL, policy shape correlation
  - Also compute a "chance" baseline (P(left) = 0.5 always)

Metrics are symmetric: pair (A,B) computes A→B and B→A, reports both.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit  # logistic sigmoid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir,
    "data", "aquinoshareddata", "shared", "behavior", "aquino_behavior.parquet",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
MODEL_METRICS_PATH = os.path.join(OUT_DIR, "checkpoint_metrics.csv")


# ── data loading ──────────────────────────────────────────────────────────────

def load_human_data():
    """Load all 22 subjects from parquet, derive left/right stim + choice side + beta Q."""
    df = pd.read_parquet(PARQUET_PATH)

    # drop trials with NaN outcome (held-out / unseen trials)
    df = df.dropna(subset=["selectedVector", "selectStimID", "rejectStimID", "outcome"])

    # derive choice side and left/right stim IDs
    sv = df["selectedVector"].astype(int)
    df["choice_side"] = sv - 1  # 0=left, 1=right
    df["left_stim_id"] = np.where(sv == 1, df["selectStimID"], df["rejectStimID"]).astype(int)
    df["right_stim_id"] = np.where(sv == 1, df["rejectStimID"], df["selectStimID"]).astype(int)
    df["chosen_stim_id"] = df["selectStimID"].astype(int)
    df["reward"] = df["outcome"].astype(int)
    df["block"] = df["blockID"].astype(int)
    df["trial"] = df["trialInBlock"].astype(int)
    df["sub_id"] = df["patientId"]

    df = df.sort_values(["sub_id", "block", "trial"]).reset_index(drop=True)

    # compute beta-posterior Q values
    df = _compute_beta_q(df)

    return df


def _compute_beta_q(df):
    """Per subject, per block: track beta(a,b) for each stim."""
    ql, qr, dq_list = [], [], []
    ul, ur, dunc_list = [], [], []

    for (sub, blk), grp in df.groupby(["sub_id", "block"]):
        alpha, beta = {}, {}
        for _, row in grp.iterrows():
            lid, rid = row["left_stim_id"], row["right_stim_id"]
            for sid in [lid, rid]:
                if sid not in alpha:
                    alpha[sid] = 1.0
                    beta[sid] = 1.0

            q_l = alpha[lid] / (alpha[lid] + beta[lid])
            q_r = alpha[rid] / (alpha[rid] + beta[rid])

            def bvar(a, b):
                ab = a + b
                return (a * b) / (ab ** 2 * (ab + 1))

            u_l = bvar(alpha[lid], beta[lid])
            u_r = bvar(alpha[rid], beta[rid])

            ql.append(q_l)
            qr.append(q_r)
            dq_list.append(q_l - q_r)
            ul.append(u_l)
            ur.append(u_r)
            dunc_list.append(u_l - u_r)

            cid = row["chosen_stim_id"]
            if row["reward"] == 1:
                alpha[cid] += 1.0
            else:
                beta[cid] += 1.0

    df = df.copy()
    df["QL"] = ql
    df["QR"] = qr
    df["dQ"] = dq_list
    df["UncL"] = ul
    df["UncR"] = ur
    df["dUnc"] = dunc_list
    return df


# ── per-subject policy fit ────────────────────────────────────────────────────

def fit_subject_policy(subj_df):
    """
    Fit P(choose_left) = logistic(beta * dQ + alpha) via MLE.
    Returns (beta, alpha) coefficients.
    """
    dQ = subj_df["dQ"].values
    chose_left = (1 - subj_df["choice_side"]).values  # choice_side=0 is left

    # simple logistic regression via scipy minimize
    from scipy.optimize import minimize

    def neg_ll(params):
        a, b = params
        p = expit(a + b * dQ)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.sum(chose_left * np.log(p) + (1 - chose_left) * np.log(1 - p))

    result = minimize(neg_ll, x0=[0.0, 1.0], method="Nelder-Mead")
    return result.x  # (intercept, slope)


def predict_p_left(params, dQ_values):
    """Given fitted (intercept, slope), return P(left) for each dQ."""
    a, b = params
    return expit(a + b * dQ_values)


# ── pairwise metrics ─────────────────────────────────────────────────────────

def compute_pair_metrics(subj_a_params, subj_b_df, subj_b_params, n_bins=8):
    """
    Use subject A's fitted policy to predict subject B's choices.
    Returns dict of metrics (same as model-human eval).
    """
    m = {}
    dQ_b = subj_b_df["dQ"].values
    chose_left_b = (1 - subj_b_df["choice_side"]).values

    # A's predicted P(left) for B's trials
    p_left_a = predict_p_left(subj_a_params, dQ_b)
    p_left_a = np.clip(p_left_a, 1e-7, 1 - 1e-7)

    # 1. choice agreement: A's greedy prediction vs B's actual choice
    pred_left_a = (p_left_a > 0.5).astype(int)
    m["choice_agreement"] = float((pred_left_a == chose_left_b).mean())

    # 2. NLL of B's choices under A's policy
    p_b_choice = np.where(chose_left_b == 1, p_left_a, 1 - p_left_a)
    m["nll"] = float(-np.log(p_b_choice).mean())

    # 3. policy shape correlation (dQ-binned P(left) curves)
    df_tmp = pd.DataFrame({
        "dQ": dQ_b,
        "chose_left_b": chose_left_b,
        "p_left_a": p_left_a,
    })
    df_tmp["dQ_bin"] = pd.qcut(df_tmp["dQ"], q=n_bins, duplicates="drop")
    binned = df_tmp.groupby("dQ_bin", observed=True).agg(
        b_p_left=("chose_left_b", "mean"),
        a_p_left=("p_left_a", "mean"),
    )
    if len(binned) >= 3:
        r, _ = stats.pearsonr(binned["b_p_left"], binned["a_p_left"])
        m["policy_shape_r"] = float(r)
    else:
        m["policy_shape_r"] = float("nan")

    # 4. dQ sensitivity comparison (slope of A vs slope of B)
    m["slope_a"] = float(subj_a_params[1])
    m["slope_b"] = float(subj_b_params[1])
    m["slope_diff"] = float(abs(subj_a_params[1] - subj_b_params[1]))

    return m


def compute_chance_metrics(subj_df, n_bins=8):
    """Baseline: P(left) = 0.5 always (pure chance)."""
    m = {}
    chose_left = (1 - subj_df["choice_side"]).values

    m["choice_agreement"] = float(max(chose_left.mean(), 1 - chose_left.mean()))
    m["nll"] = float(np.log(2))  # -log(0.5) = ln(2) ≈ 0.693
    m["policy_shape_r"] = 0.0

    return m


# ── main analysis ─────────────────────────────────────────────────────────────

def run_pairwise_analysis(human_df):
    subjects = sorted(human_df["sub_id"].unique())
    print(f"{len(subjects)} subjects → {len(subjects) * (len(subjects)-1) // 2} unique pairs\n")

    # fit each subject's policy
    print("fitting per-subject logistic policies...")
    subject_params = {}
    subject_data = {}
    for sub in subjects:
        sdf = human_df[human_df["sub_id"] == sub]
        params = fit_subject_policy(sdf)
        subject_params[sub] = params
        subject_data[sub] = sdf
        print(f"  {sub}: intercept={params[0]:.3f}, slope={params[1]:.3f} "
              f"({len(sdf)} trials, P(left)={1-sdf['choice_side'].mean():.3f})")

    # compute pairwise metrics (A→B for all ordered pairs, then average per unordered pair)
    pair_rows = []
    for a, b in itertools.combinations(subjects, 2):
        m_ab = compute_pair_metrics(subject_params[a], subject_data[b], subject_params[b])
        m_ba = compute_pair_metrics(subject_params[b], subject_data[a], subject_params[a])

        # average both directions
        avg = {
            "subj_a": a,
            "subj_b": b,
            "choice_agreement": (m_ab["choice_agreement"] + m_ba["choice_agreement"]) / 2,
            "nll": (m_ab["nll"] + m_ba["nll"]) / 2,
            "policy_shape_r": (m_ab["policy_shape_r"] + m_ba["policy_shape_r"]) / 2,
            "slope_diff": m_ab["slope_diff"],
            # also store directional values
            "agree_ab": m_ab["choice_agreement"],
            "agree_ba": m_ba["choice_agreement"],
            "nll_ab": m_ab["nll"],
            "nll_ba": m_ba["nll"],
            "shape_r_ab": m_ab["policy_shape_r"],
            "shape_r_ba": m_ba["policy_shape_r"],
        }
        pair_rows.append(avg)

    pairs_df = pd.DataFrame(pair_rows)

    # chance baseline
    chance = compute_chance_metrics(human_df)

    return pairs_df, subject_params, chance


# ── visualization ─────────────────────────────────────────────────────────────

def plot_baseline(pairs_df, subject_params, chance, human_df):
    os.makedirs(OUT_DIR, exist_ok=True)

    # load model metrics if available
    model_metrics = None
    if os.path.exists(MODEL_METRICS_PATH):
        model_metrics = pd.read_csv(MODEL_METRICS_PATH, index_col="episode")
        print(f"loaded model metrics ({len(model_metrics)} checkpoints)")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # ── (0,0) choice agreement distribution ──
    ax = axes[0, 0]
    ax.hist(pairs_df["choice_agreement"], bins=12, alpha=0.7, color="steelblue",
            edgecolor="black", label="human-human pairs")
    ax.axvline(pairs_df["choice_agreement"].mean(), color="navy", ls="--", lw=2,
               label=f'mean={pairs_df["choice_agreement"].mean():.3f}')
    ax.axvline(0.5, color="gray", ls=":", label="chance=0.50")
    if model_metrics is not None:
        best_idx = model_metrics["composite"].idxmax() if "composite" in model_metrics else None
        if best_idx is not None:
            ax.axvline(model_metrics.loc[best_idx, "choice_agreement"], color="red", ls="--",
                       lw=2, label=f'best model={model_metrics.loc[best_idx, "choice_agreement"]:.3f}')
    ax.set_xlabel("choice agreement")
    ax.set_ylabel("count (pairs)")
    ax.set_title("choice agreement: human-human")
    ax.legend(fontsize=7)

    # ── (0,1) NLL distribution ──
    ax = axes[0, 1]
    ax.hist(pairs_df["nll"], bins=12, alpha=0.7, color="seagreen", edgecolor="black",
            label="human-human pairs")
    ax.axvline(pairs_df["nll"].mean(), color="darkgreen", ls="--", lw=2,
               label=f'mean={pairs_df["nll"].mean():.3f}')
    ax.axvline(chance["nll"], color="gray", ls=":", label=f'chance={chance["nll"]:.3f}')
    if model_metrics is not None and best_idx is not None:
        ax.axvline(model_metrics.loc[best_idx, "nll"], color="red", ls="--", lw=2,
                   label=f'best model={model_metrics.loc[best_idx, "nll"]:.3f}')
    ax.set_xlabel("NLL")
    ax.set_ylabel("count (pairs)")
    ax.set_title("NLL: human-human")
    ax.legend(fontsize=7)

    # ── (0,2) policy shape r distribution ──
    ax = axes[0, 2]
    ax.hist(pairs_df["policy_shape_r"], bins=12, alpha=0.7, color="orchid", edgecolor="black",
            label="human-human pairs")
    ax.axvline(pairs_df["policy_shape_r"].mean(), color="purple", ls="--", lw=2,
               label=f'mean={pairs_df["policy_shape_r"].mean():.3f}')
    ax.axvline(0.0, color="gray", ls=":", label="chance=0.0")
    if model_metrics is not None and best_idx is not None:
        ax.axvline(model_metrics.loc[best_idx, "policy_shape_r"], color="red", ls="--", lw=2,
                   label=f'best model={model_metrics.loc[best_idx, "policy_shape_r"]:.3f}')
    ax.set_xlabel("policy shape r")
    ax.set_ylabel("count (pairs)")
    ax.set_title("policy shape r: human-human")
    ax.legend(fontsize=7)

    # ── (1,0) all subjects' psychometric curves overlaid ──
    ax = axes[1, 0]
    subjects = sorted(human_df["sub_id"].unique())
    dQ_range = np.linspace(-0.6, 0.6, 100)
    for sub in subjects:
        p = subject_params[sub]
        ax.plot(dQ_range, expit(p[0] + p[1] * dQ_range), alpha=0.5, label=sub)
    ax.axhline(0.5, ls=":", c="gray")
    ax.axvline(0.0, ls=":", c="gray")
    ax.set_xlabel("dQ (left - right)")
    ax.set_ylabel("P(choose left)")
    ax.set_title("individual psychometric curves")
    ax.legend(fontsize=6, ncol=2)

    # ── (1,1) slope distribution ──
    ax = axes[1, 1]
    slopes = [subject_params[s][1] for s in subjects]
    ax.bar(range(len(subjects)), slopes, color="teal", edgecolor="black", alpha=0.7)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, fontsize=7)
    ax.axhline(np.mean(slopes), color="navy", ls="--", label=f"mean={np.mean(slopes):.2f}")
    ax.set_xlabel("subject")
    ax.set_ylabel("logistic slope (dQ sensitivity)")
    ax.set_title("dQ sensitivity per subject")
    ax.legend(fontsize=8)

    # ── (1,2) summary comparison: human-human vs model vs chance ──
    ax = axes[1, 2]
    metrics = ["choice_agreement", "nll", "policy_shape_r"]
    labels = ["choice agree", "NLL", "shape r"]
    x = np.arange(len(metrics))
    width = 0.25

    hh_means = [pairs_df[m].mean() for m in metrics]
    hh_stds = [pairs_df[m].std() for m in metrics]
    chance_vals = [chance.get(m, 0) for m in metrics]

    bars_hh = ax.bar(x - width, hh_means, width, yerr=hh_stds, label="human-human",
                     color="steelblue", alpha=0.7, capsize=3)
    bars_ch = ax.bar(x, chance_vals, width, label="chance", color="gray", alpha=0.5)

    if model_metrics is not None and best_idx is not None:
        model_vals = [model_metrics.loc[best_idx, m] for m in metrics]
        bars_md = ax.bar(x + width, model_vals, width, label=f"best model (ep{best_idx})",
                         color="red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("human-human vs model vs chance")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "human_baseline.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nsaved figure to {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== human-human similarity baseline ===\n")

    human_df = load_human_data()
    print(f"loaded {len(human_df)} trials, {human_df['sub_id'].nunique()} subjects\n")

    pairs_df, subject_params, chance = run_pairwise_analysis(human_df)

    # summary stats
    print(f"\n── human-human pair metrics (n={len(pairs_df)} pairs) ──")
    print(f"  choice agreement: {pairs_df['choice_agreement'].mean():.3f} "
          f"± {pairs_df['choice_agreement'].std():.3f}  "
          f"(range {pairs_df['choice_agreement'].min():.3f}–{pairs_df['choice_agreement'].max():.3f})")
    print(f"  NLL:              {pairs_df['nll'].mean():.4f} "
          f"± {pairs_df['nll'].std():.4f}  "
          f"(range {pairs_df['nll'].min():.4f}–{pairs_df['nll'].max():.4f})")
    print(f"  policy shape r:   {pairs_df['policy_shape_r'].mean():.3f} "
          f"± {pairs_df['policy_shape_r'].std():.3f}  "
          f"(range {pairs_df['policy_shape_r'].min():.3f}–{pairs_df['policy_shape_r'].max():.3f})")

    print(f"\n── chance baseline ──")
    print(f"  choice agreement: 0.500")
    print(f"  NLL:              {chance['nll']:.4f}")
    print(f"  policy shape r:   0.000")

    # save
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "human_human_pairs.csv")
    pairs_df.to_csv(csv_path, index=False)
    print(f"\nsaved pair metrics to {csv_path}")

    plot_baseline(pairs_df, subject_params, chance, human_df)


if __name__ == "__main__":
    main()
