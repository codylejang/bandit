# eda on human session behavior csvs + q-value extraction via beta posterior
# mirrors the beta-based q-value approach used in RL_persist.py (BlockBeta)

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = os.path.join(os.path.dirname(__file__), "session_behavior_csvs")


# ── load all csvs into one dataframe ──
def load_all_sessions():
    files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


# ── beta posterior q-values (same as RL_persist BlockBeta) ──
def compute_q_values(df):
    # per subject, per session, per block: track beta(a,b) for each stim
    # q = a/(a+b), uncertainty = var of beta
    q_left, q_right, q_chosen = [], [], []
    unc_left, unc_right = [], []
    dq_list, dunc_list = [], []

    for (sub, sess, blk), grp in df.groupby(["sub_id", "session", "block"]):
        # beta params per stim in this block
        alpha = {}
        beta = {}

        for _, row in grp.iterrows():
            lid = row["left_stim_id"]
            rid = row["right_stim_id"]
            cid = row["chosen_stim_id"]

            # init unseen stims with uniform prior
            for sid in [lid, rid]:
                if sid not in alpha:
                    alpha[sid] = 1.0
                    beta[sid] = 1.0

            # q = mean of beta posterior
            ql = alpha[lid] / (alpha[lid] + beta[lid])
            qr = alpha[rid] / (alpha[rid] + beta[rid])

            # uncertainty = var of beta
            def beta_var(a, b):
                ab = a + b
                return (a * b) / (ab ** 2 * (ab + 1))

            ul = beta_var(alpha[lid], beta[lid])
            ur = beta_var(alpha[rid], beta[rid])

            q_left.append(ql)
            q_right.append(qr)
            q_chosen.append(ql if row["choice_side"] == 0 else qr)
            unc_left.append(ul)
            unc_right.append(ur)
            dq_list.append(ql - qr)
            dunc_list.append(ul - ur)

            # update beta posterior with reward
            if row["reward"] == 1:
                alpha[cid] += 1.0
            else:
                beta[cid] += 1.0

    df = df.copy()
    df["QL"] = q_left
    df["QR"] = q_right
    df["Q_chosen"] = q_chosen
    df["UncL"] = unc_left
    df["UncR"] = unc_right
    df["dQ"] = dq_list
    df["dUnc"] = dunc_list
    return df


# ── basic eda plots ──
def plot_eda(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1) trials per subject
    trials_per_sub = df.groupby("sub_id").size()
    axes[0, 0].bar(range(len(trials_per_sub)), trials_per_sub.values)
    axes[0, 0].set_xlabel("subject")
    axes[0, 0].set_ylabel("num trials")
    axes[0, 0].set_title("trials per subject")

    # 2) overall reward rate per block position
    reward_by_block = df.groupby("block")["reward"].mean()
    axes[0, 1].bar(reward_by_block.index, reward_by_block.values)
    axes[0, 1].set_xlabel("block")
    axes[0, 1].set_ylabel("mean reward")
    axes[0, 1].set_title("reward rate by block")
    axes[0, 1].axhline(0.5, ls="--", c="gray")

    # 3) choice side bias (fraction choosing left)
    left_frac = 1 - df.groupby("sub_id")["choice_side"].mean()
    axes[0, 2].bar(range(len(left_frac)), left_frac.values)
    axes[0, 2].axhline(0.5, ls="--", c="gray")
    axes[0, 2].set_xlabel("subject")
    axes[0, 2].set_ylabel("p(choose left)")
    axes[0, 2].set_title("left choice bias per subject")

    # 4) dQ vs p(choose left)
    # bin dQ and compute fraction choosing left in each bin
    df["dQ_bin"] = pd.qcut(df["dQ"], q=6, duplicates="drop")
    dq_agg = df.groupby("dQ_bin").agg(
        p_left=("choice_side", lambda x: 1 - x.mean()),
        dQ_mean=("dQ", "mean"),
    )
    axes[1, 0].plot(dq_agg["dQ_mean"], dq_agg["p_left"], "o-")
    axes[1, 0].set_xlabel("dQ (left - right)")
    axes[1, 0].set_ylabel("p(choose left)")
    axes[1, 0].set_title("value-guided choice (dQ)")
    axes[1, 0].axhline(0.5, ls="--", c="gray")
    axes[1, 0].axvline(0.0, ls="--", c="gray")

    # 5) dUnc vs p(choose left)
    df["dUnc_bin"] = pd.qcut(df["dUnc"], q=6, duplicates="drop")
    dunc_agg = df.groupby("dUnc_bin").agg(
        p_left=("choice_side", lambda x: 1 - x.mean()),
        dUnc_mean=("dUnc", "mean"),
    )
    axes[1, 1].plot(dunc_agg["dUnc_mean"], dunc_agg["p_left"], "o-")
    axes[1, 1].set_xlabel("dUnc (left - right)")
    axes[1, 1].set_ylabel("p(choose left)")
    axes[1, 1].set_title("uncertainty-guided choice (dUnc)")
    axes[1, 1].axhline(0.5, ls="--", c="gray")
    axes[1, 1].axvline(0.0, ls="--", c="gray")

    # 6) q_chosen distribution
    axes[1, 2].hist(df["Q_chosen"], bins=30, edgecolor="black", alpha=0.7)
    axes[1, 2].set_xlabel("Q(chosen)")
    axes[1, 2].set_ylabel("count")
    axes[1, 2].set_title("distribution of chosen Q values")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "eda_human_behavior.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    df = load_all_sessions()
    print(f"loaded {len(df)} trials across {df['sub_id'].nunique()} subjects")
    print(f"columns: {df.columns.tolist()}")
    print()

    # basic stats
    print("── summary stats ──")
    print(f"sessions per subject: {df.groupby('sub_id')['session'].nunique().to_dict()}")
    print(f"blocks per session: {df.groupby(['sub_id', 'session'])['block'].nunique().describe()}")
    print(f"mean reward rate: {df['reward'].mean():.3f}")
    print(f"left choice rate: {(1 - df['choice_side'].mean()):.3f}")
    print()

    # compute q values
    df = compute_q_values(df)
    print("── q-value stats ──")
    print(df[["QL", "QR", "Q_chosen", "dQ", "UncL", "UncR", "dUnc"]].describe().round(4))
    print()

    # save enriched csv
    out_path = os.path.join(os.path.dirname(__file__), "session_behavior_csvs", "all_subjects_with_qvalues.csv")
    df.to_csv(out_path, index=False)
    print(f"saved enriched dataframe to {out_path}")

    plot_eda(df)
