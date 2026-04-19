"""
Model selection by human-similarity

For each saved checkpoint, replays human trial sequences through the model
in synchronized mode (model observes human's choices & rewards) and computes:

  1. Choice agreement  — % trials where model greedy choice == human choice
  2. NLL              — negative log-likelihood of human choices under model softmax
  3. Policy shape r   — correlation of dQ-binned P(left) curves (model vs human)
  4. Q-value MSE      — |model_beta_Q - human_beta_Q| (sanity; ~0 in sync mode)
  5. Uncertainty MSE   — same for variance of beta posterior

- Synchronized mode means the model sees the human's (choice, reward) at feedback
so the LSTM hidden state evolves as if the model were "watching" the human play
- Beta-posterior Q values are recomputed from scratch per block (identical formula)
- Data loaded from parquet (all 22 subjects)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RL_persist import RLAgent, BlockBeta, CHECKPOINT_DIR
from eval_human_baseline import load_human_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")


# synchronized replay

@torch.no_grad()
def replay_human_trials(agent, human_df):
    """
    Replay every human trial through the model in synchronized mode.

    Per subject (= one episode):
      - randomize embeddings (fresh identity tokens)
      - remap stim IDs to sequential slots
      - reset hidden state at block boundaries
      - at each trial: stim → decision → feedback (using human's choice/reward)

    Returns a DataFrame with per-trial model + human columns.
    """
    agent.policy_network.eval()

    rows = []

    for sub, grp in human_df.groupby("sub_id"):
        hidden = None
        beta_map = {}
        current_block = None

        # fresh random embeddings for this subject (= fresh episode)
        agent.randomize_embeddings()

        # remap stim IDs to sequential slots for this subject
        unique_stims = sorted(
            set(grp["left_stim_id"].unique()) | set(grp["right_stim_id"].unique())
        )
        stim_remap = {orig: idx for idx, orig in enumerate(unique_stims)}

        for _, row in grp.iterrows():
            block = int(row["block"])
            trial = int(row["trial"])
            left_id_orig = int(row["left_stim_id"])
            right_id_orig = int(row["right_stim_id"])
            left_id = stim_remap[left_id_orig]
            right_id = stim_remap[right_id_orig]

            # reset at block boundaries (matches training: hidden=None per block)
            if block != current_block:
                hidden = None
                beta_map = {}
                current_block = block

            # init beta for unseen stims in this block (keyed by original ID)
            for sid in [left_id_orig, right_id_orig]:
                if sid not in beta_map:
                    beta_map[sid] = BlockBeta()

            # model's beta-posterior Q (before this trial's feedback)
            model_QL = beta_map[left_id_orig].mean
            model_QR = beta_map[right_id_orig].mean
            model_UncL = beta_map[left_id_orig].var
            model_UncR = beta_map[right_id_orig].var

            info = {"block": block, "trial": trial}

            # stim step
            id_feats = agent._embed_pair(left_id, right_id)
            x_stim = agent._make_step_input(id_feats, agent.state_encoder.stim(info))
            _, _, hidden = agent.policy_network(x_stim, hidden)

            # decision step
            x_dec = agent._make_step_input(id_feats, agent.state_encoder.decision(info))
            lstm_out, _, hidden = agent.policy_network(x_dec, hidden)
            context = lstm_out[:, -1, :]

            emb_l = agent.id_embedding(torch.tensor([left_id], device=agent.device, dtype=torch.long))
            emb_r = agent.id_embedding(torch.tensor([right_id], device=agent.device, dtype=torch.long))
            logits = agent.policy_network.get_policy_logits(context, emb_l, emb_r)

            probs = F.softmax(logits, dim=-1)
            model_p_left = float(probs[0, 0].item())
            model_choice = int(logits.argmax(dim=-1).item())
            logit_diff = float((logits[0, 0] - logits[0, 1]).item())

            # feedback step: use HUMAN's choice & reward
            human_choice = int(row["choice_side"])
            human_reward = float(row["reward"])
            human_chosen_id_orig = int(row["chosen_stim_id"])
            human_chosen_id = stim_remap[human_chosen_id_orig]

            id_feats_fb = agent._embed_feedback(human_chosen_id, human_choice)
            x_fb = agent._make_step_input(
                id_feats_fb, agent.state_encoder.feedback(human_reward, info)
            )
            _, _, hidden = agent.policy_network(x_fb, hidden)

            # update beta with human's observed reward
            beta_map[human_chosen_id_orig].update(human_reward)

            rows.append(
                {
                    "sub_id": row["sub_id"],
                    "block": block,
                    "trial": trial,
                    "left_stim_id": left_id_orig,
                    "right_stim_id": right_id_orig,
                    # human
                    "human_choice": human_choice,
                    "human_reward": human_reward,
                    "human_QL": float(row["QL"]),
                    "human_QR": float(row["QR"]),
                    "human_dQ": float(row["dQ"]),
                    "human_UncL": float(row["UncL"]),
                    "human_UncR": float(row["UncR"]),
                    "human_dUnc": float(row["dUnc"]),
                    # model
                    "model_choice": model_choice,
                    "model_p_left": model_p_left,
                    "model_logit_diff": logit_diff,
                    "model_QL": model_QL,
                    "model_QR": model_QR,
                    "model_dQ": float(model_QL - model_QR),
                    "model_UncL": model_UncL,
                    "model_UncR": model_UncR,
                    "model_dUnc": float(model_UncL - model_UncR),
                }
            )

    return pd.DataFrame(rows)


# metrics

def compute_metrics(replay_df, n_bins=8):
    """
    Compute similarity metrics from a synchronized replay DataFrame.

    Returns dict with:
      choice_agreement    — fraction of trials model agrees with human
      nll                 — mean negative log-likelihood of human choices
      policy_shape_r      — pearson r between dQ-binned P(left) curves
      q_mse               — mean squared error of beta-posterior Q (sanity)
      unc_mse             — mean squared error of beta-posterior uncertainty
      logit_dq_r2         — r^2 of model logit_diff ~ human dQ regression
    """
    m = {}

    # 1. choice agreement
    m["choice_agreement"] = float((replay_df["model_choice"] == replay_df["human_choice"]).mean())

    # 2. NLL of human choices under model softmax
    eps = 1e-7
    p_human = np.where(
        replay_df["human_choice"] == 0,
        replay_df["model_p_left"],
        1.0 - replay_df["model_p_left"],
    )
    p_human = np.clip(p_human, eps, 1.0 - eps)
    m["nll"] = float(-np.log(p_human).mean())

    # 3. policy shape correlation (dQ-binned P(left) for model vs human)
    df = replay_df.copy()
    df["dQ_bin"] = pd.qcut(df["human_dQ"], q=n_bins, duplicates="drop")
    binned = df.groupby("dQ_bin", observed=True).agg(
        human_p_left=("human_choice", lambda x: 1.0 - x.mean()),
        model_p_left=("model_p_left", "mean"),
    )
    if len(binned) >= 3:
        r, _ = stats.pearsonr(binned["human_p_left"], binned["model_p_left"])
        m["policy_shape_r"] = float(r)
    else:
        m["policy_shape_r"] = float("nan")

    # 4. Q-value MSE (beta posterior — should be ~0 in sync mode)
    m["q_mse"] = float(
        ((replay_df["model_QL"] - replay_df["human_QL"]) ** 2).mean()
        + ((replay_df["model_QR"] - replay_df["human_QR"]) ** 2).mean()
    )

    # 5. uncertainty MSE
    m["unc_mse"] = float(
        ((replay_df["model_UncL"] - replay_df["human_UncL"]) ** 2).mean()
        + ((replay_df["model_UncR"] - replay_df["human_UncR"]) ** 2).mean()
    )

    # 6. logit_diff vs human dQ — r^2
    slope, intercept, r_val, p_val, se = stats.linregress(
        replay_df["human_dQ"], replay_df["model_logit_diff"]
    )
    m["logit_dq_r2"] = float(r_val ** 2)
    m["logit_dq_slope"] = float(slope)

    return m


# sweep all checkpoints

def evaluate_all_checkpoints(human_df, checkpoint_dir=CHECKPOINT_DIR):
    """
    Load each checkpoint, replay human trials, compute metrics.
    Returns a summary DataFrame indexed by episode.
    """
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "ep*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")

    print(f"found {len(ckpt_files)} checkpoints in {checkpoint_dir}")

    agent = RLAgent(hidden_size=128)

    summary_rows = []

    for path in ckpt_files:
        ep = agent.load_checkpoint(path)
        replay_df = replay_human_trials(agent, human_df)
        metrics = compute_metrics(replay_df)
        metrics["episode"] = ep
        metrics["checkpoint"] = os.path.basename(path)
        summary_rows.append(metrics)
        print(
            f"  ep {ep:3d} | agree {metrics['choice_agreement']:.3f} | "
            f"NLL {metrics['nll']:.3f} | shape_r {metrics['policy_shape_r']:.3f} | "
            f"logit~dQ r² {metrics['logit_dq_r2']:.3f}"
        )

    summary = pd.DataFrame(summary_rows).set_index("episode").sort_index()
    return summary


# composite score & selection

def select_best(summary_df):
    """
    Rank checkpoints by composite score.
    Higher is better for: choice_agreement, policy_shape_r, logit_dq_r2
    Lower is better for: nll
    """
    df = summary_df.copy()

    for col in ["choice_agreement", "policy_shape_r", "logit_dq_r2"]:
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / (sd + 1e-8)

    mu, sd = df["nll"].mean(), df["nll"].std()
    df["nll_z"] = -(df["nll"] - mu) / (sd + 1e-8)

    df["composite"] = (
        df["choice_agreement_z"]
        + df["nll_z"]
        + df["policy_shape_r_z"]
        + df["logit_dq_r2_z"]
    ) / 4.0

    best_ep = df["composite"].idxmax()
    return best_ep, df


# visualization

def plot_results(summary_df, best_ep, human_df, checkpoint_dir=CHECKPOINT_DIR):
    """6-panel figure: metrics over episodes + best-model policy curve."""
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    eps = summary_df.index.values

    axes[0, 0].plot(eps, summary_df["choice_agreement"], "b-o", markersize=3)
    axes[0, 0].axvline(best_ep, color="red", ls="--", alpha=0.6, label=f"best={best_ep}")
    axes[0, 0].set_title("choice agreement")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].set_ylabel("fraction")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(eps, summary_df["nll"], "g-o", markersize=3)
    axes[0, 1].axvline(best_ep, color="red", ls="--", alpha=0.6)
    axes[0, 1].set_title("NLL of human choices")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].set_ylabel("NLL (lower = better)")

    axes[0, 2].plot(eps, summary_df["policy_shape_r"], "m-o", markersize=3)
    axes[0, 2].axvline(best_ep, color="red", ls="--", alpha=0.6)
    axes[0, 2].set_title("policy shape correlation")
    axes[0, 2].set_xlabel("episode")
    axes[0, 2].set_ylabel("pearson r")

    axes[1, 0].plot(eps, summary_df["logit_dq_r2"], "c-o", markersize=3)
    axes[1, 0].axvline(best_ep, color="red", ls="--", alpha=0.6)
    axes[1, 0].set_title("logit_diff ~ dQ r²")
    axes[1, 0].set_xlabel("episode")
    axes[1, 0].set_ylabel("r²")

    if "composite" in summary_df.columns:
        axes[1, 1].plot(eps, summary_df["composite"], "k-o", markersize=3)
        axes[1, 1].axvline(best_ep, color="red", ls="--", alpha=0.6)
        axes[1, 1].set_title("composite score (z-avg)")
        axes[1, 1].set_xlabel("episode")
        axes[1, 1].set_ylabel("z-score")

    # best model policy curve vs human
    agent = RLAgent(hidden_size=128)
    best_path = os.path.join(checkpoint_dir, f"ep{best_ep:03d}.pt")
    agent.load_checkpoint(best_path)
    replay_df = replay_human_trials(agent, human_df)

    n_bins = 8
    replay_df["dQ_bin"] = pd.qcut(replay_df["human_dQ"], q=n_bins, duplicates="drop")
    binned = replay_df.groupby("dQ_bin", observed=True).agg(
        human_p_left=("human_choice", lambda x: 1.0 - x.mean()),
        model_p_left=("model_p_left", "mean"),
        dQ_mean=("human_dQ", "mean"),
    )
    axes[1, 2].plot(binned["dQ_mean"], binned["human_p_left"], "ko-", label="human")
    axes[1, 2].plot(binned["dQ_mean"], binned["model_p_left"], "rs--", label=f"model (ep {best_ep})")
    axes[1, 2].axhline(0.5, ls=":", c="gray")
    axes[1, 2].axvline(0.0, ls=":", c="gray")
    axes[1, 2].set_xlabel("dQ (left - right)")
    axes[1, 2].set_ylabel("P(choose left)")
    axes[1, 2].set_title("policy curve: best model vs human")
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "model_selection.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved figure to {out_path}")


def main():
    print("=== human-similarity evaluation pipeline ===\n")

    human_df = load_human_data()
    print(f"loaded {len(human_df)} human trials, {human_df['sub_id'].nunique()} subjects\n")

    summary = evaluate_all_checkpoints(human_df)

    best_ep, summary_with_composite = select_best(summary)
    print(f"\n>>> best episode: {best_ep}")
    print(f"    metrics: {summary.loc[best_ep].to_dict()}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "checkpoint_metrics.csv")
    summary_with_composite.to_csv(csv_path)
    print(f"saved metrics to {csv_path}")

    # save best-model replay for downstream analysis
    agent = RLAgent(hidden_size=128)
    best_path = os.path.join(CHECKPOINT_DIR, f"ep{best_ep:03d}.pt")
    agent.load_checkpoint(best_path)
    replay_df = replay_human_trials(agent, human_df)
    replay_path = os.path.join(OUT_DIR, "best_model_replay.csv")
    replay_df.to_csv(replay_path, index=False)
    print(f"saved best-model replay ({len(replay_df)} trials) to {replay_path}")

    plot_results(summary_with_composite, best_ep, human_df)


if __name__ == "__main__":
    main()
