"""
Sanity test: can the model exploit obvious reward differences?

Loads the best checkpoint and runs it on a condensed task with controlled
probability gaps. Same experimental structure (3 identities per block,
holdout, identity persistence) but with fixed probability assignments:

  - EASY blocks: one identity at 0.95, one at 0.05, one at 0.50
  - HARD blocks: all three identities at 0.50

If the model learned *anything* about value tracking, it should be
above chance on EASY blocks. HARD blocks should be near chance (that's correct).

Checkpoint compatibility: supports both the legacy shared-scorer architecture
(ph_fc1/ph_fc2 heads that take [context, embedding]) and the current bilinear
architecture (policy_proj). Detected by inspecting state_dict keys.
"""

import os
import sys
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RL_persist import RLAgent, Bandit, BanditTask, TRIALS_PER_BLOCK, CHECKPOINT_DIR


class SanityTask(BanditTask):
    """
    Same structure as BanditTask but with controlled reward probabilities.
    Alternates EASY and HARD blocks.
    """

    EASY_PROBS = [0.95, 0.05, 0.50]
    HARD_PROBS = [0.50, 0.50, 0.50]

    def __init__(self, n_blocks=10, trials_per_block=TRIALS_PER_BLOCK, stim_pool_size=200):
        super().__init__(n_blocks=n_blocks, trials_per_block=trials_per_block, stim_pool_size=stim_pool_size)

    def _sample_reward_probs(self) -> np.ndarray:
        # overridden by start_block below
        return np.array([0.5, 0.5, 0.5])

    def start_block(self, block_idx: int):
        # use parent logic for identity management (persistence, holdout, novelty)
        super().start_block(block_idx)

        # override probabilities: even blocks = EASY, odd blocks = HARD
        if block_idx % 2 == 0:
            probs = self.EASY_PROBS.copy()
            random.shuffle(probs)
        else:
            probs = self.HARD_PROBS.copy()

        for i, b in enumerate(self.current_bandits):
            b.true_prob = probs[i]


# Legacy head compatibility

class LegacyScorer(nn.Module):
    """
    Reimplements the pre-bilinear shared scorer: [context, embedding] -> scalar.
    Loaded directly from checkpoint ph_fc1/ph_fc2 weights when present.
    """

    def __init__(self, hidden_size=128, id_emb_dim=16, ph_hidden=64):
        super().__init__()
        self.ph_fc1 = nn.Linear(hidden_size + id_emb_dim, ph_hidden)
        self.ph_fc2 = nn.Linear(ph_hidden, 1)

    def score(self, context, emb):
        x = torch.cat([context, emb], dim=-1)
        return self.ph_fc2(F.relu(self.ph_fc1(x))).squeeze(-1)

    def logits(self, context, emb_left, emb_right):
        sl = self.score(context, emb_left)
        sr = self.score(context, emb_right)
        return torch.stack([sl, sr], dim=-1)


def load_checkpoint_any(agent, path):
    """
    Load a checkpoint that may be either the current bilinear architecture
    (policy_proj/aux_proj) or the legacy shared-scorer (ph_fc*/rp_fc*).

    Returns (episode, legacy_scorer_or_None). When a legacy scorer is
    returned, callers should use it instead of agent.policy_network.get_policy_logits.
    """
    ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    sd = ckpt["policy_network"]

    is_legacy = any(k.startswith("ph_fc") for k in sd.keys())

    if is_legacy:
        # load only the LSTM weights into the current network; leave policy_proj/aux_proj alone
        lstm_sd = {k: v for k, v in sd.items() if k.startswith("lstm.")}
        missing, unexpected = agent.policy_network.load_state_dict(lstm_sd, strict=False)
        # missing will include policy_proj/aux_proj/vh_* which we don't care about here

        # instantiate the legacy scorer and load its weights
        hidden_size = agent.hidden_size
        id_emb_dim = agent.id_emb_dim
        ph_hidden = sd["ph_fc1.weight"].shape[0]
        legacy = LegacyScorer(hidden_size, id_emb_dim, ph_hidden).to(agent.device)
        legacy.load_state_dict({
            "ph_fc1.weight": sd["ph_fc1.weight"],
            "ph_fc1.bias": sd["ph_fc1.bias"],
            "ph_fc2.weight": sd["ph_fc2.weight"],
            "ph_fc2.bias": sd["ph_fc2.bias"],
        })
        legacy.eval()
        return int(ckpt.get("episode", 0)), legacy
    else:
        agent.policy_network.load_state_dict(sd)
        return int(ckpt.get("episode", 0)), None


def compute_logits(agent, legacy, context, left_id, right_id):
    emb_l = agent.id_embedding(torch.tensor([left_id], device=agent.device, dtype=torch.long))
    emb_r = agent.id_embedding(torch.tensor([right_id], device=agent.device, dtype=torch.long))
    if legacy is not None:
        return legacy.logits(context, emb_l, emb_r)
    return agent.policy_network.get_policy_logits(context, emb_l, emb_r)


@torch.no_grad()
def run_sanity(agent, legacy, task, num_episodes=20):
    agent.policy_network.eval()

    easy_correct, easy_total = 0, 0
    hard_correct, hard_total = 0, 0

    # per-trial tracking for easy blocks: does accuracy improve within a block?
    easy_by_trial = {t: {"correct": 0, "total": 0} for t in range(task.trials_per_block)}

    for ep in range(num_episodes):
        task.reset_episode()
        agent.randomize_embeddings()

        for block in range(task.n_blocks):
            task.start_block(block)
            is_easy = (block % 2 == 0)
            hidden = None

            for trial in range(task.trials_per_block):
                left_bandit, right_bandit = task.sample_two_bandits(trial)
                left_id, right_id = left_bandit.stim_id, right_bandit.stim_id
                info = {"block": block, "trial": trial}

                id_feats = agent._embed_pair(left_id, right_id)

                # stim
                x_stim = agent._make_step_input(id_feats, agent.state_encoder.stim(info))
                _, _, hidden = agent.policy_network(x_stim, hidden)

                # decision
                x_dec = agent._make_step_input(id_feats, agent.state_encoder.decision(info))
                lstm_out, _, hidden = agent.policy_network(x_dec, hidden)
                context = lstm_out[:, -1, :]

                logits = compute_logits(agent, legacy, context, left_id, right_id)

                chosen_side = int(logits.argmax(dim=-1).item())
                chosen = left_bandit if chosen_side == 0 else right_bandit
                unchosen = right_bandit if chosen_side == 0 else left_bandit
                chosen_id = chosen.stim_id

                # was this the better option?
                correct = 1 if chosen.true_prob >= unchosen.true_prob else 0
                # if equal probs, count as correct
                if abs(chosen.true_prob - unchosen.true_prob) < 0.01:
                    correct = 1

                if is_easy:
                    easy_correct += correct
                    easy_total += 1
                    easy_by_trial[trial]["correct"] += correct
                    easy_by_trial[trial]["total"] += 1
                else:
                    hard_correct += correct
                    hard_total += 1

                # feedback
                r = chosen.sample_reward()
                id_feats_fb = agent._embed_feedback(chosen_id, chosen_side)
                x_fb = agent._make_step_input(id_feats_fb, agent.state_encoder.feedback(r, info))
                _, _, hidden = agent.policy_network(x_fb, hidden)


    print("SANITY TEST RESULTS")
    print(f"\nEASY blocks (0.95 vs 0.05 vs 0.50):")
    print(f"  accuracy: {easy_correct}/{easy_total} = {easy_correct/easy_total:.3f}")
    print(f"  (chance = 0.50)")

    print(f"\nHARD blocks (0.50 vs 0.50 vs 0.50):")
    print(f"  accuracy: {hard_correct}/{hard_total} = {hard_correct/hard_total:.3f}")
    print(f"  (chance = 1.00 — all options equal)")

    print(f"\nEASY accuracy by trial position (does it improve within block?):")
    for t in range(task.trials_per_block):
        bt = easy_by_trial[t]
        acc = bt["correct"] / bt["total"] if bt["total"] > 0 else float("nan")
        bar = "#" * int(acc * 40)
        print(f"  trial {t:2d}: {acc:.3f} |{bar}")

    print("=" * 60)


def find_best_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "ep*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"no checkpoints in {checkpoint_dir}")
    # use the latest checkpoint
    return ckpt_files[-1]


def main():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    ckpt_path = find_best_checkpoint()
    print(f"loading checkpoint: {ckpt_path}")

    agent = RLAgent(hidden_size=128)
    ep, legacy = load_checkpoint_any(agent, ckpt_path)
    arch = "legacy shared-scorer" if legacy is not None else "bilinear"
    print(f"loaded episode {ep} ({arch})")

    task = SanityTask(n_blocks=10, trials_per_block=TRIALS_PER_BLOCK, stim_pool_size=200)
    run_sanity(agent, legacy, task, num_episodes=20)


if __name__ == "__main__":
    main()
