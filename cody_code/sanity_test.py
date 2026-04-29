"""
Sanity test: can the model exploit obvious reward differences?

Loads the latest checkpoint and runs it on a condensed task with controlled
probability gaps. Same experimental structure (3 identities per block,
holdout, identity persistence) but with fixed probability assignments:

  - EASY blocks: one identity at 0.95, one at 0.05, one at 0.50
  - HARD blocks: all three identities at 0.50

If the model learned anything about value tracking, it should be above
chance on EASY blocks. HARD blocks should be near chance (correct).

Uses the current RL²-input architecture: every step gets
(left_emb, right_emb, last_chosen_emb, last_action, last_reward) as input.
Legacy (pre-RL², pre-bilinear) checkpoints have a different LSTM input
dim and are not loadable here.
"""

import os
import sys
import glob
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RL_persist import RLAgent, BanditTask, TRIALS_PER_BLOCK, CHECKPOINT_DIR


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
        return np.array([0.5, 0.5, 0.5])

    def start_block(self, block_idx: int):
        super().start_block(block_idx)
        if block_idx % 2 == 0:
            probs = self.EASY_PROBS.copy()
            random.shuffle(probs)
        else:
            probs = self.HARD_PROBS.copy()
        for i, b in enumerate(self.current_bandits):
            b.true_prob = probs[i]


@torch.no_grad()
def run_sanity(agent, task, num_episodes=20):
    agent.policy_network.eval()

    easy_correct, easy_total = 0, 0
    hard_correct, hard_total = 0, 0
    easy_by_trial = {t: {"correct": 0, "total": 0} for t in range(task.trials_per_block)}

    for ep in range(num_episodes):
        task.reset_episode()
        agent.randomize_embeddings()

        for block in range(task.n_blocks):
            task.start_block(block)
            is_easy = (block % 2 == 0)
            hidden = None
            last_chosen_id = None
            last_action = 0.0
            last_reward = 0.0

            for trial in range(task.trials_per_block):
                left_bandit, right_bandit = task.sample_two_bandits(trial)
                left_id, right_id = left_bandit.stim_id, right_bandit.stim_id
                info = {"block": block, "trial": trial}

                id_feats = agent._embed_triple(left_id, right_id, last_chosen_id)

                # stim
                x_stim = agent._make_step_input(id_feats, agent.state_encoder.stim(info, last_action, last_reward))
                _, _, hidden = agent.policy_network(x_stim, hidden)

                # decision
                x_dec = agent._make_step_input(id_feats, agent.state_encoder.decision(info, last_action, last_reward))
                lstm_out, _, hidden = agent.policy_network(x_dec, hidden)
                context = lstm_out[:, -1, :]

                emb_l = agent.id_embedding(torch.tensor([left_id], device=agent.device, dtype=torch.long))
                emb_r = agent.id_embedding(torch.tensor([right_id], device=agent.device, dtype=torch.long))
                logits = agent.policy_network.get_policy_logits(context, emb_l, emb_r)

                chosen_side = int(logits.argmax(dim=-1).item())
                chosen = left_bandit if chosen_side == 0 else right_bandit
                unchosen = right_bandit if chosen_side == 0 else left_bandit
                chosen_id = chosen.stim_id

                correct = 1 if chosen.true_prob >= unchosen.true_prob else 0
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

                # feedback (RL² update of last_*)
                r = chosen.sample_reward()
                last_chosen_id = chosen_id
                last_action = float(chosen_side)
                last_reward = float(r)
                id_feats_fb = agent._embed_triple(left_id, right_id, last_chosen_id)
                x_fb = agent._make_step_input(id_feats_fb, agent.state_encoder.feedback(r, info, last_action, last_reward))
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


def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "ep*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"no checkpoints in {checkpoint_dir}")
    return ckpt_files[-1]


def main():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    ckpt_path = find_latest_checkpoint()
    print(f"loading checkpoint: {ckpt_path}")

    agent = RLAgent(hidden_size=128)
    ep = agent.load_checkpoint(ckpt_path)
    print(f"loaded episode {ep} (bilinear + RL² inputs)")

    task = SanityTask(n_blocks=10, trials_per_block=TRIALS_PER_BLOCK, stim_pool_size=200)
    run_sanity(agent, task, num_episodes=20)


if __name__ == "__main__":
    main()
