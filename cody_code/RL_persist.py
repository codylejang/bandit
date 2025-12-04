import numpy as np
import pandas as pd
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed = 408


class Bandit:
    def __init__(self, stim_id: int, true_prob: float):
        self.stim_id = int(stim_id)
        self.true_prob = float(true_prob)

    def sample_reward(self) -> float:
        return float(np.random.rand() < self.true_prob)


class BanditTask:
    """
    aquino-style identity-based task.

    per episode:
      - a pool of unique stimulus identities (like paintings)
      - each block has 3 identities
      - block 0: 3 new identities
      - later blocks: keep 2 identities, drop 1, add 1 new identity
      - within each block: one identity is held out until a random trial threshold
      - each block re-samples reward probabilities for its 3 identities
    """

    def __init__(self, n_blocks=30, trials_per_block=15, stim_pool_size=200):
        self.n_blocks = int(n_blocks)
        self.trials_per_block = int(trials_per_block)
        self.stim_pool_size = int(stim_pool_size)

        self.stim_pool = None
        self.used_stims = None
        self.current_bandits = None  # list[bandit]
        self.heldout_idx = None
        self.heldout_start_trial = None

        self.reset_episode()

    def reset_episode(self):
        self.stim_pool = list(range(self.stim_pool_size))
        random.shuffle(self.stim_pool)
        self.used_stims = set()
        self.current_bandits = None
        self.heldout_idx = None
        self.heldout_start_trial = None

    def _get_new_stim_id(self) -> int:
        if not self.stim_pool:
            raise RuntimeError("ran out of stimulus ids. increase stim_pool_size or reset_episode().")
        sid = self.stim_pool.pop()
        self.used_stims.add(sid)
        return sid

    def _sample_reward_probs(self) -> np.ndarray:
        # uniform in [0.2, 0.8]
        return np.round(np.random.uniform(0.2, 0.8, size=3), 2)

    def start_block(self, block_idx: int):
        if block_idx == 0:
            ids = [self._get_new_stim_id() for _ in range(3)]
        else:
            if self.current_bandits is None:
                raise RuntimeError("call start_block(0) first in an episode.")
            prev_ids = [b.stim_id for b in self.current_bandits]
            keep_two = random.sample(prev_ids, 2)
            new_id = self._get_new_stim_id()
            ids = keep_two + [new_id]

        probs = self._sample_reward_probs()
        self.current_bandits = [Bandit(stim_id=ids[i], true_prob=probs[i]) for i in range(3)]

        # held-out option logic:
        # - heldout_idx selects which of the 3 is withheld early in the block
        # - heldout_start_trial is inclusive threshold:
        #   if heldout_start_trial == trials_per_block, held-out never appears this block
        self.heldout_idx = random.randint(0, 2)
        lo = min(7, self.trials_per_block)
        hi = self.trials_per_block
        self.heldout_start_trial = random.randint(lo, hi)

    def sample_two_bandits(self, trial_idx: int):
        if self.current_bandits is None:
            raise RuntimeError("call start_block(block_idx) before sampling trials.")

        if trial_idx < self.heldout_start_trial:
            idxs = [0, 1, 2]
            idxs.remove(self.heldout_idx)
            chosen = random.sample(idxs, 2)
        else:
            chosen = random.sample([0, 1, 2], 2)

        left_idx, right_idx = chosen
        return self.current_bandits[left_idx], self.current_bandits[right_idx]

    def summarize(self):
        print("bandit task ready (identity-based, held-out option, cross-block novelty)")


class StateEncoder(nn.Module):
    """
    minimal step encoding, no one-hot placeholders.

    output dim = 5:
      - 1 trial fraction
      - 3 step flags: [stim, decision, feedback]
      - 1 reward scalar (only nonzero at feedback)
    """

    def __init__(self, norm_trials=15):
        super().__init__()
        self.norm_trials = float(norm_trials)

    @property
    def out_dim(self) -> int:
        return 5

    def _trial(self, info) -> torch.Tensor:
        t = float(info["trial"]) / self.norm_trials
        return torch.tensor([t], dtype=torch.float32)

    def stim(self, info) -> torch.Tensor:
        return torch.cat([self._trial(info), torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)])

    def decision(self, info) -> torch.Tensor:
        return torch.cat([self._trial(info), torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)])

    def feedback(self, reward, info) -> torch.Tensor:
        return torch.cat([self._trial(info), torch.tensor([0.0, 0.0, 1.0, float(reward)], dtype=torch.float32)])


class LSTMPolicyNetwork(nn.Module):
    # lstm backbone with policy (2 actions: left/right) + value head
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.0, num_actions=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_actions),  # left vs right
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)  # out: (b,s,h)
        logits = self.policy_head(out)      # (b,s,2)
        values = self.value_head(out)       # (b,s,1)
        return logits, values, hidden


class BlockBeta:
    # block-local beta posterior for reward uncertainty (analysis only)
    def __init__(self):
        self.a = 1.0
        self.b = 1.0

    def update(self, r: float):
        if r > 0.5:
            self.a += 1.0
        else:
            self.b += 1.0

    @property
    def mean(self):
        return self.a / (self.a + self.b)

    @property
    def var(self):
        ab = self.a + self.b
        return (self.a * self.b) / ((ab ** 2) * (ab + 1.0))


def beta_var(alpha: float, beta: float) -> float:
    # var[beta(alpha, beta)]
    ab = alpha + beta
    return float((alpha * beta) / ((ab * ab) * (ab + 1.0)))


class RLAgent:
    def __init__(
        self,
        hidden_size=128,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.002,
        value_coef=0.5,
        norm_trials=15,
        num_stimuli=200,
        id_emb_dim=16,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = int(hidden_size)
        self.num_stimuli = int(num_stimuli)
        self.id_emb_dim = int(id_emb_dim)

        self.state_encoder = StateEncoder(norm_trials=norm_trials)

        # identity embeddings for left/right options and chosen option at feedback
        # learned lookup table: maps stim_id to embedding vector
        self.id_embedding = nn.Embedding(self.num_stimuli, self.id_emb_dim).to(self.device)

        # total input dim = left_emb + right_emb + base_state
        lstm_input_size = (2 * self.id_emb_dim) + self.state_encoder.out_dim
        self.policy_network = LSTMPolicyNetwork(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.0,
            num_actions=2,
        ).to(self.device)

        # optimize both lstm weights and embedding table
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.id_embedding.parameters()),
            lr=lr,
        )

        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)

        self.training_stats = {
            "rewards": [],
            "losses": [],
            "entropies": [],
            "probe_rows": [],
        }

    """
    input sequence construction (what the lstm receives)

    - each trial: three sequential steps: stimulus,  decision,  feedback
    - at each step, the lstm input is a concatenation of:
        - identity features (left/right embeddings, shape (1, 2*id_emb_dim))
        - a small step/state vector (trial fraction + step flags + reward slot, shape (base_dim,))
    - stimulus step:
        - identity features encode the offered pair (left_id, right_id)
        - base state marks "stimulus" and reward=0
    - decision step:
        - identity features encode the same offered pair (left_id, right_id)
        - base state marks "decision" and reward=0
    - feedback step:
        - identity features encode the chosen identity only (chosen_id embedded in the left slot, right slot zeroed)
        - base state marks "feedback" and reward is the observed outcome
    - why feedback is encoded this way:
        - it tells the lstm which identity the reward should be credited to 
        - no action one-hots like prev iteration
    """
    def _embed_pair(self, left_id: int, right_id: int) -> torch.Tensor:
        left_t = torch.tensor([left_id], device=self.device, dtype=torch.long)
        right_t = torch.tensor([right_id], device=self.device, dtype=torch.long)
        e_left = self.id_embedding(left_t)    # (1, emb)
        e_right = self.id_embedding(right_t)  # (1, emb)
        return torch.cat([e_left, e_right], dim=-1)  # (1, 2*emb)

    def _embed_feedback(self, chosen_id: int) -> torch.Tensor:
        # feedback only tells the rnn which identity produced the reward
        chosen_t = torch.tensor([chosen_id], device=self.device, dtype=torch.long)
        e_chosen = self.id_embedding(chosen_t)      # (1, emb)
        e_zero = torch.zeros_like(e_chosen)         # (1, emb)
        return torch.cat([e_chosen, e_zero], dim=-1)  # (1, 2*emb)

    def _make_step_input(self, id_feats: torch.Tensor, base_state: torch.Tensor) -> torch.Tensor:
        # id_feats: (1, 2*emb), base_state: (base_dim,)
        base_state = base_state.to(self.device).unsqueeze(0)        # (1, base_dim)
        x = torch.cat([id_feats, base_state], dim=-1).unsqueeze(0)  # (1, 1, 2*emb+base_dim)
        return x

    def train_episode(self, task: BanditTask, render=False):
        self.policy_network.train()
        self.id_embedding.train()

        task.reset_episode()

        # session-wide exposure counts for novelty regressor
        # seen_count[sid] = number of times sid has been offered so far in this episode
        seen_count = defaultdict(int)

        episode_reward = 0.0

        for block in range(task.n_blocks):
            task.start_block(block)
            if render:
                bdesc = [f"id{b.stim_id}:{b.true_prob}" for b in task.current_bandits]
                print(f"block {block+1}: {bdesc} | heldout={task.heldout_idx} start={task.heldout_start_trial}")

            hidden = None

            # block-local beta trackers keyed by stim_id
            beta_map = {b.stim_id: BlockBeta() for b in task.current_bandits}

            saved_logps, saved_values, saved_rewards, saved_entropies = [], [], [], []

            for trial in range(task.trials_per_block):
                left_bandit, right_bandit = task.sample_two_bandits(trial)
                left_id, right_id = left_bandit.stim_id, right_bandit.stim_id
                info = {"block": block, "trial": trial}

                # novelty regressor:
                # k is "number of prior exposures + 1" so first-ever presentation has k=1
                kL = seen_count[left_id] + 1
                kR = seen_count[right_id] + 1
                novL = beta_var(float(kL), 1.0)
                novR = beta_var(float(kR), 1.0)
                dNov = float(novL - novR)

                # increment exposures as soon as they are offered on screen
                seen_count[left_id] += 1
                seen_count[right_id] += 1

                # identity features for stim/decision steps (left/right)
                id_feats = self._embed_pair(left_id, right_id)

                # 1) stimulus step
                x_stim = self._make_step_input(id_feats, self.state_encoder.stim(info))
                _, _, hidden = self.policy_network(x_stim, hidden)
                h_stim = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                # 2) decision step
                x_dec = self._make_step_input(id_feats, self.state_encoder.decision(info))
                logits, values, hidden = self.policy_network(x_dec, hidden)

                h_dec = hidden[0][-1].squeeze(0).detach().cpu().numpy()
                v_dec = values[:, -1, 0].detach().cpu().item()

                # policy is always 2-way: choose left (0) vs right (1)
                dec_logits = logits[:, -1, :]                 # (1,2)
                log_probs = F.log_softmax(dec_logits, dim=-1) # (1,2)
                probs = log_probs.exp()
                dist = torch.distributions.Categorical(probs)

                action = dist.sample()                        # 0=left, 1=right
                logp = dist.log_prob(action)
                entropy = dist.entropy()

                chosen_side = int(action.item())
                chosen = left_bandit if chosen_side == 0 else right_bandit
                chosen_id = chosen.stim_id

                # analysis-only beta regressors (within block)
                QL, QR = beta_map[left_id].mean, beta_map[right_id].mean
                UL, UR = beta_map[left_id].var, beta_map[right_id].var
                dQ = float(QL - QR)
                dUnc = float(UL - UR)
                logit_diff = float((dec_logits[0, 0] - dec_logits[0, 1]).detach().cpu().item())

                # 3) feedback step
                r = chosen.sample_reward()
                episode_reward += r

                # update block beta for chosen identity
                beta_map[chosen_id].update(r)

                # feed back chosen identity + reward into memory
                id_feats_fb = self._embed_feedback(chosen_id)
                x_fb = self._make_step_input(id_feats_fb, self.state_encoder.feedback(r, info))
                _, _, hidden = self.policy_network(x_fb, hidden)
                h_fb = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                # store decision-step tensors for block update
                saved_logps.append(logp.squeeze(0))
                saved_values.append(values[:, -1, 0].squeeze(0))
                saved_rewards.append(torch.tensor(float(r), device=self.device))
                saved_entropies.append(entropy.squeeze(0))

                self.training_stats["probe_rows"].append(
                    {
                        "episode_idx": len(self.training_stats["rewards"]),
                        "block": block,
                        "trial": trial,
                        "left_id": int(left_id),
                        "right_id": int(right_id),
                        "choice_side": chosen_side,
                        "chosen_id": int(chosen_id),
                        "reward": float(r),
                        "QL": float(QL),
                        "QR": float(QR),
                        "UncL": float(UL),
                        "UncR": float(UR),
                        "dQ": dQ,
                        "dUnc": dUnc,
                        "novL": float(novL),
                        "novR": float(novR),
                        "dNov": dNov,
                        "seenL_before": int(kL - 1),
                        "seenR_before": int(kR - 1),
                        "logit_diff": logit_diff,
                        "value_dec": float(v_dec),
                        "h_stim": h_stim.tolist(),
                        "h_dec": h_dec.tolist(),
                        "h_fb": h_fb.tolist(),
                    }
                )

            # update once per block over all decision steps
            if saved_logps:
                logp_t = torch.stack(saved_logps)           # (T,)
                value_t = torch.stack(saved_values)         # (T,)
                reward_t = torch.stack(saved_rewards)       # (T,)
                entropy_t = torch.stack(saved_entropies).mean()

                # contextual bandit: return is immediate reward
                returns = reward_t

                # advantage and normalization for stability
                advantages = returns - value_t.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))

                policy_loss = -(logp_t * advantages).mean()
                value_loss = F.mse_loss(value_t, returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_t

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters()) + list(self.id_embedding.parameters()),
                    1.0,
                )
                self.optimizer.step()

                self.training_stats["losses"].append(loss.item())
                self.training_stats["entropies"].append(entropy_t.item())

        self.training_stats["rewards"].append(episode_reward)
        return episode_reward, None

    def evaluate(self, task: BanditTask, num_episodes=10):
        self.policy_network.eval()
        self.id_embedding.eval()

        total_rewards = []

        with torch.no_grad():
            for _ in range(num_episodes):
                task.reset_episode()
                episode_reward = 0.0

                for block in range(task.n_blocks):
                    task.start_block(block)
                    hidden = None

                    for trial in range(task.trials_per_block):
                        left_bandit, right_bandit = task.sample_two_bandits(trial)
                        left_id, right_id = left_bandit.stim_id, right_bandit.stim_id
                        info = {"block": block, "trial": trial}

                        id_feats = self._embed_pair(left_id, right_id)

                        # stimulus step
                        x_stim = self._make_step_input(id_feats, self.state_encoder.stim(info))
                        _, _, hidden = self.policy_network(x_stim, hidden)

                        # decision step (greedy)
                        x_dec = self._make_step_input(id_feats, self.state_encoder.decision(info))
                        logits, _, hidden = self.policy_network(x_dec, hidden)
                        dec_logits = logits[:, -1, :]  # (1,2)

                        action = int(dec_logits.argmax(dim=-1).item())
                        chosen = left_bandit if action == 0 else right_bandit
                        chosen_id = chosen.stim_id

                        r = chosen.sample_reward()
                        episode_reward += r

                        # feedback step
                        id_feats_fb = self._embed_feedback(chosen_id)
                        x_fb = self._make_step_input(id_feats_fb, self.state_encoder.feedback(r, info))
                        _, _, hidden = self.policy_network(x_fb, hidden)

                total_rewards.append(episode_reward)

        return float(np.mean(total_rewards)), float(np.std(total_rewards))

    def build_probe_dataframe(self):
        if len(self.training_stats.get("probe_rows", [])) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(self.training_stats["probe_rows"])

        # flatten hidden vectors into columns for easy regression
        for col in ["h_stim", "h_dec", "h_fb"]:
            if col in df.columns:
                H = np.stack(df[col].to_numpy())
                H_cols = {f"{col}_{i}": H[:, i] for i in range(H.shape[1])}
                df = pd.concat([df.drop(columns=[col]), pd.DataFrame(H_cols)], axis=1)

        return df

    def plot_training_progress(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(self.training_stats["rewards"])
        axes[0, 0].set_title("episode rewards")
        axes[0, 0].set_xlabel("episode")
        axes[0, 0].set_ylabel("total reward")

        axes[0, 1].plot(self.training_stats["losses"])
        axes[0, 1].set_title("training loss")
        axes[0, 1].set_xlabel("update")
        axes[0, 1].set_ylabel("loss")

        axes[1, 0].plot(self.training_stats["entropies"])
        axes[1, 0].set_title("policy entropy")
        axes[1, 0].set_xlabel("update")
        axes[1, 0].set_ylabel("entropy")

        if len(self.training_stats["rewards"]) > 10:
            window = min(50, len(self.training_stats["rewards"]) // 4)
            moving_avg = pd.Series(self.training_stats["rewards"]).rolling(window=window).mean()
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title(f"moving avg rewards (window={window})")
            axes[1, 1].set_xlabel("episode")
            axes[1, 1].set_ylabel("avg reward")

        plt.tight_layout()
        plt.savefig("training.png")
        plt.show()


def main():
    print("start rl agent for 3-step identity-based bandit...")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    task = BanditTask(n_blocks=30, trials_per_block=15, stim_pool_size=200)

    agent = RLAgent(
        hidden_size=128,
        lr=3e-4,
        norm_trials=task.trials_per_block,
        num_stimuli=task.stim_pool_size,
        id_emb_dim=16,
    )

    num_episodes = 100
    eval_interval = 10

    print(f"training for {num_episodes} episodes...")
    for ep in range(num_episodes):
        reward, _ = agent.train_episode(task, render=(ep % 20 == 0))

        if ep % eval_interval == 0:
            eval_mean, eval_std = agent.evaluate(task, num_episodes=5)
            print(f"episode {ep:3d} | train reward: {reward:6.1f} | eval: {eval_mean:6.1f} ± {eval_std:4.1f}")

            agent.policy_network.train()
            agent.id_embedding.train()

    print("\nfinal evaluation:")
    mean_r, std_r = agent.evaluate(task, num_episodes=20)
    print(f"final performance: {mean_r:.2f} ± {std_r:.2f}")

    probe_df = agent.build_probe_dataframe()
    print(f"probe df shape: {probe_df.shape}")
    probe_df.to_csv("probe_rows.csv", index=False)

    agent.plot_training_progress()
    return agent


if __name__ == "__main__":
    trained_agent = main()
