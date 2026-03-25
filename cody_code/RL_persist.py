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
TRIALS_PER_BLOCK = 60
NUM_EPISODES = 500
EVAL_INTERVAL = 5

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

    def __init__(self, n_blocks=30, trials_per_block=TRIALS_PER_BLOCK, stim_pool_size=200):
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
    """
    lstm backbone with policy (2 actions: left/right) + value head

    requested probes:
      - capture the vector that feeds the heads (lstm output)
      - capture head hidden activations (pre/post relu) during greedy or training
      - capture logits/value outputs
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.0, num_actions=2):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.head_hidden = self.hidden_size // 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # explicit heads so we can probe intermediates
        self.ph_fc1 = nn.Linear(self.hidden_size, self.head_hidden)
        self.ph_relu = nn.ReLU()
        self.ph_drop = nn.Dropout(dropout)
        self.ph_fc2 = nn.Linear(self.head_hidden, num_actions)

        self.vh_fc1 = nn.Linear(self.hidden_size, self.head_hidden)
        self.vh_relu = nn.ReLU()
        self.vh_drop = nn.Dropout(dropout)
        self.vh_fc2 = nn.Linear(self.head_hidden, 1)

    def forward(self, x, hidden=None, return_activations=False):
        out, hidden = self.lstm(x, hidden)  # out: (b,s,h)

        # policy head with optional intermediates
        ph_pre = self.ph_fc1(out)           # (b,s,hh)
        ph_post = self.ph_relu(ph_pre)      # (b,s,hh)
        ph_post = self.ph_drop(ph_post)
        logits = self.ph_fc2(ph_post)       # (b,s,2)

        # value head with optional intermediates
        vh_pre = self.vh_fc1(out)           # (b,s,hh)
        vh_post = self.vh_relu(vh_pre)      # (b,s,hh)
        vh_post = self.vh_drop(vh_post)
        values = self.vh_fc2(vh_post)       # (b,s,1)

        if not return_activations:
            return logits, values, hidden

        # note: out is the true "hidden activation feeding into the heads"
        activations = {
            "lstm_out": out,
            "ph_pre": ph_pre,
            "ph_post": ph_post,
            "vh_pre": vh_pre,
            "vh_post": vh_post,
        }
        return logits, values, hidden, activations


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
        entropy_coef=0.01,
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
            "probe_rows": [],        # training (sampled) probes
            "greedy_probe_rows": [], # greedy probes (eval)
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
        - identity features encode the chosen identity in its actual position
          (chosen_side=0 → left slot, chosen_side=1 → right slot; other slot zeroed)
        - base state marks "feedback" and reward is the observed outcome

    probe additions:
      - record lstm_out feeding heads at decision
      - record head hidden activations (pre/post relu) at decision
      - do this in training probes and in greedy probes
    """

    def _embed_pair(self, left_id: int, right_id: int) -> torch.Tensor:
        left_t = torch.tensor([left_id], device=self.device, dtype=torch.long)
        right_t = torch.tensor([right_id], device=self.device, dtype=torch.long)
        e_left = self.id_embedding(left_t)    # (1, emb)
        e_right = self.id_embedding(right_t)  # (1, emb)
        return torch.cat([e_left, e_right], dim=-1)  # (1, 2*emb)

    def _embed_feedback(self, chosen_id: int, chosen_side: int) -> torch.Tensor:
        # feedback tells the rnn which identity produced the reward,
        # placed in its actual position to avoid left-slot bias
        chosen_t = torch.tensor([chosen_id], device=self.device, dtype=torch.long)
        e_chosen = self.id_embedding(chosen_t)      # (1, emb)
        e_zero = torch.zeros_like(e_chosen)         # (1, emb)
        if chosen_side == 0:  # chose left
            return torch.cat([e_chosen, e_zero], dim=-1)
        else:  # chose right
            return torch.cat([e_zero, e_chosen], dim=-1)

    def _make_step_input(self, id_feats: torch.Tensor, base_state: torch.Tensor) -> torch.Tensor:
        # id_feats: (1, 2*emb), base_state: (base_dim,)
        base_state = base_state.to(self.device).unsqueeze(0)        # (1, base_dim)
        x = torch.cat([id_feats, base_state], dim=-1).unsqueeze(0)  # (1, 1, 2*emb+base_dim)
        return x

    def _tensor_to_list(self, t: torch.Tensor) -> list:
        # safe conversion for csv logging
        return t.detach().cpu().float().view(-1).tolist()

    def train_episode(self, task: BanditTask, render=False, capture_probes=True):
        self.policy_network.train()
        self.id_embedding.train()

        task.reset_episode()

        # session-wide exposure counts for novelty regressor
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

                # novelty regressor (analysis only)
                kL = seen_count[left_id] + 1
                kR = seen_count[right_id] + 1
                novL = beta_var(float(kL), 1.0)
                novR = beta_var(float(kR), 1.0)
                dNov = float(novL - novR)

                # increment exposures on offer
                seen_count[left_id] += 1
                seen_count[right_id] += 1

                # identity features for stim/decision steps
                id_feats = self._embed_pair(left_id, right_id)

                # 1) stimulus step
                x_stim = self._make_step_input(id_feats, self.state_encoder.stim(info))
                _, _, hidden = self.policy_network(x_stim, hidden)
                h_stim = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                # 2) decision step (capture head activations here)
                x_dec = self._make_step_input(id_feats, self.state_encoder.decision(info))
                logits, values, hidden, acts = self.policy_network(x_dec, hidden, return_activations=True)

                h_dec = hidden[0][-1].squeeze(0).detach().cpu().numpy()
                v_dec = values[:, -1, 0].detach().cpu().item()

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
                beta_map[chosen_id].update(r)

                id_feats_fb = self._embed_feedback(chosen_id, chosen_side)
                x_fb = self._make_step_input(id_feats_fb, self.state_encoder.feedback(r, info))
                _, _, hidden = self.policy_network(x_fb, hidden)
                h_fb = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                # store decision-step tensors for block update
                saved_logps.append(logp.squeeze(0))
                saved_values.append(values[:, -1, 0].squeeze(0))
                saved_rewards.append(torch.tensor(float(r), device=self.device))
                saved_entropies.append(entropy.squeeze(0))

                # probes (training, sampled actions)
                if capture_probes:
                    # key point: acts["lstm_out"][:, -1, :] is what feeds both heads
                    # these are the downstream activations
                    lstm_out_dec = acts["lstm_out"][:, -1, :]  # (1,h)
                    ph_pre_dec = acts["ph_pre"][:, -1, :]      # (1,hh)
                    ph_post_dec = acts["ph_post"][:, -1, :]    # (1,hh)
                    vh_pre_dec = acts["vh_pre"][:, -1, :]      # (1,hh)
                    vh_post_dec = acts["vh_post"][:, -1, :]    # (1,hh)

                    self.training_stats["probe_rows"].append(
                        {
                            "mode": "train_sampled",
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
                            "logit_left": float(dec_logits[0, 0].detach().cpu().item()),
                            "logit_right": float(dec_logits[0, 1].detach().cpu().item()),
                            "logit_diff": logit_diff,
                            "value_dec": float(v_dec),
                            "h_stim": h_stim.tolist(),
                            "h_dec": h_dec.tolist(),
                            "h_fb": h_fb.tolist(),
                            # requested captures
                            "lstm_out_dec": self._tensor_to_list(lstm_out_dec),
                            "ph_pre_dec": self._tensor_to_list(ph_pre_dec),
                            "ph_post_dec": self._tensor_to_list(ph_post_dec),
                            "vh_pre_dec": self._tensor_to_list(vh_pre_dec),
                            "vh_post_dec": self._tensor_to_list(vh_post_dec),
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

                # advantage normalization for stability
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

                        chosen_side = int(dec_logits.argmax(dim=-1).item())
                        chosen = left_bandit if chosen_side == 0 else right_bandit
                        chosen_id = chosen.stim_id

                        r = chosen.sample_reward()
                        episode_reward += r

                        # feedback step
                        id_feats_fb = self._embed_feedback(chosen_id, chosen_side)
                        x_fb = self._make_step_input(id_feats_fb, self.state_encoder.feedback(r, info))
                        _, _, hidden = self.policy_network(x_fb, hidden)

                total_rewards.append(episode_reward)

        return float(np.mean(total_rewards)), float(np.std(total_rewards))

    def run_greedy_probes(self, task: BanditTask, num_episodes=1, clear_existing=True):
        """
        greedy probe run (no backward, weights fixed)

        key point:
          - hidden state updates every step (stim/dec/fb), even in no_grad
          - we capture the same "what feeds the head" + head hidden activations at decision
        """
        self.policy_network.eval()
        self.id_embedding.eval()

        if clear_existing:
            self.training_stats["greedy_probe_rows"] = []

        with torch.no_grad():
            for ep_i in range(num_episodes):
                task.reset_episode()
                seen_count = defaultdict(int)  # keep the same novelty bookkeeping for analysis
                episode_reward = 0.0

                for block in range(task.n_blocks):
                    task.start_block(block)
                    hidden = None

                    beta_map = {b.stim_id: BlockBeta() for b in task.current_bandits}

                    for trial in range(task.trials_per_block):
                        left_bandit, right_bandit = task.sample_two_bandits(trial)
                        left_id, right_id = left_bandit.stim_id, right_bandit.stim_id
                        info = {"block": block, "trial": trial}

                        # novelty (analysis only)
                        kL = seen_count[left_id] + 1
                        kR = seen_count[right_id] + 1
                        novL = beta_var(float(kL), 1.0)
                        novR = beta_var(float(kR), 1.0)
                        dNov = float(novL - novR)
                        seen_count[left_id] += 1
                        seen_count[right_id] += 1

                        id_feats = self._embed_pair(left_id, right_id)

                        # stim step (we keep it for hidden dynamics, but probe at decision)
                        x_stim = self._make_step_input(id_feats, self.state_encoder.stim(info))
                        _, _, hidden = self.policy_network(x_stim, hidden)
                        h_stim = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                        # decision step (greedy + activation capture)
                        x_dec = self._make_step_input(id_feats, self.state_encoder.decision(info))
                        logits, values, hidden, acts = self.policy_network(x_dec, hidden, return_activations=True)

                        dec_logits = logits[:, -1, :]  # (1,2)
                        v_dec = values[:, -1, 0].detach().cpu().item()

                        # greedy action
                        chosen_side = int(dec_logits.argmax(dim=-1).item())
                        chosen = left_bandit if chosen_side == 0 else right_bandit
                        chosen_id = chosen.stim_id

                        # beta-based q/unc (analysis only)
                        QL, QR = beta_map[left_id].mean, beta_map[right_id].mean
                        UL, UR = beta_map[left_id].var, beta_map[right_id].var
                        dQ = float(QL - QR)
                        dUnc = float(UL - UR)

                        logit_left = float(dec_logits[0, 0].detach().cpu().item())
                        logit_right = float(dec_logits[0, 1].detach().cpu().item())
                        logit_diff = float((dec_logits[0, 0] - dec_logits[0, 1]).detach().cpu().item())

                        # feedback step
                        r = chosen.sample_reward()
                        episode_reward += r
                        beta_map[chosen_id].update(r)

                        id_feats_fb = self._embed_feedback(chosen_id, chosen_side)
                        x_fb = self._make_step_input(id_feats_fb, self.state_encoder.feedback(r, info))
                        _, _, hidden = self.policy_network(x_fb, hidden)
                        h_fb = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                        # requested captures at decision
                        lstm_out_dec = acts["lstm_out"][:, -1, :]
                        ph_pre_dec = acts["ph_pre"][:, -1, :]
                        ph_post_dec = acts["ph_post"][:, -1, :]
                        vh_pre_dec = acts["vh_pre"][:, -1, :]
                        vh_post_dec = acts["vh_post"][:, -1, :]

                        self.training_stats["greedy_probe_rows"].append(
                            {
                                "mode": "eval_greedy",
                                "greedy_episode_idx": ep_i,
                                "block": block,
                                "trial": trial,
                                "left_id": int(left_id),
                                "right_id": int(right_id),
                                "choice_side": int(chosen_side),
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
                                "logit_left": logit_left,
                                "logit_right": logit_right,
                                "logit_diff": logit_diff,
                                "value_dec": float(v_dec),
                                "h_stim": h_stim.tolist(),
                                "h_fb": h_fb.tolist(),
                                # key point: this is the actual head input
                                "lstm_out_dec": self._tensor_to_list(lstm_out_dec),
                                "ph_pre_dec": self._tensor_to_list(ph_pre_dec),
                                "ph_post_dec": self._tensor_to_list(ph_post_dec),
                                "vh_pre_dec": self._tensor_to_list(vh_pre_dec),
                                "vh_post_dec": self._tensor_to_list(vh_post_dec),
                            }
                        )

        return self.training_stats["greedy_probe_rows"]

    def build_probe_dataframe_from_rows(self, rows):
        if rows is None or len(rows) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # flatten list-vectors into columns for easy decoding/regression
        vec_cols = [
            "h_stim",
            "h_dec",
            "h_fb",
            "lstm_out_dec",
            "ph_pre_dec",
            "ph_post_dec",
            "vh_pre_dec",
            "vh_post_dec",
        ]
        for col in vec_cols:
            if col in df.columns and df[col].notnull().all():
                try:
                    H = np.stack(df[col].to_numpy())
                    H_cols = {f"{col}_{i}": H[:, i] for i in range(H.shape[1])}
                    df = pd.concat([df.drop(columns=[col]), pd.DataFrame(H_cols)], axis=1)
                except Exception:
                    # if a column is missing for some rows, leave it as-is
                    pass

        return df

    def build_probe_dataframe(self):
        # training probes only
        return self.build_probe_dataframe_from_rows(self.training_stats.get("probe_rows", []))

    def build_greedy_probe_dataframe(self):
        return self.build_probe_dataframe_from_rows(self.training_stats.get("greedy_probe_rows", []))

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

    task = BanditTask(n_blocks=30, trials_per_block=TRIALS_PER_BLOCK, stim_pool_size=200)

    agent = RLAgent(
        hidden_size=128,
        lr=3e-4,
        norm_trials=task.trials_per_block,
        num_stimuli=task.stim_pool_size,
        id_emb_dim=16,
    )



    print(f"training for {NUM_EPISODES} episodes...")
    for ep in range(NUM_EPISODES):
        reward, _ = agent.train_episode(task, render=(ep % 20 == 0), capture_probes=True)

        if ep % EVAL_INTERVAL == 0:
            eval_mean, eval_std = agent.evaluate(task, num_episodes=5)
            print(f"episode {ep:3d} | train reward: {reward:6.1f} | eval: {eval_mean:6.1f} ± {eval_std:4.1f}")

            agent.policy_network.train()
            agent.id_embedding.train()

    print("\nfinal evaluation:")
    mean_r, std_r = agent.evaluate(task, num_episodes=20)
    print(f"final performance: {mean_r:.2f} ± {std_r:.2f}")

    # training probes csv (sampled actions during training)
    # probe_df = agent.build_probe_dataframe()
    # print(f"train probe df shape: {probe_df.shape}")
    # probe_df.to_csv("probe_rows.csv", index=False)

    # greedy probes csv (deterministic argmax, fixed weights)
    agent.run_greedy_probes(task, num_episodes=1, clear_existing=True)
    greedy_df = agent.build_greedy_probe_dataframe()
    print(f"greedy probe df shape: {greedy_df.shape}")
    greedy_df.to_csv("probe_rows_greedy_02.csv", index=False)

    agent.plot_training_progress()
    return agent



if __name__ == "__main__":
    trained_agent = main()
