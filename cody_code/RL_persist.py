import os
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
TRIALS_PER_BLOCK = 15
NUM_EPISODES = 1500
EVAL_INTERVAL = 5
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

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
            ids = keep_two + [new_id]  # index 2 is always the novel stimulus

        probs = self._sample_reward_probs()
        self.current_bandits = [Bandit(stim_id=ids[i], true_prob=probs[i]) for i in range(3)]

        if block_idx == 0:
            # no held-out in first block
            self.heldout_idx = None
            self.heldout_start_trial = 0  # all 3 available from trial 0
        else:
            # alternate: odd blocks hold out novel (idx 2),
            #            even blocks hold out a familiar one (idx 0 or 1)
            if block_idx % 2 == 1:
                self.heldout_idx = 2  # novel
            else:
                self.heldout_idx = random.choice([0, 1])  # familiar

            lo = min(7, self.trials_per_block)
            hi = self.trials_per_block
            self.heldout_start_trial = random.randint(lo, hi)

    def sample_two_bandits(self, trial_idx: int):
        if self.current_bandits is None:
            raise RuntimeError("call start_block(block_idx) before sampling trials.")

        if self.heldout_idx is not None and trial_idx < self.heldout_start_trial:
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
    RL²-style step encoding: includes last (action, reward) at every step.

    output dim = 7:
      - 1 trial fraction
      - 3 step flags: [stim, decision, feedback]
      - 1 reward scalar (current-step reward; nonzero only at feedback)
      - 1 last_action scalar (side of most recent chosen, 0/1; 0 at block start)
      - 1 last_reward scalar (reward of most recent feedback; 0 at block start)

    the "last_*" fields carry the most recently observed feedback event's
    (action, reward), updated right before each feedback step. this gives the
    LSTM an explicit RL² view of (s_t, a_{t-1}, r_{t-1}) at every step.
    """

    def __init__(self, norm_trials=15):
        super().__init__()
        self.norm_trials = float(norm_trials)

    @property
    def out_dim(self) -> int:
        return 7

    def _trial(self, info) -> torch.Tensor:
        t = float(info["trial"]) / self.norm_trials
        return torch.tensor([t], dtype=torch.float32)

    def stim(self, info, last_action, last_reward) -> torch.Tensor:
        return torch.cat([
            self._trial(info),
            torch.tensor([1.0, 0.0, 0.0, 0.0, float(last_action), float(last_reward)], dtype=torch.float32),
        ])

    def decision(self, info, last_action, last_reward) -> torch.Tensor:
        return torch.cat([
            self._trial(info),
            torch.tensor([0.0, 1.0, 0.0, 0.0, float(last_action), float(last_reward)], dtype=torch.float32),
        ])

    def feedback(self, reward, info, last_action, last_reward) -> torch.Tensor:
        return torch.cat([
            self._trial(info),
            torch.tensor([0.0, 0.0, 1.0, float(reward), float(last_action), float(last_reward)], dtype=torch.float32),
        ])


class LSTMPolicyNetwork(nn.Module):
    """
    lstm backbone with bilinear identity-queried policy + value heads

    policy: each option's score is a bilinear form over (context, embedding):
        score(option) = proj(context) · embedding
    this forces multiplicative interaction — if context is uninformative,
    all scores are zero regardless of embedding. identity embeddings serve
    as queries into the LSTM's hidden-state memory, not as additive bypasses.

    symmetry is preserved: the same projection is used for both options,
    so no slot bias can develop.
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.0, id_emb_dim=4):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.head_hidden = self.hidden_size // 2
        self.id_emb_dim = int(id_emb_dim)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # policy head: context → id_emb_dim projection
        # score(option) = projected_context · embedding  (bilinear retrieval)
        # shared weights across both options → no slot bias
        self.policy_proj = nn.Linear(self.hidden_size, self.id_emb_dim)

        self.vh_fc1 = nn.Linear(self.hidden_size, self.head_hidden)
        self.vh_relu = nn.ReLU()
        self.vh_drop = nn.Dropout(dropout)
        self.vh_fc2 = nn.Linear(self.head_hidden, 1)

        # auxiliary reward prediction head: bilinear retrieval of per-identity reward
        # predicted_reward(id) = sigmoid(projected_context · embedding)
        self.aux_proj = nn.Linear(self.hidden_size, self.id_emb_dim)

    def forward(self, x, hidden=None, return_activations=False):
        out, hidden = self.lstm(x, hidden)  # out: (b,s,h)

        # value head (context only — scalar state value, not per-option)
        vh_pre = self.vh_fc1(out)           # (b,s,hh)
        vh_post = self.vh_relu(vh_pre)      # (b,s,hh)
        vh_post = self.vh_drop(vh_post)
        values = self.vh_fc2(vh_post)       # (b,s,1)

        if not return_activations:
            return out, values, hidden

        activations = {
            "lstm_out": out,
            "vh_pre": vh_pre,
            "vh_post": vh_post,
        }
        return out, values, hidden, activations

    def predict_reward(self, lstm_out, emb):
        """
        Predict expected reward for a specific identity via bilinear retrieval.
        lstm_out: (1, h), emb: (1, id_emb_dim)
        returns predicted reward in [0,1]: (1, 1)
        """
        proj = self.aux_proj(lstm_out)  # (1, id_emb_dim)
        score = (proj * emb).sum(dim=-1, keepdim=True)  # (1, 1) — dot product
        return torch.sigmoid(score)

    def get_policy_logits(self, lstm_out, emb_left, emb_right, return_activations=False):
        """
        Policy logits via bilinear scoring:
          score(option) = projected_context · option_embedding
        lstm_out: (1, h), emb_left/emb_right: (1, id_emb_dim)
        returns logits: (1, 2) — [left_score, right_score]
        """
        proj = self.policy_proj(lstm_out)  # (1, id_emb_dim)
        score_l = (proj * emb_left).sum(dim=-1, keepdim=True)   # (1, 1)
        score_r = (proj * emb_right).sum(dim=-1, keepdim=True)  # (1, 1)
        logits = torch.cat([score_l, score_r], dim=-1)  # (1, 2)

        if not return_activations:
            return logits

        activations = {
            "policy_proj": proj,
        }
        return logits, activations


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
        entropy_coef=0.02,
        value_coef=0.5,
        aux_coef=2.0,
        update_freq=15,
        norm_trials=15,
        num_stimuli=200,
        id_emb_dim=4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = int(hidden_size)
        self.num_stimuli = int(num_stimuli)
        self.id_emb_dim = int(id_emb_dim)

        self.state_encoder = StateEncoder(norm_trials=norm_trials)

        # identity embeddings: re-randomized every episode (not learned across episodes)
        # provides distinguishable but arbitrary identity tokens, like fresh paintings
        self.id_embedding = nn.Embedding(self.num_stimuli, self.id_emb_dim).to(self.device)

        # total input dim = left_emb + right_emb + last_chosen_emb + base_state
        # last_chosen_emb is zero before any feedback, else the most recent chosen id's embedding
        lstm_input_size = (3 * self.id_emb_dim) + self.state_encoder.out_dim
        self.policy_network = LSTMPolicyNetwork(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.0,
            id_emb_dim=self.id_emb_dim,
        ).to(self.device)

        # only optimize LSTM/policy weights — embeddings are random per episode
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=lr,
        )
        # cosine decay: lr → 0.1*lr over total training
        self.scheduler = None  # set up in main() after NUM_EPISODES is known

        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.aux_coef = float(aux_coef)
        self.update_freq = int(update_freq)

        # best model checkpoint (by eval reward) — only LSTM/policy weights
        self.best_eval_reward = -float("inf")
        self.best_eval_episode = -1
        self.best_state = None

        # best model checkpoint (by human similarity — highest shape_r).
        # shape_r is primary because raw NLL has a uniform-baseline floor
        # (~ln 2) tighter than achievable signal given human accuracy.
        self.best_human_shape = -float("inf")
        self.best_human_episode = -1
        self.best_human_state = None
        # also track best temp-fit NLL as a secondary diagnostic
        self.best_temp_nll = float("inf")
        self.best_temp_episode = -1
        self.best_temp_state = None

        self.training_stats = {
            "rewards": [],
            "losses": [],
            "entropies": [],
            "eval_rewards": [],        # (episode_idx, mean, std)
            "greedy_left_frac": [],    # (episode_idx, fraction_left)
            "greedy_accuracy": [],     # (episode_idx, accuracy)
            "greedy_slot0_frac": [],   # (episode_idx, fraction picking slot 0)
            "greedy_acc_easy": [],     # (episode_idx, accuracy on easy trials)
            "greedy_acc_hard": [],     # (episode_idx, accuracy on hard trials)
            "human_similarity": [],    # (episode_idx, nll, agreement, shape_r, temp_fit_nll, temp_fit_tau)
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

    def _embed_triple(self, left_id: int, right_id: int, last_chosen_id) -> torch.Tensor:
        """
        RL² input embedding: left/right offered pair + last chosen id.
        last_chosen_id is None (zero vector) before any feedback has been observed
        in the current block.
        """
        left_t = torch.tensor([left_id], device=self.device, dtype=torch.long)
        right_t = torch.tensor([right_id], device=self.device, dtype=torch.long)
        e_left = self.id_embedding(left_t)    # (1, emb)
        e_right = self.id_embedding(right_t)  # (1, emb)
        if last_chosen_id is None:
            e_last = torch.zeros_like(e_left)
        else:
            last_t = torch.tensor([int(last_chosen_id)], device=self.device, dtype=torch.long)
            e_last = self.id_embedding(last_t)
        return torch.cat([e_left, e_right, e_last], dim=-1)  # (1, 3*emb)

    def checkpoint_if_best(self, eval_reward: float, episode: int = -1):
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.best_eval_episode = episode
            self.best_state = {
                "policy_network": {k: v.clone() for k, v in self.policy_network.state_dict().items()},
            }

    def restore_best(self):
        if self.best_state is not None:
            self.policy_network.load_state_dict(self.best_state["policy_network"])
            print(f"restored best model (ep {self.best_eval_episode}, eval reward: {self.best_eval_reward:.1f})")

    def checkpoint_if_best_human(self, shape_r: float, episode: int = -1):
        if not (shape_r != shape_r) and shape_r > self.best_human_shape:
            self.best_human_shape = float(shape_r)
            self.best_human_episode = episode
            self.best_human_state = {
                "policy_network": {k: v.clone() for k, v in self.policy_network.state_dict().items()},
            }

    def restore_best_human(self):
        if self.best_human_state is not None:
            self.policy_network.load_state_dict(self.best_human_state["policy_network"])
            print(f"restored most human-similar model (ep {self.best_human_episode}, shape_r: {self.best_human_shape:.4f})")

    def checkpoint_if_best_temp(self, temp_nll: float, episode: int = -1):
        if temp_nll < self.best_temp_nll:
            self.best_temp_nll = float(temp_nll)
            self.best_temp_episode = episode
            self.best_temp_state = {
                "policy_network": {k: v.clone() for k, v in self.policy_network.state_dict().items()},
            }

    def restore_best_temp(self):
        if self.best_temp_state is not None:
            self.policy_network.load_state_dict(self.best_temp_state["policy_network"])
            print(f"restored best temp-fit-NLL model (ep {self.best_temp_episode}, tNLL: {self.best_temp_nll:.4f})")

    def save_checkpoint(self, episode, checkpoint_dir=CHECKPOINT_DIR):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"ep{episode:03d}.pt")
        torch.save({
            "episode": episode,
            "policy_network": self.policy_network.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(ckpt["policy_network"])
        return ckpt.get("episode", -1)

    def _make_step_input(self, id_feats: torch.Tensor, base_state: torch.Tensor) -> torch.Tensor:
        # id_feats: (1, 2*emb), base_state: (base_dim,)
        base_state = base_state.to(self.device).unsqueeze(0)        # (1, base_dim)
        x = torch.cat([id_feats, base_state], dim=-1).unsqueeze(0)  # (1, 1, 2*emb+base_dim)
        return x

    def _tensor_to_list(self, t: torch.Tensor) -> list:
        # safe conversion for csv logging
        return t.detach().cpu().float().view(-1).tolist()

    def randomize_embeddings(self):
        #Fresh random identity tokens — like seeing new paintings
        nn.init.normal_(self.id_embedding.weight, mean=0.0, std=1.0)

    def train_episode(self, task: BanditTask, render=False):
        self.policy_network.train()
        self.randomize_embeddings()

        task.reset_episode()
        episode_reward = 0.0

        for block in range(task.n_blocks):
            task.start_block(block)
            if render:
                bdesc = [f"id{b.stim_id}:{b.true_prob}" for b in task.current_bandits]
                print(f"block {block+1}: {bdesc} | heldout={task.heldout_idx} start={task.heldout_start_trial}")

            hidden = None  # reset per block (probs reset each block)

            # RL² memory: most recent feedback event (reset at block start)
            last_chosen_id = None
            last_action = 0.0
            last_reward = 0.0

            # block-local beta trackers keyed by stim_id
            beta_map = {b.stim_id: BlockBeta() for b in task.current_bandits}

            saved_logps, saved_values, saved_rewards, saved_entropies, saved_aux_preds = [], [], [], [], []

            for trial in range(task.trials_per_block):
                left_bandit, right_bandit = task.sample_two_bandits(trial)
                left_id, right_id = left_bandit.stim_id, right_bandit.stim_id
                info = {"block": block, "trial": trial}

                id_feats = self._embed_triple(left_id, right_id, last_chosen_id)

                # 1) stimulus step
                x_stim = self._make_step_input(id_feats, self.state_encoder.stim(info, last_action, last_reward))
                _, _, hidden = self.policy_network(x_stim, hidden)

                # 2) decision step
                x_dec = self._make_step_input(id_feats, self.state_encoder.decision(info, last_action, last_reward))
                lstm_out, values, hidden = self.policy_network(x_dec, hidden)

                # policy via bilinear retrieval: embeddings as queries into context-memory
                context = lstm_out[:, -1, :]  # (1, h)
                emb_l = self.id_embedding(torch.tensor([left_id], device=self.device, dtype=torch.long))
                emb_r = self.id_embedding(torch.tensor([right_id], device=self.device, dtype=torch.long))
                dec_logits = self.policy_network.get_policy_logits(context, emb_l, emb_r)

                log_probs = F.log_softmax(dec_logits, dim=-1) # (1,2)
                probs = log_probs.exp()
                dist = torch.distributions.Categorical(probs)

                action = dist.sample()                        # 0=left, 1=right
                logp = dist.log_prob(action)
                entropy = dist.entropy()

                chosen_side = int(action.item())
                chosen = left_bandit if chosen_side == 0 else right_bandit
                chosen_id = chosen.stim_id

                # aux: predict reward for the chosen identity via bilinear retrieval
                emb_chosen = emb_l if chosen_side == 0 else emb_r
                aux_pred = self.policy_network.predict_reward(context, emb_chosen)  # (1, 1)

                # 3) feedback step — update last_* before building input so fb
                # input reflects just-observed (chosen, action, reward)
                r = chosen.sample_reward()
                episode_reward += r
                beta_map[chosen_id].update(r)

                last_chosen_id = chosen_id
                last_action = float(chosen_side)
                last_reward = float(r)

                id_feats_fb = self._embed_triple(left_id, right_id, last_chosen_id)
                x_fb = self._make_step_input(id_feats_fb, self.state_encoder.feedback(r, info, last_action, last_reward))
                _, _, hidden = self.policy_network(x_fb, hidden)

                saved_logps.append(logp.squeeze(0))
                saved_values.append(values[:, -1, 0].squeeze(0))
                saved_rewards.append(torch.tensor(float(r), device=self.device))
                saved_entropies.append(entropy.squeeze(0))
                saved_aux_preds.append(aux_pred.squeeze())

                # update every update_freq trials, or at block end
                if (trial + 1) % self.update_freq == 0 or trial == task.trials_per_block - 1:
                    if saved_logps:
                        logp_t = torch.stack(saved_logps)
                        value_t = torch.stack(saved_values)
                        reward_t = torch.stack(saved_rewards)
                        entropy_t = torch.stack(saved_entropies).mean()
                        aux_pred_t = torch.stack(saved_aux_preds)

                        returns = reward_t

                        advantages = returns - value_t.detach()
                        adv_std = advantages.std()
                        if adv_std > 0.01:
                            advantages = (advantages - advantages.mean()) / adv_std

                        policy_loss = -(logp_t * advantages).mean()
                        value_loss = F.mse_loss(value_t, returns)
                        aux_loss = F.mse_loss(aux_pred_t, reward_t)

                        loss = (policy_loss
                                + self.value_coef * value_loss
                                - self.entropy_coef * entropy_t
                                + self.aux_coef * aux_loss)

                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.policy_network.parameters(),
                            1.0,
                        )
                        self.optimizer.step()

                        # detach hidden: graph is stale after weight update
                        if hidden is not None:
                            hidden = (hidden[0].detach(), hidden[1].detach())

                        self.training_stats["losses"].append(loss.item())
                        self.training_stats["entropies"].append(entropy_t.item())

                        saved_logps, saved_values, saved_rewards, saved_entropies, saved_aux_preds = [], [], [], [], []

        self.training_stats["rewards"].append(episode_reward)
        return episode_reward, None

    def evaluate(self, task: BanditTask, num_episodes=10):
        self.policy_network.eval()

        total_rewards = []
        all_sides = []       # 0=left, 1=right (actual side)
        all_slots = []       # 0=slot0, 1=slot1 (network action)
        all_correct = []     # did it pick the higher-prob option?
        all_prob_diffs = []  # true_prob(left) - true_prob(right)
        all_entropies = []   # softmax entropy at each decision (exploration health)

        with torch.no_grad():
            for _ in range(num_episodes):
                task.reset_episode()
                self.randomize_embeddings()
                episode_reward = 0.0

                for block in range(task.n_blocks):
                    task.start_block(block)
                    hidden = None
                    last_chosen_id = None
                    last_action = 0.0
                    last_reward = 0.0

                    for trial in range(task.trials_per_block):
                        left_bandit, right_bandit = task.sample_two_bandits(trial)
                        left_id, right_id = left_bandit.stim_id, right_bandit.stim_id
                        info = {"block": block, "trial": trial}

                        id_feats = self._embed_triple(left_id, right_id, last_chosen_id)

                        # stimulus step
                        x_stim = self._make_step_input(id_feats, self.state_encoder.stim(info, last_action, last_reward))
                        _, _, hidden = self.policy_network(x_stim, hidden)

                        # decision step (greedy via bilinear policy head)
                        x_dec = self._make_step_input(id_feats, self.state_encoder.decision(info, last_action, last_reward))
                        lstm_out, _, hidden = self.policy_network(x_dec, hidden)
                        context = lstm_out[:, -1, :]
                        emb_l = self.id_embedding(torch.tensor([left_id], device=self.device, dtype=torch.long))
                        emb_r = self.id_embedding(torch.tensor([right_id], device=self.device, dtype=torch.long))
                        dec_logits = self.policy_network.get_policy_logits(context, emb_l, emb_r)

                        # softmax entropy (exploration diagnostic)
                        probs = F.softmax(dec_logits, dim=-1)
                        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                        all_entropies.append(float(ent.item()))

                        chosen_side = int(dec_logits.argmax(dim=-1).item())
                        chosen = left_bandit if chosen_side == 0 else right_bandit
                        chosen_id = chosen.stim_id

                        r = chosen.sample_reward()
                        episode_reward += r

                        # track diagnostics
                        all_sides.append(chosen_side)
                        all_slots.append(chosen_side)  # no slot distinction anymore
                        prob_diff = left_bandit.true_prob - right_bandit.true_prob
                        all_prob_diffs.append(prob_diff)
                        if prob_diff > 0:
                            all_correct.append(1 if chosen_side == 0 else 0)
                        elif prob_diff < 0:
                            all_correct.append(1 if chosen_side == 1 else 0)
                        else:
                            all_correct.append(1)

                        # feedback step
                        last_chosen_id = chosen_id
                        last_action = float(chosen_side)
                        last_reward = float(r)
                        id_feats_fb = self._embed_triple(left_id, right_id, last_chosen_id)
                        x_fb = self._make_step_input(id_feats_fb, self.state_encoder.feedback(r, info, last_action, last_reward))
                        _, _, hidden = self.policy_network(x_fb, hidden)

                total_rewards.append(episode_reward)

        left_frac = float(np.mean([s == 0 for s in all_sides]))
        slot0_frac = float(np.mean([s == 0 for s in all_slots]))
        accuracy = float(np.mean(all_correct))

        # easy vs hard split (threshold = 0.2 on |prob_diff|)
        EASY_THRESH = 0.2
        prob_diffs_arr = np.array(all_prob_diffs)
        correct_arr = np.array(all_correct)
        easy_mask = np.abs(prob_diffs_arr) > EASY_THRESH
        hard_mask = ~easy_mask
        acc_easy = float(correct_arr[easy_mask].mean()) if easy_mask.sum() > 0 else float("nan")
        acc_hard = float(correct_arr[hard_mask].mean()) if hard_mask.sum() > 0 else float("nan")

        mean_entropy = float(np.mean(all_entropies)) if all_entropies else float("nan")
        return float(np.mean(total_rewards)), float(np.std(total_rewards)), left_frac, slot0_frac, accuracy, acc_easy, acc_hard, all_prob_diffs, all_sides, all_correct, mean_entropy

    def run_greedy_probes(self, task: BanditTask, num_episodes=1, clear_existing=True):
        """
        greedy probe run (no backward, weights fixed)

        key point:
          - hidden state updates every step (stim/dec/fb), even in no_grad
          - we capture the same "what feeds the head" + head hidden activations at decision
        """
        self.policy_network.eval()

        if clear_existing:
            self.training_stats["greedy_probe_rows"] = []

        with torch.no_grad():
            for ep_i in range(num_episodes):
                task.reset_episode()
                self.randomize_embeddings()
                seen_count = defaultdict(int)  # keep the same novelty bookkeeping for analysis
                episode_reward = 0.0

                for block in range(task.n_blocks):
                    task.start_block(block)
                    hidden = None
                    last_chosen_id = None
                    last_action = 0.0
                    last_reward = 0.0

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

                        id_feats = self._embed_triple(left_id, right_id, last_chosen_id)

                        # stim step
                        x_stim = self._make_step_input(id_feats, self.state_encoder.stim(info, last_action, last_reward))
                        _, _, hidden = self.policy_network(x_stim, hidden)
                        h_stim = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                        # decision step (greedy via bilinear policy head + activation capture)
                        x_dec = self._make_step_input(id_feats, self.state_encoder.decision(info, last_action, last_reward))
                        lstm_out, values, hidden, acts = self.policy_network(x_dec, hidden, return_activations=True)
                        v_dec = values[:, -1, 0].detach().cpu().item()

                        context = lstm_out[:, -1, :]
                        emb_l = self.id_embedding(torch.tensor([left_id], device=self.device, dtype=torch.long))
                        emb_r = self.id_embedding(torch.tensor([right_id], device=self.device, dtype=torch.long))
                        dec_logits, ph_acts = self.policy_network.get_policy_logits(context, emb_l, emb_r, return_activations=True)

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

                        last_chosen_id = chosen_id
                        last_action = float(chosen_side)
                        last_reward = float(r)
                        id_feats_fb = self._embed_triple(left_id, right_id, last_chosen_id)
                        x_fb = self._make_step_input(id_feats_fb, self.state_encoder.feedback(r, info, last_action, last_reward))
                        _, _, hidden = self.policy_network(x_fb, hidden)
                        h_fb = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                        # captures
                        lstm_out_dec = acts["lstm_out"][:, -1, :]
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
                                "lstm_out_dec": self._tensor_to_list(lstm_out_dec),
                                "policy_proj": self._tensor_to_list(ph_acts["policy_proj"]),
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
            "policy_proj",
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

    def plot_training_progress(self, final_prob_diffs=None, final_sides=None, final_correct=None):
        EASY_THRESH = 0.2

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # (0,0) train + eval rewards
        axes[0, 0].plot(self.training_stats["rewards"], alpha=0.3, label="train (per ep)")
        if len(self.training_stats["rewards"]) > 10:
            window = min(50, len(self.training_stats["rewards"]) // 4)
            moving_avg = pd.Series(self.training_stats["rewards"]).rolling(window=window).mean()
            axes[0, 0].plot(moving_avg, label=f"train MA({window})")
        if self.training_stats["eval_rewards"]:
            ep_idxs, means, stds = zip(*self.training_stats["eval_rewards"])
            means, stds = np.array(means), np.array(stds)
            axes[0, 0].plot(ep_idxs, means, "r-o", markersize=2, label="eval (greedy)")
            axes[0, 0].fill_between(ep_idxs, means - stds, means + stds, color="r", alpha=0.15)
        axes[0, 0].set_title("rewards: train vs eval")
        axes[0, 0].set_xlabel("episode")
        axes[0, 0].set_ylabel("total reward")
        axes[0, 0].legend(fontsize=8)

        # (0,1) loss
        axes[0, 1].plot(self.training_stats["losses"])
        axes[0, 1].set_title("training loss")
        axes[0, 1].set_xlabel("update")
        axes[0, 1].set_ylabel("loss")

        # (0,2) entropy
        axes[0, 2].plot(self.training_stats["entropies"])
        axes[0, 2].set_title("policy entropy")
        axes[0, 2].set_xlabel("update")
        axes[0, 2].set_ylabel("entropy")

        # (1,0) left-choice fraction AND slot0 fraction
        if self.training_stats["greedy_left_frac"]:
            ep_idxs, fracs = zip(*self.training_stats["greedy_left_frac"])
            axes[1, 0].plot(ep_idxs, fracs, "b-o", markersize=2, label="left (actual side)")
        if self.training_stats["greedy_slot0_frac"]:
            ep_idxs, fracs = zip(*self.training_stats["greedy_slot0_frac"])
            axes[1, 0].plot(ep_idxs, fracs, "r-x", markersize=2, label="slot0 (network action)")
        axes[1, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        axes[1, 0].set_title("choice fractions")
        axes[1, 0].set_xlabel("episode")
        axes[1, 0].set_ylabel("fraction")
        axes[1, 0].set_ylim(-0.05, 1.05)
        axes[1, 0].legend(fontsize=8)

        # (1,1) greedy accuracy: easy vs hard over training
        if self.training_stats["greedy_accuracy"]:
            ep_idxs, accs = zip(*self.training_stats["greedy_accuracy"])
            axes[1, 1].plot(ep_idxs, accs, "k-o", markersize=2, label="all")
        if self.training_stats["greedy_acc_easy"]:
            ep_idxs, accs_e = zip(*self.training_stats["greedy_acc_easy"])
            axes[1, 1].plot(ep_idxs, accs_e, "g-o", markersize=2, label=f"easy (|dp|>{EASY_THRESH})")
        if self.training_stats["greedy_acc_hard"]:
            ep_idxs, accs_h = zip(*self.training_stats["greedy_acc_hard"])
            axes[1, 1].plot(ep_idxs, accs_h, "r-o", markersize=2, label=f"hard (|dp|≤{EASY_THRESH})")
        axes[1, 1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("greedy accuracy: easy vs hard")
        axes[1, 1].set_xlabel("episode")
        axes[1, 1].set_ylabel("accuracy")
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].legend(fontsize=8)

        # (1,2) final eval: accuracy vs |prob_diff| bins, colored by easy/hard
        if final_prob_diffs is not None and final_correct is not None:
            prob_diffs = np.array(final_prob_diffs)
            correct = np.array(final_correct)
            sides = np.array(final_sides) if final_sides is not None else None

            abs_diffs = np.abs(prob_diffs)
            bins = np.linspace(0, 0.6, 7)  # 0, 0.1, 0.2, ..., 0.6
            bin_centers = []
            bin_accs = []
            bin_colors = []
            bin_left_fracs = []
            for i in range(len(bins) - 1):
                mask = (abs_diffs >= bins[i]) & (abs_diffs < bins[i + 1])
                if mask.sum() > 0:
                    c = (bins[i] + bins[i + 1]) / 2
                    bin_centers.append(c)
                    bin_accs.append(correct[mask].mean())
                    bin_colors.append("tab:green" if c > EASY_THRESH else "tab:red")
                    if sides is not None:
                        bin_left_fracs.append((sides[mask] == 0).mean())

            axes[1, 2].bar(bin_centers, bin_accs, width=0.08, color=bin_colors, alpha=0.7)
            if bin_left_fracs:
                axes[1, 2].plot(bin_centers, bin_left_fracs, "bx-", label="left frac")
            axes[1, 2].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            axes[1, 2].axvline(EASY_THRESH, color="gray", linestyle=":", alpha=0.5, label=f"threshold={EASY_THRESH}")
            axes[1, 2].set_title("final eval: accuracy vs |prob diff|")
            axes[1, 2].set_xlabel("|P(left) - P(right)|")
            axes[1, 2].set_ylabel("fraction")
            axes[1, 2].set_ylim(-0.05, 1.05)
            axes[1, 2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig("training.png", dpi=150)
        plt.close()


def diagnose_learning(agent, task, n_blocks=3):
    """
    Run a single episode and print per-trial diagnostics to check whether
    the LSTM hidden state encodes identity-specific value information.

    Checks:
      1. aux head predictions — do they vary with reward history or stay ~0.5?
      2. logit diffs — do they shift after seeing rewards for an identity?
      3. gradient norms — is the learning signal reaching the LSTM?
    """
    agent.policy_network.eval()
    agent.randomize_embeddings()
    task.reset_episode()

    print("\n" + "-" * 80)
    print("DIAGNOSTIC: per-trial aux predictions, logits, and reward history")

    with torch.no_grad():
        for block in range(n_blocks):
            task.start_block(block)
            hidden = None
            last_chosen_id = None
            last_action = 0.0
            last_reward = 0.0
            reward_history = defaultdict(list)  # stim_id -> [rewards]

            probs_desc = [f"id{b.stim_id}(p={b.true_prob:.2f})" for b in task.current_bandits]
            print(f"\n--- block {block} | {' '.join(probs_desc)} ---")
            print(f"{'trial':>5} | {'left':>6} {'right':>6} | {'auxL':>6} {'auxR':>6} | {'logit_L':>7} {'logit_R':>7} {'diff':>6} | {'chose':>5} {'r':>3} | reward history")

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
                aux_pred_l = agent.policy_network.predict_reward(context, emb_l)
                aux_pred_r = agent.policy_network.predict_reward(context, emb_r)

                logit_l = float(logits[0, 0].item())
                logit_r = float(logits[0, 1].item())
                aux_val_l = float(aux_pred_l.item())
                aux_val_r = float(aux_pred_r.item())

                # greedy choice + reward
                chosen_side = int(logits.argmax(dim=-1).item())
                chosen = left_bandit if chosen_side == 0 else right_bandit
                chosen_id = chosen.stim_id
                r = chosen.sample_reward()
                reward_history[chosen_id].append(r)

                # feedback
                last_chosen_id = chosen_id
                last_action = float(chosen_side)
                last_reward = float(r)
                id_feats_fb = agent._embed_triple(left_id, right_id, last_chosen_id)
                x_fb = agent._make_step_input(id_feats_fb, agent.state_encoder.feedback(r, info, last_action, last_reward))
                _, _, hidden = agent.policy_network(x_fb, hidden)

                # format reward history
                hist_str = "  ".join(
                    f"id{sid}: [{', '.join(f'{rv:.0f}' for rv in rvs)}]"
                    for sid, rvs in sorted(reward_history.items())
                )
                side_str = "L" if chosen_side == 0 else "R"
                print(f"{trial:5d} | {left_id:6d} {right_id:6d} | {aux_val_l:6.3f} {aux_val_r:6.3f} | {logit_l:7.4f} {logit_r:7.4f} {logit_l - logit_r:6.3f} | {side_str:>5} {r:3.0f} | {hist_str}")

    # gradient norms check (one training step)
    print(f"\n--- gradient norms (1 training episode) ---")
    agent.policy_network.train()
    agent.train_episode(task)

    for name, param in agent.policy_network.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            print(f"  {name:30s} | grad norm: {norm:.6f}")
        else:
            print(f"  {name:30s} | no grad")

    print("-" * 80 + "\n")


def main():
    print("start rl agent for 3-step identity-based bandit...")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    task = BanditTask(n_blocks=20, trials_per_block=TRIALS_PER_BLOCK, stim_pool_size=200)

    agent = RLAgent(
        hidden_size=128,
        lr=3e-4,
        aux_coef=2.0,
        update_freq=15,
        norm_trials=task.trials_per_block,
    )

    # load human data once for similarity tracking during training
    from eval_human_similarity import replay_human_trials, compute_metrics
    from eval_human_baseline import load_human_data
    human_df = load_human_data()
    print(f"loaded {len(human_df)} human trials ({human_df['sub_id'].nunique()} subjects) for similarity tracking")

    # cosine LR decay: lr → 0.1*lr over training
    agent.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        agent.optimizer, T_max=NUM_EPISODES, eta_min=3e-5,
    )

    print(f"training for {NUM_EPISODES} episodes...")
    for ep in range(NUM_EPISODES):
        reward, _ = agent.train_episode(task, render=(ep % 100 == 0))
        agent.scheduler.step()

        if ep % EVAL_INTERVAL == 0:
            eval_mean, eval_std, left_frac, slot0_frac, accuracy, acc_easy, acc_hard, _, _, _, eval_entropy = agent.evaluate(task, num_episodes=5)
            agent.training_stats["eval_rewards"].append((ep, eval_mean, eval_std))
            agent.training_stats["greedy_left_frac"].append((ep, left_frac))
            agent.training_stats["greedy_slot0_frac"].append((ep, slot0_frac))
            agent.training_stats["greedy_accuracy"].append((ep, accuracy))
            agent.training_stats["greedy_acc_easy"].append((ep, acc_easy))
            agent.training_stats["greedy_acc_hard"].append((ep, acc_hard))

            # human similarity eval
            replay_df = replay_human_trials(agent, human_df)
            metrics = compute_metrics(replay_df)
            h_nll = metrics["nll"]
            h_agree = metrics["choice_agreement"]
            h_shape = metrics["policy_shape_r"]
            h_tnll = metrics["temp_fit_nll"]
            h_tau = metrics["temp_fit_tau"]
            agent.training_stats["human_similarity"].append((ep, h_nll, h_agree, h_shape, h_tnll, h_tau))
            # primary human-similarity selection: shape_r (NLL has uniform-floor problem)
            agent.checkpoint_if_best_human(h_shape, episode=ep)
            agent.checkpoint_if_best_temp(h_tnll, episode=ep)

            print(f"episode {ep:3d} | train reward: {reward:6.1f} | eval: {eval_mean:6.1f} ± {eval_std:4.1f} | left%: {left_frac:.2f} | slot0%: {slot0_frac:.2f} | acc: {accuracy:.2f} (easy:{acc_easy:.2f} hard:{acc_hard:.2f}) | H: {eval_entropy:.3f} | shape_r: {h_shape:+.3f} | tNLL: {h_tnll:.4f} τ={h_tau:.2f} | rNLL: {h_nll:.4f} agree: {h_agree:.3f} | lr: {agent.scheduler.get_last_lr()[0]:.2e}")
            agent.checkpoint_if_best(eval_mean, episode=ep)
            agent.save_checkpoint(ep)

            agent.policy_network.train()
            agent.id_embedding.train()

    # ---- final eval: best performing ----
    print("\n" + "=" * 70)
    print("BEST PERFORMING (highest eval reward)")
    print("=" * 70)
    agent.restore_best()

    mean_r, std_r, final_left_frac, final_slot0_frac, final_acc, final_acc_easy, final_acc_hard, final_prob_diffs, final_sides, final_correct, final_entropy = agent.evaluate(task, num_episodes=20)
    print(f"performance: {mean_r:.2f} ± {std_r:.2f} | left%: {final_left_frac:.2f} | slot0%: {final_slot0_frac:.2f} | acc: {final_acc:.2f} (easy:{final_acc_easy:.2f} hard:{final_acc_hard:.2f}) | H: {final_entropy:.3f}")

    replay_df = replay_human_trials(agent, human_df)
    metrics = compute_metrics(replay_df)
    print(f"human similarity: NLL={metrics['nll']:.4f} | agree={metrics['choice_agreement']:.3f} | shape_r={metrics['policy_shape_r']:.3f}")

    # greedy probes for best performing
    agent.run_greedy_probes(task, num_episodes=1, clear_existing=True)
    greedy_df = agent.build_greedy_probe_dataframe()
    print(f"greedy probe df shape: {greedy_df.shape}")
    greedy_df.to_csv("probe_rows_greedy_02.csv", index=False)

    agent.plot_training_progress(
        final_prob_diffs=final_prob_diffs,
        final_sides=final_sides,
        final_correct=final_correct,
    )

    diagnose_learning(agent, task)

    # ---- final eval: most human-similar (highest shape_r) ----
    print("\n" + "=" * 70)
    print("MOST HUMAN-SIMILAR (highest shape_r)")
    print("=" * 70)
    agent.restore_best_human()

    mean_r, std_r, hs_left, hs_slot0, hs_acc, hs_easy, hs_hard, _, _, _, hs_entropy = agent.evaluate(task, num_episodes=20)
    print(f"performance: {mean_r:.2f} ± {std_r:.2f} | left%: {hs_left:.2f} | slot0%: {hs_slot0:.2f} | acc: {hs_acc:.2f} (easy:{hs_easy:.2f} hard:{hs_hard:.2f}) | H: {hs_entropy:.3f}")

    replay_df = replay_human_trials(agent, human_df)
    metrics = compute_metrics(replay_df)
    print(f"human similarity: NLL={metrics['nll']:.4f} | tNLL={metrics['temp_fit_nll']:.4f} (τ={metrics['temp_fit_tau']:.2f}) | agree={metrics['choice_agreement']:.3f} | shape_r={metrics['policy_shape_r']:.3f}")

    diagnose_learning(agent, task)

    # ---- final eval: best temp-fit NLL ----
    print("\n" + "=" * 70)
    print("BEST TEMP-FIT NLL (calibration-decoupled)")
    print("=" * 70)
    agent.restore_best_temp()

    mean_r, std_r, t_left, t_slot0, t_acc, t_easy, t_hard, _, _, _, t_entropy = agent.evaluate(task, num_episodes=20)
    print(f"performance: {mean_r:.2f} ± {std_r:.2f} | left%: {t_left:.2f} | slot0%: {t_slot0:.2f} | acc: {t_acc:.2f} (easy:{t_easy:.2f} hard:{t_hard:.2f}) | H: {t_entropy:.3f}")

    replay_df = replay_human_trials(agent, human_df)
    metrics = compute_metrics(replay_df)
    uniform_floor = float(np.log(2.0))
    below = "BELOW" if metrics["temp_fit_nll"] < uniform_floor else "above"
    print(f"human similarity: NLL={metrics['nll']:.4f} | tNLL={metrics['temp_fit_nll']:.4f} ({below} uniform={uniform_floor:.4f}, τ={metrics['temp_fit_tau']:.2f}) | agree={metrics['choice_agreement']:.3f} | shape_r={metrics['policy_shape_r']:.3f}")

    return agent

if __name__ == "__main__":
    trained_agent = main()
