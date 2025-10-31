import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fixed mapping for 3-bandit setup
BANDIT_INDEX = {'A': 0, 'B': 1, 'C': 2}
BANDIT_NAMES = ['A', 'B', 'C']


class Bandit:
    def __init__(self, name, true_prob):
        self.name = name
        self.true_prob = true_prob

    def sample_reward(self):
        # sample bernoulli reward
        # reward or no (1 or 0)
        return float(np.random.rand() < self.true_prob)


class BanditTask:
    # block-wise nonstationary bandits (per paper design)
    def __init__(self, n_blocks=30, trials_per_block=15):
        self.n_blocks = n_blocks
        self.trials_per_block = trials_per_block
        self.log = []

    def generate_bandits(self):
        # 3 bandits with probs in [0.2, 0.8]
        names = ['A', 'B', 'C']
        probs = np.round(np.random.uniform(0.2, 0.8, size=3), 2)
        return [Bandit(names[i], probs[i]) for i in range(3)]

    def summarize(self):
        print("bandit task ready")


class StateEncoder(nn.Module):
    # 8-d: 3 avail + 1 trial norm (where are you in a block) + 3 step flags (curr time step) + 1 reward
    # keeping explicit state one-hots for now
    # reward is 0 except at feedback
    def __init__(self, state_dim=8, norm_trials=15):
        super().__init__()
        self.state_dim = state_dim
        self.norm_trials = float(norm_trials)

    def _avail(self, avail):
        x = torch.zeros(3, dtype=torch.float32)
        for b in avail:
            x[BANDIT_INDEX[b]] = 1.0
        return x

    def _trial(self, info):
        # [0,1) scaling by default
        t = info['trial'] / self.norm_trials
        return torch.tensor([t], dtype=torch.float32)

    def stim(self, avail, info):
        # step: stimulus (reward slot zero)
        return torch.cat([
            self._avail(avail), self._trial(info),
            torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0])
        ]).to(torch.float32)

    def decision(self, avail, info):
        # step: decision (reward slot zero)
        return torch.cat([
            self._avail(avail), self._trial(info),
            torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0])
        ]).to(torch.float32)

    def feedback(self, reward, info):
        # step: feedback (reward slot carries outcome)
        return torch.cat([
            torch.zeros(3), self._trial(info),
            torch.tensor([0.0, 0.0, 1.0]), torch.tensor([float(reward)])
        ]).to(torch.float32)


class LSTMPolicyNetwork(nn.Module):
    # lstm backbone with policy + value heads
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.0): #disable dropout (prev 0.1)
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )

        #prob not necessary
        # self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # 3 bandits
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, hidden=None, mask=None):
        # x: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        logits = self.policy_head(out)  # (b,s,3)
        values = self.value_head(out)   # (b,s,1)

        # mask invalid actions if provided
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        return logits, values, hidden


# simple beta tracker per arm within a block for analysis-only regressors
# (not used by the policy)
class BlockBeta:
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
        return (self.a * self.b) / (((ab ** 2) * (ab + 1.0)))


class RLAgent:
    def __init__(
        self,
        input_size=8,
        hidden_size=128,
        lr=3e-4, #prev 1e-3
        gamma=0.99,
        entropy_coef=0.002,
        value_coef=0.5,
        norm_trials=15
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model parts
        self.state_encoder = StateEncoder(state_dim=input_size, norm_trials=norm_trials)
        self.policy_network = LSTMPolicyNetwork(input_size=input_size, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # rl hyperparams
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # logs
        self.training_stats = {
            'rewards': [],
            'losses': [],
            'entropies': [],
            # rows for probing hidden states across steps
            # (analysis-only; does not affect learning)
            'probe_rows': []
        }

    def _build_action_mask(self, avail, batch_shape=(1, 1)):
        # true for available actions
        mask = torch.zeros(*batch_shape, 3, dtype=torch.bool, device=self.device)
        for b in avail:
            mask[..., BANDIT_INDEX[b]] = True
        return mask

    def _valid_slice(self, logits, avail):
        # slice logits down to the 2 valid arms for unbiased sampling
        idx = torch.tensor([BANDIT_INDEX[n] for n in avail], device=self.device)
        return logits.index_select(1, idx), idx

    def train_episode(self, task, render=False):
        # train one episode across all blocks
        self.policy_network.train()  #enable train mode (dropout on)
        episode_reward = 0.0

        for block in range(task.n_blocks):
            bandits = task.generate_bandits()
            if render:
                print(f"Block {block+1}: {[f'{b.name}:{b.true_prob}' for b in bandits]}")

            # reset recurrent state at block start
            hidden = None

            # analysis-only trackers per arm for this block
            # beta map for q/uncertainty regressors
            beta_map = {n: BlockBeta() for n in ['A', 'B', 'C']}

            # per-block decision-step accumulators
            saved_logps = []
            saved_values = []
            saved_rewards = []
            saved_entropies = []

            for trial in range(task.trials_per_block):
                # sample two bandits this trial
                A, B = random.sample(bandits, 2)
                avail = [A.name, B.name]
                info = {'block': block, 'trial': trial}

                # 1) stimulus step
                x_stim = self.state_encoder.stim(avail, info).to(self.device).view(1, 1, -1)
                mask_stim = self._build_action_mask(avail, batch_shape=(1, 1))
                _, _, hidden = self.policy_network(x_stim, hidden, mask_stim)
                # capture hidden at stimulus
                h_stim = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                # 2) decision step
                x_dec = self.state_encoder.decision(avail, info).to(self.device).view(1, 1, -1)
                mask_dec = self._build_action_mask(avail, batch_shape=(1, 1))
                logits, values, hidden = self.policy_network(x_dec, hidden, mask_dec)  # (1,1,3), (1,1,1)
                # capture hidden and value at decision
                h_dec = hidden[0][-1].squeeze(0).detach().cpu().numpy()
                v_dec = values[:, -1, 0].detach().cpu().item()

                # safe sampling over only the two valid arms (no fallback)
                dec_logits = logits[:, -1, :]                       # (1,3)
                valid_logits, valid_idx = self._valid_slice(dec_logits, avail)  # (1,2)
                log_probs = F.log_softmax(valid_logits, dim=-1)     # (1,2)
                probs = log_probs.exp()                              # (1,2)
                dist = torch.distributions.Categorical(probs)
                action2 = dist.sample()                              # 0/1
                logp = dist.log_prob(action2)
                entropy = dist.entropy()

                chosen_name = avail[action2.item()]
                chosen = A if chosen_name == A.name else B

                # compute simple analysis-only regressors before outcome
                # q and uncertainty from per-arm beta (within block)
                QL, QR = beta_map[A.name].mean, beta_map[B.name].mean
                UL, UR = beta_map[A.name].var,  beta_map[B.name].var
                dQ = float(QL - QR)
                dUnc = float(UL - UR)
                # policy logit diff over valid arms
                logit_diff = float((valid_logits[0, 0] - valid_logits[0, 1]).detach().cpu().item())

                # 3) env feedback step
                r = chosen.sample_reward()
                episode_reward += r

                x_fb = self.state_encoder.feedback(r, info).to(self.device).view(1, 1, -1)
                # feedback has no valid actions; zero mask
                mask_fb = torch.zeros_like(mask_dec)
                _, _, hidden = self.policy_network(x_fb, hidden, mask_fb)
                # capture hidden at feedback
                h_fb = hidden[0][-1].squeeze(0).detach().cpu().numpy()

                # save decision-step tensors for a2c
                saved_logps.append(logp.squeeze(0))
                saved_values.append(values[:, -1, 0].squeeze(0))
                saved_rewards.append(torch.tensor(float(r), device=self.device))
                saved_entropies.append(entropy.squeeze(0))

                # update analysis-only beta for chosen arm
                beta_map[chosen_name].update(r)

                # append probe row (step-specific hidden states + regressors)
                self.training_stats['probe_rows'].append({
                    'episode_idx': len(self.training_stats['rewards']),  # current episode index before push
                    'block': block, 'trial': trial,
                    'offer_L': A.name, 'offer_R': B.name,
                    'choice': 0 if chosen is A else 1,
                    'reward': float(r),
                    'QL': float(QL), 'QR': float(QR),
                    'UncL': float(UL), 'UncR': float(UR),
                    'dQ': dQ, 'dUnc': dUnc,
                    'logit_diff_valid': logit_diff,
                    'value_dec': float(v_dec),
                    'h_stim': h_stim.tolist(),
                    'h_dec': h_dec.tolist(),
                    'h_fb': h_fb.tolist(),
                })

            # update once per block over all decision steps
            if saved_logps:
                logp_t = torch.stack(saved_logps)
                value_t = torch.stack(saved_values)
                reward_t = torch.stack(saved_rewards)
                entropy_t = torch.stack(saved_entropies).mean()

                # returns equal rewards for bandit setting
                returns = reward_t

                # advantage with normalization
                advantages = returns - value_t.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))

                policy_loss = -(logp_t * advantages).mean()
                value_loss = F.mse_loss(value_t, returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_t

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
                self.optimizer.step()

                self.training_stats['losses'].append(loss.item())
                self.training_stats['entropies'].append(entropy_t.item())

        self.training_stats['rewards'].append(episode_reward)
        return episode_reward, None

    def evaluate(self, task, num_episodes=10):
        # greedy evaluation (argmax at decision step)
        self.policy_network.eval()  # new: eval mode (dropout off)
        total_rewards = []

        with torch.no_grad():  # new: disable grad for speed and stability
            for _ in range(num_episodes):
                episode_reward = 0.0
                for block in range(task.n_blocks):
                    bandits = task.generate_bandits()
                    hidden = None

                    # analysis-only trackers for evaluation block
                    beta_map = {n: BlockBeta() for n in ['A', 'B', 'C']}

                    for trial in range(task.trials_per_block):
                        A, B = random.sample(bandits, 2)
                        avail = [A.name, B.name]
                        info = {'block': block, 'trial': trial}

                        # stimulus step
                        x_stim = self.state_encoder.stim(avail, info).to(self.device).view(1, 1, -1)
                        mask_stim = self._build_action_mask(avail, batch_shape=(1, 1))
                        _, _, hidden = self.policy_network(x_stim, hidden, mask_stim)

                        # decision step
                        x_dec = self.state_encoder.decision(avail, info).to(self.device).view(1, 1, -1)
                        mask_dec = self._build_action_mask(avail, batch_shape=(1, 1))
                        logits, _, hidden = self.policy_network(x_dec, hidden, mask_dec)
                        dec_logits = logits[:, -1, :]  # (1,3)

                        # argmax only among valid arms
                        valid_logits, valid_idx = self._valid_slice(dec_logits, avail)  # (1,2)
                        action2 = valid_logits.argmax(dim=-1).item()
                        chosen_name = avail[action2]
                        chosen = A if chosen_name == A.name else B

                        # feedback step
                        r = chosen.sample_reward()
                        episode_reward += r

                        # update eval beta for analysis-only parity
                        beta_map[chosen_name].update(r)

                        x_fb = self.state_encoder.feedback(r, info).to(self.device).view(1, 1, -1)
                        mask_fb = torch.zeros_like(mask_dec)
                        _, _, hidden = self.policy_network(x_fb, hidden, mask_fb)

                total_rewards.append(episode_reward)

        return np.mean(total_rewards), np.std(total_rewards)

    def build_probe_dataframe(self):
        # helper to convert probe rows to df
        if len(self.training_stats.get('probe_rows', [])) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(self.training_stats['probe_rows'])
        # flatten hidden states into columns for quick external probing if desired
        if 'h_dec' in df.columns:
            H = np.stack(df['h_dec'].to_numpy())
            H_cols = {f'h_dec_{i}': H[:, i] for i in range(H.shape[1])}
            df = pd.concat([df.drop(columns=['h_dec']), pd.DataFrame(H_cols)], axis=1)
        if 'h_stim' in df.columns:
            Hs = np.stack(df['h_stim'].to_numpy())
            Hs_cols = {f'h_stim_{i}': Hs[:, i] for i in range(Hs.shape[1])}
            df = pd.concat([df.drop(columns=['h_stim']), pd.DataFrame(Hs_cols)], axis=1)
        if 'h_fb' in df.columns:
            Hf = np.stack(df['h_fb'].to_numpy())
            Hf_cols = {f'h_fb_{i}': Hf[:, i] for i in range(Hf.shape[1])}
            df = pd.concat([df.drop(columns=['h_fb']), pd.DataFrame(Hf_cols)], axis=1)
        return df

    def plot_training_progress(self):
        # simple plots for reward/loss/entropy
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
        axes[0, 0].plot(self.training_stats['rewards'])
        axes[0, 0].set_title('episode rewards')
        axes[0, 0].set_xlabel('episode')
        axes[0, 0].set_ylabel('total reward')
    
        axes[0, 1].plot(self.training_stats['losses'])
        axes[0, 1].set_title('training loss')
        axes[0, 1].set_xlabel('update')
        axes[0, 1].set_ylabel('loss')
    
        axes[1, 0].plot(self.training_stats['entropies'])
        axes[1, 0].set_title('policy entropy')
        axes[1, 0].set_xlabel('update')
        axes[1, 0].set_ylabel('entropy')
    
        if len(self.training_stats['rewards']) > 10:
            window = min(50, len(self.training_stats['rewards']) // 4)
            moving_avg = pd.Series(self.training_stats['rewards']).rolling(window=window).mean()
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title(f'moving average rewards (window={window})')
            axes[1, 1].set_xlabel('episode')
            axes[1, 1].set_ylabel('avg reward')
    
        plt.tight_layout()
        plt.savefig('training.png')
        plt.show()


def main():
    # main training loop
    print("initializing rl agent for 3-step bandit...")

    np.random.seed(408); random.seed(408); torch.manual_seed(408)

    task = BanditTask(n_blocks=30, trials_per_block=15)
    agent = RLAgent(
        input_size=8,
        hidden_size=128,
        lr=3e-4, #prev 1e-3
         norm_trials=task.trials_per_block
    )

    num_episodes = 100
    eval_interval = 10

    print(f"training for {num_episodes} episodes...")
    for ep in range(num_episodes):
        reward, _ = agent.train_episode(task, render=(ep % 20 == 0))
        if ep % eval_interval == 0:
            eval_mean, eval_std = agent.evaluate(task, num_episodes=5)
            print(f"episode {ep:3d} | train reward: {reward:6.1f} | eval: {eval_mean:6.1f} ± {eval_std:4.1f}")
            
            #switch back to train mode explicitly (defensive)
            agent.policy_network.train()

    print("\nfinal evaluation:")
    mean_r, std_r = agent.evaluate(task, num_episodes=20)
    print(f"final performance: {mean_r:.2f} ± {std_r:.2f}")

    # new: build and preview probe dataframe shape
    probe_df = agent.build_probe_dataframe()
    print(f"probe df shape: {probe_df.shape}")
    
    # save for external analysis
    probe_df.to_csv("probe_rows.csv", index=False)

    agent.plot_training_progress()
    return agent


if __name__ == "__main__":
    trained_agent = main()
