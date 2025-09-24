import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

BANDIT_INDEX = {'A': 0, 'B': 1, 'C': 2}   # CHANGED: reuse map instead of rebuilding repeatedly
BANDIT_NAMES = ['A', 'B', 'C']


class Bandit:
    def __init__(self, name, true_prob):
        self.name = name
        self.true_prob = true_prob

    def sample_reward(self):
        """Sample reward from the bandit's true probability."""
        return float(np.random.rand() < self.true_prob)


class BanditTask:
    def __init__(self, n_blocks=30, trials_per_block=15):
        self.n_blocks = n_blocks
        self.trials_per_block = trials_per_block
        self.log = []

    def generate_bandits(self):
        names = ['A', 'B', 'C']
        probs = np.round(np.random.uniform(0.2, 0.8, size=3), 2)
        return [Bandit(names[i], probs[i]) for i in range(3)]

    def summarize(self):
        print("BanditTask ready for RL agent training")


class StateEncoder(nn.Module):
    """
    Encodes the 3-step bandit game state into neural network input.
    8-D input like the original: 3 (bandit one-hot) + 2 (block/trial) + 3 (step type).
    """
    def __init__(self, state_dim=8, norm_blocks=30, norm_trials=15):
        super().__init__()
        self.state_dim = state_dim
        # CHANGED: make normalization dynamic (no hard-coded 30/15)
        self.norm_blocks = float(norm_blocks)
        self.norm_trials = float(norm_trials)

    def _bt_enc(self, block_info):
        # CHANGED: dynamic scaling using provided maxima
        return torch.tensor(
            [block_info['block'] / self.norm_blocks, block_info['trial'] / self.norm_trials],
            dtype=torch.float32
        )

    # encode stimulus presentation step
    def encode_stimulus_step(self, available_bandits, block_info):
        # One-hot encoding of available bandits [A, B, C]
        bandit_encoding = torch.zeros(3, dtype=torch.float32)
        for bandit in available_bandits:
            bandit_encoding[BANDIT_INDEX[bandit]] = 1.0

        block_encoding = self._bt_enc(block_info)

        # Step type encoding (stimulus=1, decision=0, feedback=0)
        step_encoding = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        return torch.cat([bandit_encoding, block_encoding, step_encoding])

    # encode making a choice
    def encode_decision_step(self, available_bandits, block_info):
        bandit_encoding = torch.zeros(3, dtype=torch.float32)
        for bandit in available_bandits:
            bandit_encoding[BANDIT_INDEX[bandit]] = 1.0

        block_encoding = self._bt_enc(block_info)

        # Step type encoding (stimulus=0, decision=1, feedback=0)
        step_encoding = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

        return torch.cat([bandit_encoding, block_encoding, step_encoding])

    # encode getting feedback
    def encode_feedback_step(self, reward, block_info):
        bandit_encoding = torch.zeros(3, dtype=torch.float32)
        block_encoding = self._bt_enc(block_info)

        # Step type and reward encoding
        step_encoding = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        # NOTE: original code appended reward to the state, making 9 dims.
        # We keep the original declared 8-D interface and *exclude* reward here
        # to remain consistent with policy input (decision step). If desired,
        # you can create a separate critic input that includes reward.
        # To preserve original behavior, comment the above NOTE and uncomment below.
        # reward_encoding = torch.tensor([float(reward)], dtype=torch.float32)
        # return torch.cat([bandit_encoding, block_encoding, step_encoding, reward_encoding])

        return torch.cat([bandit_encoding, block_encoding, step_encoding])


class LSTMPolicyNetwork(nn.Module):
    """LSTM-based policy network for the 3-step bandit task."""
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        # NOTE: With seq_len=1 at training time, attention is effectively a no-op,
        # but we keep it for future sequence parsing plans.

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # 3 bandits
        )

        # Value head for baseline
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, hidden=None, mask=None):
        # x: (batch, seq_len, input_size)
        lstm_out, hidden = self.lstm(x, hidden)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection
        out = lstm_out + attn_out

        # Policy and value outputs
        policy_logits = self.policy_head(out)
        values = self.value_head(out)

        # Apply mask if provided (for unavailable bandits)
        if mask is not None:
            # CHANGED: avoid -inf for better numerics during softmax/logsoftmax
            policy_logits = policy_logits.masked_fill(~mask, -1e9)

        return policy_logits, values, hidden


class RLAgent:
    def __init__(
        self,
        input_size=8,
        hidden_size=128,
        lr=0.001,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        norm_blocks=30,
        norm_trials=15
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Neural network components
        self.state_encoder = StateEncoder(state_dim=input_size, norm_blocks=norm_blocks, norm_trials=norm_trials)
        self.policy_network = LSTMPolicyNetwork(input_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # RL hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)

        # Training statistics
        self.training_stats = {
            'rewards': [],
            'losses': [],
            'entropies': []
        }

    def _build_action_mask(self, available_bandits, batch_shape=(1, 1)):
        """Build boolean mask with True for available actions."""
        mask = torch.zeros(*batch_shape, 3, dtype=torch.bool, device=self.device)
        for b in available_bandits:
            mask[..., BANDIT_INDEX[b]] = True
        return mask

    def get_action(self, state, available_bandits, hidden=None, training=True):
        """Get action from policy network."""
        with torch.set_grad_enabled(training):
            if training:
                self.policy_network.train()
            else:
                self.policy_network.eval()

            # Encode state -> (1,1,state_dim)
            state_tensor = state.to(dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

            # Create mask for available bandits
            mask = self._build_action_mask(available_bandits, batch_shape=(1, 1))

            # Forward pass
            policy_logits, values, hidden = self.policy_network(state_tensor, hidden, mask)

            # CHANGED: detach hidden so we don't backprop across the whole episode
            if hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())

            # Sample/select action
            action_probs = F.softmax(policy_logits, dim=-1)  # (1,1,3)
            if training:
                action_dist = torch.distributions.Categorical(action_probs.squeeze(1))  # (1,3) -> batch of 1
                action = action_dist.sample()  # (1,)
                log_prob = action_dist.log_prob(action)  # (1,)
                entropy = action_dist.entropy()  # (1,)
                action_idx = action.item()
            else:
                action_idx = action_probs.argmax(dim=-1).item()
                log_prob = None
                entropy = None

            return action_idx, log_prob, values, hidden, entropy

    def store_experience(self, state, action, reward, action_mask, done):
        """
        Store minimal experience needed for updates.
        CHANGED: we store (state, action, reward, action_mask, done) only.
        """
        self.experience_buffer.append({
            'state': state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state, dtype=np.float32),
            'action': int(action),
            'reward': float(reward),
            'action_mask': np.asarray(action_mask, dtype=bool),
            'done': bool(done),
        })

    def update_policy(self, batch_size=32):
        """
        Update policy using collected experiences.
        CHANGED: (1) proper LSTM input shape (B,1,8),
                 (2) use true action mask,
                 (3) normalize advantages,
                 (4) remove requires_grad on inputs.
        """
        if len(self.experience_buffer) < batch_size:
            return

        batch = random.sample(self.experience_buffer, batch_size)

        states_np = np.stack([exp['state'] for exp in batch], axis=0)         # (B, 8)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long, device=self.device)  # (B,)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32, device=self.device)  # (B,)
        masks_np = np.stack([exp['action_mask'] for exp in batch], axis=0)    # (B, 3) booleans

        states = torch.tensor(states_np, dtype=torch.float32, device=self.device).unsqueeze(1)        # (B, 1, 8)
        action_mask = torch.tensor(masks_np, dtype=torch.bool, device=self.device).unsqueeze(1)       # (B, 1, 3)

        # Forward (we don't carry hidden across samples here)
        policy_logits, values, _ = self.policy_network(states, hidden=None, mask=action_mask)
        # policy_logits: (B,1,3) masked already inside forward
        # values: (B,1,1)

        # Compute probabilities/log-probabilities
        log_probs = F.log_softmax(policy_logits, dim=-1)[:, 0, :]  # (B,3)
        probs = torch.exp(log_probs)                                # (B,3)

        # Select taken actions
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        values = values[:, 0, 0]  # (B,)

        # For a contextual bandit, 1-step return = reward
        returns = rewards

        # Advantage (normalize for stability)
        advantages = returns - values.detach()
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp_min(1e-6)
        advantages = (advantages - adv_mean) / adv_std

        # Losses
        policy_loss = -(selected_log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (encourage exploration)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # total loss function
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy  # CHANGED: correct sign

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Stats
        self.training_stats['losses'].append(total_loss.item())
        self.training_stats['entropies'].append(entropy.item())

        return total_loss.item()

    def train_episode(self, task, render=False):
        """Train the agent on one episode of the bandit task."""
        episode_reward = 0
        episode_steps = []

        hidden = None

        for block in range(task.n_blocks):
            bandits = task.generate_bandits()
            if render:
                print(f"Block {block+1}: {[f'{b.name}:{b.true_prob}' for b in bandits]}")

            for trial in range(task.trials_per_block):
                # Sample two bandits
                A, B = random.sample(bandits, 2)
                available_bandits = [A.name, B.name]

                # Stimulus & decision states
                block_info = {'block': block, 'trial': trial}
                _ = self.state_encoder.encode_stimulus_step(available_bandits, block_info)  # kept for completeness
                decision_state = self.state_encoder.encode_decision_step(available_bandits, block_info)

                # Choose action
                action, log_prob, value, hidden, entropy = self.get_action(
                    decision_state, available_bandits, hidden, training=True
                )

                # Convert action index to bandit name
                chosen_bandit_name = BANDIT_NAMES[action]

                # Sanity: chosen should be available since logits were masked
                if chosen_bandit_name not in available_bandits:
                    # NOTE: This should not happen due to masking. Keep fallback with a warning.
                    # print("Warning: sampled unavailable action; falling back to a valid one.")
                    chosen_bandit_name = random.choice(available_bandits)

                chosen = A if chosen_bandit_name == A.name else B

                # Get reward
                reward = chosen.sample_reward()

                # Build and store action mask used at this step
                action_mask_np = np.zeros(3, dtype=bool)
                for b in available_bandits:
                    action_mask_np[BANDIT_INDEX[b]] = True

                # Store minimal experience
                self.store_experience(
                    decision_state, action, reward, action_mask_np, done=False
                )

                episode_reward += reward
                episode_steps.append({
                    'block': block,
                    'trial': trial,
                    'chosen': chosen_bandit_name,
                    'reward': reward,
                    'available': available_bandits
                })

                # CHANGED: update frequently with small batches for faster learning
                if len(self.experience_buffer) >= 8:
                    bs = min(32, len(self.experience_buffer))
                    self.update_policy(batch_size=bs)

        self.training_stats['rewards'].append(episode_reward)
        return episode_reward, episode_steps

    def evaluate(self, task, num_episodes=10):
        """Evaluate the agent's performance (greedy)."""
        total_rewards = []

        for _ in range(num_episodes):
            episode_reward = 0
            hidden = None

            for block in range(task.n_blocks):
                bandits = task.generate_bandits()

                for trial in range(task.trials_per_block):
                    A, B = random.sample(bandits, 2)
                    available_bandits = [A.name, B.name]

                    block_info = {'block': block, 'trial': trial}
                    decision_state = self.state_encoder.encode_decision_step(available_bandits, block_info)

                    action, _, _, hidden, _ = self.get_action(
                        decision_state, available_bandits, hidden, training=False
                    )

                    chosen_bandit_name = BANDIT_NAMES[action]
                    if chosen_bandit_name not in available_bandits:
                        chosen_bandit_name = random.choice(available_bandits)

                    chosen = A if chosen_bandit_name == A.name else B
                    reward = chosen.sample_reward()
                    episode_reward += reward

            total_rewards.append(episode_reward)

        return np.mean(total_rewards), np.std(total_rewards)

    def plot_training_progress(self):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Rewards
        axes[0, 0].plot(self.training_stats['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')

        # Losses
        axes[0, 1].plot(self.training_stats['losses'])
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')

        # Entropies
        axes[1, 0].plot(self.training_stats['entropies'])
        axes[1, 0].set_title('Policy Entropy')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Entropy')

        # Moving average rewards
        if len(self.training_stats['rewards']) > 10:
            window = min(50, len(self.training_stats['rewards']) // 4)
            moving_avg = pd.Series(self.training_stats['rewards']).rolling(window=window).mean()
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')

        plt.tight_layout()
        plt.show()


def main():
    """Main training loop."""
    print("Initializing RL Agent for 3-Step Bandit Task...")

    # Create task and agent
    task = BanditTask(n_blocks=30, trials_per_block=15)
    agent = RLAgent(
        input_size=8,
        hidden_size=128,
        lr=0.001,
        norm_blocks=task.n_blocks,     # CHANGED: dynamic scaling
        norm_trials=task.trials_per_block
    )

    # Training parameters
    num_episodes = 100
    eval_interval = 10

    print(f"Training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        reward, steps = agent.train_episode(task, render=(episode % 20 == 0))

        # Evaluation
        if episode % eval_interval == 0:
            eval_reward, eval_std = agent.evaluate(task, num_episodes=5)
            print(f"Episode {episode:3d} | Train Reward: {reward:6.1f} | "
                  f"Eval Reward: {eval_reward:6.1f} ± {eval_std:4.1f}")

    # Final evaluation
    print("\nFinal Evaluation:")
    final_reward, final_std = agent.evaluate(task, num_episodes=20)
    print(f"Final Performance: {final_reward:.2f} ± {final_std:.2f}")

    # Plot training progress
    agent.plot_training_progress()

    return agent


if __name__ == "__main__":
    trained_agent = main()
