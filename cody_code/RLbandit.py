import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from collections import deque
import copy

# Import the game mechanics from 3stepbandit
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
    Encodes the 3-step bandit game state into neural network input
    """
    
    def __init__(self, state_dim=8):
        super().__init__()
        self.state_dim = state_dim

    # encode stimulus presentation step    
    def encode_stimulus_step(self, available_bandits, block_info):
        # One-hot encoding of available bandits [A, B, C]
        bandit_encoding = torch.zeros(3)
        for bandit in available_bandits:
            bandit_idx = {'A': 0, 'B': 1, 'C': 2}[bandit]
            bandit_encoding[bandit_idx] = 1.0
            
        # Block and trial information
        block_encoding = torch.tensor([block_info['block'] / 30.0, 
                                     block_info['trial'] / 15.0])
        
        # Step type encoding (stimulus=1, decision=0, feedback=0)
        step_encoding = torch.tensor([1.0, 0.0, 0.0])
        
        return torch.cat([bandit_encoding, block_encoding, step_encoding])
    
    # encode making a choice
    def encode_decision_step(self, available_bandits, block_info):
        # Same bandit encoding as stimulus
        bandit_encoding = torch.zeros(3)
        for bandit in available_bandits:
            bandit_idx = {'A': 0, 'B': 1, 'C': 2}[bandit]
            bandit_encoding[bandit_idx] = 1.0
            
        block_encoding = torch.tensor([block_info['block'] / 30.0, 
                                     block_info['trial'] / 15.0])
        
        # Step type encoding (stimulus=0, decision=1, feedback=0)
        step_encoding = torch.tensor([0.0, 1.0, 0.0])
        
        return torch.cat([bandit_encoding, block_encoding, step_encoding])
    
    # encode getting feedback
    def encode_feedback_step(self, reward, block_info):
        # No bandit encoding for feedback
        bandit_encoding = torch.zeros(3)
        
        block_encoding = torch.tensor([block_info['block'] / 30.0, 
                                     block_info['trial'] / 15.0])
        
        # Step type and reward encoding
        step_encoding = torch.tensor([0.0, 0.0, 1.0])
        reward_encoding = torch.tensor([float(reward)])
        
        return torch.cat([bandit_encoding, block_encoding, step_encoding, reward_encoding])

class LSTMPolicyNetwork(nn.Module):
    """LSTM-based policy network for the 3-step bandit task."""
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
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
        # LSTM forward pass
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
            policy_logits = policy_logits.masked_fill(~mask, float('-inf'))
            
        return policy_logits, values, hidden

class RLAgent:
    def __init__(self, input_size=8, hidden_size=128, lr=0.001, gamma=0.99, 
                 entropy_coef=0.01, value_coef=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Neural network components
        self.state_encoder = StateEncoder()
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
        
    def get_action(self, state, available_bandits, hidden=None, training=True):
        """Get action from policy network."""
        with torch.set_grad_enabled(training):
            if training:
                self.policy_network.train()
            else:
                self.policy_network.eval()
                
            # Encode state
            state_tensor = state.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, state_dim]
            
            # Create mask for available bandits
            mask = torch.zeros(1, 1, 3, dtype=torch.bool, device=self.device)
            for bandit in available_bandits:
                bandit_idx = {'A': 0, 'B': 1, 'C': 2}[bandit]
                mask[0, 0, bandit_idx] = True
            
            # Forward pass
            policy_logits, values, hidden = self.policy_network(state_tensor, hidden, mask)
            
            # Sample action
            action_probs = F.softmax(policy_logits, dim=-1)
            if training:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                entropy = action_dist.entropy()
            else:
                action = action_probs.argmax(dim=-1)
                log_prob = None
                entropy = None
                
            return action.item(), log_prob, values, hidden, entropy
    
    def store_experience(self, state, action, reward, next_state, log_prob, value, 
                        done, hidden, entropy):
        """Store experience in buffer."""
        # Convert tensors to numpy to avoid gradient issues
        experience = {
            'state': state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.detach().cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state,
            'log_prob': log_prob.detach().cpu().numpy() if isinstance(log_prob, torch.Tensor) else log_prob,
            'value': value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value,
            'done': done,
            'hidden': (hidden[0].detach().cpu().numpy(), hidden[1].detach().cpu().numpy()) if hidden is not None else None,
            'entropy': entropy.detach().cpu().numpy() if isinstance(entropy, torch.Tensor) else entropy
        }
        self.experience_buffer.append(experience)
    
    def compute_returns(self, rewards, values, dones):
        """Compute discounted returns."""
        returns = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)
    
    def update_policy(self, batch_size=32):
        """
        Update policy using collected experiences
        """
        if len(self.experience_buffer) < batch_size:
            return
            
        # Sample batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        # Extract batch data - states are already numpy arrays
        states_np = np.array([exp['state'] for exp in batch])
        actions = [exp['action'] for exp in batch]
        rewards = [exp['reward'] for exp in batch]
        dones = [exp['done'] for exp in batch]
        
        # Convert back to tensors with fresh computational graph
        states = torch.tensor(states_np, dtype=torch.float32, device=self.device, requires_grad=True)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # Forward pass to get current policy and values
        policy_logits, values, _ = self.policy_network(states)
        
        # Create mask for available actions (simplified - assume all actions available)
        mask = torch.ones_like(policy_logits, dtype=torch.bool, device=self.device)
        masked_logits = policy_logits.masked_fill(~mask, float('-inf'))
        
        # Compute policy probabilities
        action_probs = F.softmax(masked_logits, dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        
        # Select actions and compute log probabilities
        selected_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute returns (simplified - no discounting for this example)
        returns = rewards_tensor
        
        # Compute advantages
        advantages = returns - values.squeeze()
        
        # Policy loss
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy loss
        entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -entropy
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store statistics
        self.training_stats['losses'].append(total_loss.item())
        self.training_stats['entropies'].append(entropy.item())
        
        return total_loss.item()
    
    def train_episode(self, task, render=False):
        """Train the agent on one episode of the bandit task."""
        episode_reward = 0
        episode_steps = []
        
        # Initialize hidden state
        hidden = None
        
        for block in range(task.n_blocks):
            bandits = task.generate_bandits()
            if render:
                print(f"Block {block+1}: {[f'{b.name}:{b.true_prob}' for b in bandits]}")
            
            for trial in range(task.trials_per_block):
                # Sample two bandits
                A, B = random.sample(bandits, 2)
                available_bandits = [A.name, B.name]
                
                # Stimulus step
                block_info = {'block': block, 'trial': trial}
                state = self.state_encoder.encode_stimulus_step(available_bandits, block_info)
                
                # Decision step
                decision_state = self.state_encoder.encode_decision_step(available_bandits, block_info)
                action, log_prob, value, hidden, entropy = self.get_action(
                    decision_state, available_bandits, hidden, training=True)
                
                # Convert action to bandit choice
                # Map action index (0, 1, or 2) to bandit names, then find in available bandits
                bandit_names = ['A', 'B', 'C']
                chosen_bandit_name = bandit_names[action]
                
                # Find the chosen bandit in available options
                if chosen_bandit_name in available_bandits:
                    chosen_bandit = chosen_bandit_name
                else:
                    # Fallback: choose randomly from available bandits
                    chosen_bandit = random.choice(available_bandits)
                
                chosen = A if chosen_bandit == A.name else B
                
                # Get reward
                reward = chosen.sample_reward()
                
                # Feedback step
                feedback_state = self.state_encoder.encode_feedback_step(reward, block_info)
                
                # Store experience
                self.store_experience(
                    decision_state, action, reward, feedback_state, 
                    log_prob, value, False, hidden, entropy
                )
                
                episode_reward += reward
                episode_steps.append({
                    'block': block,
                    'trial': trial,
                    'chosen': chosen_bandit,
                    'reward': reward,
                    'available': available_bandits
                })
                
                # Update policy every few steps
                if len(self.experience_buffer) >= 32:
                    self.update_policy()
        
        self.training_stats['rewards'].append(episode_reward)
        return episode_reward, episode_steps
    
    def evaluate(self, task, num_episodes=10):
        """Evaluate the agent's performance."""
        total_rewards = []
        
        for episode in range(num_episodes):
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
                        decision_state, available_bandits, hidden, training=False)
                    
                    # Convert action to bandit choice
                    bandit_names = ['A', 'B', 'C']
                    chosen_bandit_name = bandit_names[action]
                    
                    # Find the chosen bandit in available options
                    if chosen_bandit_name in available_bandits:
                        chosen_bandit = chosen_bandit_name
                    else:
                        # Fallback: choose randomly from available bandits
                        chosen_bandit = random.choice(available_bandits)
                    
                    chosen = A if chosen_bandit == A.name else B
                    
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
    agent = RLAgent(hidden_size=128, lr=0.001)
    
    # Training parameters
    num_episodes = 100
    eval_interval = 10
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Train episode 
        # render every 20 episodes
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

