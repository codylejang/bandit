# train.py
# builds inputs, trains the model on behavior sequences

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent import RNNAgent
from data import simulate_delta_agent

def train_rnn_on_behavior():
    # Simulate training data
    reward_probs = [0.8, 0.3]
    epsilon = 0.1
    num_trials = 10000
    actions, rewards, _ = simulate_delta_agent(reward_probs, epsilon, num_trials)

    # Build input sequence: [action one-hot, reward] at t-1
    input_size = 3
    X = []
    Y = []
    for t in range(1, num_trials):
        prev_action = actions[t-1]
        onehot = [int(prev_action == 0), int(prev_action == 1)]
        x_t = onehot + [rewards[t-1]]
        X.append(x_t)
        Y.append(actions[t])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, 3]
    Y = torch.tensor(Y, dtype=torch.long).unsqueeze(0)     # shape: [1, T]

    # Initialize model
    model = RNNAgent(input_size=3, hidden_size=10, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.squeeze(0), Y.squeeze(0))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model