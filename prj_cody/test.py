# test.py
#  evaluates the model in a new environment (Phase 2)
import torch
import numpy as np

def run_rnn_in_new_environment(model):
    model.eval()

    reward_probs = [0.3, 0.8]  # new test environment
    num_trials = 500

    prev_action = 0
    prev_reward = 1
    h = None

    actions = []
    rewards = []

    for _ in range(num_trials):
        x = [int(prev_action == 0), int(prev_action == 1), prev_reward]
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, 1, -1)  # shape: [1, 1, 3]
        logits, h = model(x_tensor, h)
        probs = torch.softmax(logits[0, 0], dim=0).detach().numpy()
        action = np.random.choice([0, 1], p=probs)
        reward = np.random.rand() < reward_probs[action]

        actions.append(action)
        rewards.append(reward)

        prev_action = action
        prev_reward = reward

    # Optional: plot or analyze results here
    print("Final action distribution:", np.mean(actions))
    print("Total reward:", np.sum(rewards))