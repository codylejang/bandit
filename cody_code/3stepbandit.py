import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

class Bandit:
    def __init__(self, name, true_prob, eta=0.15, uI=1.0, nI=5.0):
        self.name = name
        self.true_prob = true_prob
        self.eta = eta # recency rate for counts (0<eta<1)
        self.uI = uI
        self.nI = nI
        self.reset() # reset values when bandit created

    def reset(self):
        """
        Reset recency-weighted counts and optimistic-init priors at block start.
        W, L are recency-weighted wins/losses; alpha0, beta0 hold novelty-biased priors.
        """
        self.W = 0.0
        self.L = 0.0
        self.alpha0 = 1.0
        self.beta0  = 1.0
        self.exposures = 0

    def on_exposed(self):
        """
        Novelty as optimistic initiation (Equation 3):
            Apply a one-time prior bias on first exposure in a block.

            w0_t = (1 - eta)^(exposures - 1)
            bias = nI * w0_t
            alpha0 = 1 + max(bias, 0)   # novelty-seeking
            beta0  = 1 + max(-bias, 0)  # novelty-avoidance

        Called each time the bandit is shown; only the first exposure changes priors.
        """
        self.exposures += 1
        if self.exposures == 1:
            w0 = (1.0 - self.eta) ** (self.exposures - 1)  # = 1.0 on first exposure
            bias = self.nI * w0
            if bias >= 0:
                self.alpha0 = 1.0 + bias
                self.beta0  = 1.0
            else:
                self.alpha0 = 1.0
                self.beta0  = 1.0 + (-bias)

    def _alpha_beta(self):
        """
        Current posterior parameters:
            alpha_t = alpha0 + W_t
            beta_t  = beta0  + L_t
        """
        alpha = self.alpha0 + self.W
        beta  = self.beta0  + self.L
        return alpha, beta

    def expected_value(self):
        """
        Expected value (Equation 4) using recency-weighted counts (Eqs. 5–7):
            Q_t = alpha_t / (alpha_t + beta_t)
        where alpha_t = alpha0 + W_t, beta_t = beta0 + L_t.
        """
        a, b = self._alpha_beta()
        return a / (a + b)

    def uncertainty(self):
        """
        Stimulus uncertainty (Equation 9), normalized as in Methods:
            V_t = (1/12) * [ alpha_t * beta_t / ((alpha_t + beta_t)^2 * (alpha_t + beta_t + 1)) ]
        """
        a, b = self._alpha_beta()
        var = (a * b) / (((a + b) ** 2) * (a + b + 1))
        return var / 12.0

    def sample_reward(self):
        """
        Environment reward draw (Bernoulli with parameter true_prob).
        Note: This does NOT update beliefs; call update(reward) afterward.
        """
        return float(np.random.rand() < self.true_prob)

    def update(self, reward):
        """
        Recency-weighted Bayesian count update for the chosen bandit:
            W_t = (1 - eta) * W_{t-1} + 1{reward = 1}
            L_t = (1 - eta) * L_{t-1} + 1{reward = 0}
        """
        self.W = (1.0 - self.eta) * self.W + (1.0 if reward else 0.0)
        self.L = (1.0 - self.eta) * self.L + (0.0 if reward else 1.0)

class BanditTask:
    '''
    On each trial, two of the three bandits are randomly sampled (without replacement),
    we apply optimistic-init novelty on first exposure, compute U = Q + uI*V, and choose
    via a binary softmax (Equation 2).
    '''
    def __init__(self, n_blocks=30, trials_per_block=15):
        self.n_blocks = n_blocks
        self.trials_per_block = trials_per_block
        self.log = []

    def generate_bandits(self):
        names = ['A','B','C']
        probs = np.round(np.random.uniform(0.2, 0.8, size=3), 2)
        return [Bandit(names[i], probs[i]) for i in range(3)]

    def run(self):
        for block in range(self.n_blocks):
            bandits = self.generate_bandits()
            print(f"Block {block+1}: {[f'{b.name}:{b.true_prob}' for b in bandits]}")

            for trial in range(self.trials_per_block):
                # sample two options
                A, B = random.sample(bandits, 2)

                # register exposures (optimistic-init novelty on first sight)
                A.on_exposed()
                B.on_exposed()

                # compute utilities: U = Q + uI * V  
                qA, qB = A.expected_value(), B.expected_value()
                vA, vB = A.uncertainty(),    B.uncertainty()
                uA = qA + A.uI * vA
                uB = qB + B.uI * vB

                # binary softmax (Eq 2)
                inv_temp = 3.0
                pA   = 1.0 / (1.0 + np.exp(inv_temp * (uB - uA)))
                chosen = A if np.random.rand() < pA else B

                # reward + update; compute RPE using PRE-update EV of the chosen arm
                Q_pre = qA if (chosen is A) else qB
                reward = chosen.sample_reward()
                chosen.update(reward)
                rpe = float(reward) - Q_pre

                # log trial data
                self.log.append({
                    'block':      block,
                    'trial':      trial,
                    'options':    (A.name, B.name),
                    'utilities':  {A.name: uA, B.name: uB},
                    'ev':         {A.name: qA, B.name: qB},    # pre-update EVs
                    'uncertainty_vals': {A.name: vA, B.name: vB},
                    'chosen':     chosen.name,
                    'reward':     float(reward),
                    'rpe':        float(rpe)
                })

    def summarize(self):
        total_reward = sum(entry['reward'] for entry in self.log)
        total_trials = len(self.log)
        print(f"Total rewards: {total_reward}")
        print(f"Total trials:  {total_trials}")

class LSTMAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x, hc=None):
        out, (hn, cn) = self.lstm(x, hc)
        logits = self.fc(out)
        return logits, (hn, cn)

# run sim
task = BanditTask()
task.run()
task.summarize()

# Build 3-step [avail_A, avail_B, avail_C, go, reward] sequence
def build_3step_sequence(log):
    bandit2index = {'A':0, 'B':1, 'C':2}
    X, Y, decision_mask = [], [], []
    for entry in log:
        # 1) stimulus step: which bandits are shown
        avail = [0.0, 0.0, 0.0]
        for name in entry['options']:
            avail[bandit2index[name]] = 1.0  # one-hot available
        X.append(avail + [0.0] + [0.0])     # default: [go=0]+[reward=0]
        decision_mask.append(False)

        # 2) decision step
        X.append([0.0, 0.0, 0.0] + [1.0] + [0.0]) # encoding marks decision step
        bandit_index = bandit2index[entry['chosen']]
        Y.append(bandit_index) # label: which arm chosen
        decision_mask.append(True)

        # 3) feedback step
        X.append([0.0, 0.0, 0.0] + [0.0] + [entry['reward']]) # last encoding: reward or no
        decision_mask.append(False)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)               # [1, 3T, 5]
    Y = torch.tensor(Y, dtype=torch.long)                               
    decision_mask = torch.tensor(decision_mask, dtype=torch.bool).unsqueeze(0)  # [1, 3T]
    return X, Y, decision_mask

X_seq, Y_seq, seq_mask = build_3step_sequence(task.log)

# init and train LSTM with input_size=5
input_size, hidden_size, output_size = 5, 64, 3
model     = LSTMAgent(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

model.train()
block_size = 3 * task.trials_per_block
n_blocks   = task.n_blocks

for epoch in range(1, 31):
    total_loss = 0.0
    for b in range(n_blocks):
        # slice out one block’s 3T steps
        start = b * block_size
        end = (b + 1) * block_size
        xb = X_seq[:, start:end, :]               # [1, 3*trials_per_block, input_size]
        mb = seq_mask[:, start:end]               # [1, 3*trials_per_block]

        label_start = b * task.trials_per_block
        label_end   = (b + 1) * task.trials_per_block
        yb_block = Y_seq[label_start:label_end]   # [trials_per_block]

        optimizer.zero_grad()
        logits, _ = model(xb, None)               # [1, 3*trials_per_block, output_size]
        logits_dec = logits[mb].view(-1, output_size)
        loss = criterion(logits_dec, yb_block)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    #compute avg loss over all blocks per epoch to visualize learning trend
    avg_loss = total_loss / n_blocks
    print(f"Epoch {epoch:02d} — Avg Loss: {avg_loss:.4f}")

# check training accuracy on decision steps
model.eval()
with torch.no_grad():
    logits, _ = model(X_seq, None)
    logits_dec = logits[seq_mask].view(-1, output_size)
    preds = logits_dec.argmax(dim=-1)
    acc = (preds == Y_seq).float().mean()
    print(f"Train accuracy (decision-only): {acc:.3%}")

# extract hidden states and decode
model.eval()
with torch.no_grad():
    hidden_seq, _ = model.lstm(X_seq, None)   # [1, 3T, 64]
h = hidden_seq.squeeze(0).cpu().numpy()       # [3T, 64]

T = h.shape[0] // 3
stim_states     = h[0::3]   # [T, 64]
decision_states = h[1::3]   # [T, 64]
feedback_states = h[2::3]   # [T, 64]

# prepare targets
u0 = np.array([e['utilities'][e['options'][0]] for e in task.log])
u1 = np.array([e['utilities'][e['options'][1]] for e in task.log])
u_diff = u0 - u1

bandit2index = {'A':0, 'B':1, 'C':2}
choice = np.array([bandit2index[e['chosen']] for e in task.log])

# use logged, policy-aligned RPE
rpe = np.array([e['rpe'] for e in task.log])

# decode stimulus difference
lr = LinearRegression().fit(stim_states, u_diff)
print("Stimulus u_diff R2:", r2_score(u_diff, lr.predict(stim_states)))

# decode choice
clf = LogisticRegression(max_iter=200).fit(decision_states, choice)
print("Decision accuracy:", accuracy_score(choice, clf.predict(decision_states)))

# decode feedback RPE
lr2 = LinearRegression().fit(feedback_states, rpe)
print("Feedback RPE R2:", r2_score(rpe, lr2.predict(feedback_states)))
