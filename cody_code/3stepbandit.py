import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

class Bandit:
    def __init__(self, name, true_prob, decay_rate=0.9, uI=1.0, nI=5.0):
        self.name = name
        self.true_prob = true_prob
        self.decay_rate = decay_rate
        self.uI = uI
        self.nI = nI
        self.reset()  # reset values when bandit created

    def reset(self):
        self.alpha = 1.0
        self.beta = 1.0
        self.seen = 0

    def pull(self):  # generates probabilistic reward outcome after selection
        win = np.random.rand() < self.true_prob
        self.alpha = self.decay_rate * self.alpha + (1.0 if win else 0.0)
        self.beta  = self.decay_rate * self.beta  + (1.0 if not win else 0.0)
        return win

    def expected_value(self):
        '''
        Estimate of reward probability (Equation 4):
            Q = alpha / (alpha + beta)
        '''
        return self.alpha / (self.alpha + self.beta)

    def uncertainty(self):
        '''
        Uncertainty bonus (Equation 9):
            V = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        '''
        a, b = self.alpha, self.beta
        return (a * b) / (((a + b) ** 2) * (a + b + 1))

    def novelty(self):
        '''
        Novelty-initiation bias (Equation 3):
            nI * w0_t, approximated here as nI / (nI + seen)
        '''
        return self.nI / (self.nI + self.seen)

class BanditTask:
    '''
    On each trial, two of the three bandits are randomly sampled (without replacement),
    their exposure counts incremented (for the novelty term), and utilities computed
    via Equation 3. Decisions follow the binary softmax rule (Equation 2)
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
                # increment exposure for novelty
                A.seen += 1
                B.seen += 1

                # compute each bandit’s utility (Eq 3)
                qA, qB = A.expected_value(), B.expected_value()
                vA, vB = A.uncertainty(),    B.uncertainty()
                nA, nB = A.novelty(),        B.novelty()

                uA = qA + A.uI * vA + nA
                uB = qB + B.uI * vB + nB

                # binary softmax (Eq 2)
                beta = 3.0
                pA   = 1.0 / (1.0 + np.exp(beta * (uB - uA)))
                chosen = A if np.random.rand() < pA else B

                reward = chosen.pull()

                # log trial data (now including raw Q, V, n for sequence building)
                self.log.append({
                    'block':      block,
                    'trial':      trial,
                    'options':    (A.name, B.name),
                    'utilities':  {A.name: uA, B.name: uB},
                    'ev':         {A.name: qA, B.name: qB},
                    'uncertainty_vals': {A.name: vA, B.name: vB},
                    'novelty_vals':     {A.name: nA, B.name: nB},
                    'chosen':     chosen.name,
                    'reward':     float(reward)
                })

    def summarize(self):
        total_reward = sum(entry['reward'] for entry in self.log)
        total_trials = len(self.log)
        print(f"Total rewards: {total_reward}")
        print(f"Total trials:  {total_trials}")

class LSTMAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # replace RNN with LSTM
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x, hc=None):
        out, (hn, cn) = self.rnn(x, hc)
        logits = self.fc(out)
        return logits, (hn, cn)

# --- Run simulation ---
task = BanditTask()
task.run()
task.summarize()

# --- Build 3-step [u0, u1, c, delta] sequence ---
def build_3step_sequence(log):

    # convert bandit to index for availability encoding
    bandit2index = {'A':0, 'B':1, 'C':2}
    X, Y, decision_mask = [], [], []
    for entry in log:
        # 1) stimulus step: what bandits are shown
        avail = [0.0, 0.0, 0.0]
        for name in entry['options']:
             avail[bandit2index[name]] = 1.0 #one hot encode trial available bandits
        
        X.append(avail + [0.0] + [0.0])  # +[go=0]+[reward=0]
        decision_mask.append(False)

        # 2) decision step
        X.append([0.0,0.0,0.0] + [1.0] + [0.0])
        bandit_index = bandit2index[entry['chosen']]
        Y.append(bandit_index)     # real label: which arm
        decision_mask.append(True)

        # 3) feedback step: no delta, network must learn to internally subtract its own expected value
        X.append([0.0,0.0,0.0] + [0.0] + [entry['reward']])
        decision_mask.append(False)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)        # [1, 3T, 5]
    Y = torch.tensor(Y, dtype=torch.long)
    decision_mask = torch.tensor(decision_mask, dtype=torch.bool).unsqueeze(0)  # [1, 3T]
    return X, Y, decision_mask

X_seq, Y_seq, seq_mask = build_3step_sequence(task.log)

# init and train RNN with input_size=4
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
        label_end = (b+1) * task.trials_per_block
        yb_block = Y_seq[label_start:label_end]  # [trials_per_block]

        optimizer.zero_grad()
        # reset hidden state by passing None
        logits, _ = model(xb, None)               # [1, 3*trials_per_block, output_size]
        # pick only decision steps
        logits_dec = logits[mb].view(-1, output_size)
        y_dec = yb_block
        loss = criterion(logits_dec, y_dec)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

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
    hidden_seq, _ = model.rnn(X_seq, None)   # [1, 3T, 64]
h = hidden_seq.squeeze(0).cpu().numpy()     # [3T, 64]

T = h.shape[0] // 3
stim_states = h[0::3]   # [T, 64]
decision_states = h[1::3]   # [T, 64]
feedback_states = h[2::3]   # [T, 64]

# prepare targets
u0 = np.array([e['utilities'][e['options'][0]] for e in task.log])
u1 = np.array([e['utilities'][e['options'][1]] for e in task.log])
u_diff = u0 - u1

bandit2index = {'A':0, 'B':1, 'C':2}
choice = np.array([bandit2index[e['chosen']] for e in task.log])
rpe = np.array([e['reward'] - e['ev'][e['chosen']] for e in task.log])

# decode stimulus difference
lr = LinearRegression().fit(stim_states, u_diff)
print("Stimulus u_diff R2:", r2_score(u_diff, lr.predict(stim_states)))

# decode choice
clf = LogisticRegression(max_iter=200).fit(decision_states, choice)
print("Decision accuracy:", accuracy_score(choice, clf.predict(decision_states)))

# decode feedback RPE
lr2 = LinearRegression().fit(feedback_states, rpe)
print("Feedback RPE R2:", r2_score(rpe, lr2.predict(feedback_states)))

