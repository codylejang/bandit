import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim

class Bandit:
    def __init__(self, name, true_prob, decay_rate=0.9, uI=1.0, nI=5.0):
        self.name = name
        self.true_prob = true_prob
        self.decay_rate = decay_rate
        self.uI = uI
        self.nI = nI
        self.reset() #reset values when bandit created

    def reset(self):
        self.alpha = 1.0
        self.beta = 1.0
        self.seen = 0

    def pull(self): #generates probabilistic reward outcome after selection
        
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
    def __init__(self, n_blocks=20, trials_per_block=15):
        self.n_blocks         = n_blocks
        self.trials_per_block = trials_per_block
        self.log              = []

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

                # log trial data
                self.log.append({
                    'block':      block,
                    'trial':      trial,
                    'options':    (A.name, B.name),
                    'utilities':  {A.name: uA, B.name: uB},
                    'chosen':     chosen.name,
                    'reward':     reward,
                    'ev':         chosen.expected_value(),
                    'uncertainty':chosen.uncertainty(),
                    'novelty':    chosen.novelty()
                })

    def summarize(self):
        total_reward = sum(entry['reward'] for entry in self.log)
        total_trials = len(self.log)
        print(f"Total rewards: {total_reward}")
        print(f"Total trials:  {total_trials}")

class RNNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        logits = self.fc(out)
        return logits, h


# --- Run simulation ---
task = BanditTask()
task.run()
task.summarize()

#flattening
actions = []
rewards = []
bandit2index = {'A':0, 'B':1, 'C':2 } #mapping actions to indexes
for entry in task.log:
    actions.append(bandit2index[entry['chosen']])
    rewards.append(entry['reward'])

# X (one-hot + reward)
# Y (next action)
X, Y = [], []
for t in range(1, len(actions)):
    prev_a = actions[t-1]
    onehot = [int(prev_a == i) for i in range(3)]
    X.append(onehot + [rewards[t-1]])  
    Y.append(actions[t])               

#mask handling, so we see which bandit is not selected during the 3 choose 2
mask_list = []
for entry in task.log:
    avail = [0,0,0]
    for name in entry['options']: #marking which bandits were available during given trial
        avail[bandit2index[name]] = 1
    mask_list.append(avail)

#convert to tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # [1, T, 4]
Y = torch.tensor(Y, dtype=torch.long).unsqueeze(0)   # [1, T]
mask = torch.tensor(mask_list, dtype=torch.bool).unsqueeze(0)
mask = mask[:, 1:, :]

assert mask.shape[1] == X.shape[1] == Y.shape[1]

#init and train RNN
input_size, hidden_size, output_size = 4, 16, 3
model     = RNNAgent(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

LARGE_NEG = -1e9

model.train()
'''
Per block, RNN learns dependencies and the weights for each bandit. 
Each block would reset the hidden states, while maintaining the learned weights.
Parallels Aquino: new block, new reward probabilities, drawn for each bandit, remained fixed for that block.
'''
for epoch in range(1, 31): #repeat the entire set of 20 blocks 30 times 
    total_loss = 0.0
    for b in range(task.n_blocks):
        #slice tensors to correspond to block
        start = b * task.trials_per_block
        end = (b+1) * task.trials_per_block
        xb = X[:, start:end, :]        # dim [1,15,4]
        yb = Y[:, start:end]           # dim [1,15]
        mb = mask[:, start:end, :]     # dim [1,15,3]

        optimizer.zero_grad()
        # reset hidden+cell by passing None for start of every block
        logits, _ = model(xb, None)    # [1,15,3]
        logits = logits.masked_fill(~mb, LARGE_NEG) #inverse mask unavail bandits so they get ~0 prob
        loss   = criterion(logits.squeeze(0), yb.squeeze(0))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() #add to epoch total
    print(f"Epoch {epoch:02d} — Avg Loss: {total_loss/task.n_blocks:.4f}")

# check training accuracy with actual decision
model.eval()
with torch.no_grad():
    logits, _ = model(X, None)        # [1, T-1, 3]
    logits = logits.masked_fill(~mask, LARGE_NEG)
    preds  = logits[0].argmax(dim=-1)  # [T-1]
    acc    = (preds == Y.squeeze(0)).float().mean()
    print(f"Train accuracy (masked): {acc:.3%}")
