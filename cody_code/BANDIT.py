import numpy as np
import pandas as pd
import random

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


# --- Run simulation ---
task = BanditTask()
task.run()
task.summarize()