import numpy as np
import pandas as pd
import random

class Bandit:
    def __init__(self, name, true_prob):
        self.name = name
        self.true_prob = true_prob
        self.reset() #reset values when bandit created

    def reset(self):
        self.n_chosen = 0
        self.n_wins = 0
        self.n_seen = 0

    def pull(self): #generates probabilistic reward outcome after selection
        self.n_chosen += 1
        win = np.random.rand() < self.true_prob
        if win:
            self.n_wins += 1
        return win

    def beta_params(self):
        '''
        Bayesian approach where:
            - alpha = n_wins + 1
            - beta  = (n_chosen - n_wins) + 1
        
            +1 count prevents division by 0, ensuring all bandits have initial uncertainty even if unchosen
        
        Beta distribution models:
            - The expected value: alpha / (alpha + beta)
            - The uncertainty (variance): (alpha * beta) / [ (alpha + beta)^2 * (alpha + beta + 1) ]
        '''
        alpha = self.n_wins + 1
        beta = self.n_chosen - self.n_wins + 1
        return alpha, beta

    def expected_value(self):
        a, b = self.beta_params()
        return a / (a + b)

    def uncertainty(self):
        a, b = self.beta_params()
        return (a * b) / (((a + b) ** 2) * (a + b + 1))

    def novelty(self):
        return 1 / (1 + self.n_seen)  # fewer views = more novel

class BanditTask:
    def __init__(self, n_blocks=20, trials_per_block=15):
        self.n_blocks = n_blocks
        self.trials_per_block = trials_per_block
        self.log = []

    def generate_bandits(self):
        # 3 bandits per block, with fixed but random reward probabilities
        
        # three random values between 0.2 and 0.8: the true reward probabilities for each bandit in this block
        output = []
        probs = np.round(np.random.uniform(0.2, 0.8, size=3), 2) 
        names = ['A', 'B', 'C']
        for i in range(len(names)):
            output.append(Bandit(names[i],probs[i]))
        return output

    def run(self):
        for block_num in range(self.n_blocks):
            bandits = self.generate_bandits()

            print(f"Block {block_num + 1}: {[f'{b.name}:{b.true_prob}' for b in bandits]}")

            for trial_num in range(self.trials_per_block):
                # sample of 2 out of 3 bandits
                options = random.sample(bandits, 2)
                for b in options:
                    b.n_seen += 1

                # arbitrary utility calculation as EV + Uncertainty + Novelty (simplified)
                
                #parameters that can be tuned 
                uI = 1.0   # uncertainty weight (positive = seeking, negative = aversion)
                nI = 1.0   # novelty weight
                
                utilities = []
                for b in options:
                    utilities.append(b.expected_value() + uI * b.uncertainty() + nI * b.novelty())
                
                # Softmax over utility (can be manipulated or changed)
                beta = 3.0  # decision inverse temperature
                probs = np.exp(beta * np.array(utilities))
                #normalization
                probs /= probs.sum()

                choice_idx = np.random.choice([0, 1], p=probs)
                chosen = options[choice_idx]
                reward = chosen.pull()

                #log set of various vars for each block
                self.log.append({
                    'block': block_num,
                    'trial': trial_num,
                    'bandits': [b.name for b in options],
                    'utilities': dict(zip([b.name for b in options], utilities)),
                    'chosen': chosen.name,
                    'reward': reward,
                    'ev': chosen.expected_value(),
                    'uncertainty': chosen.uncertainty(),
                    'novelty': chosen.novelty()
                })

    def summarize(self):
        total_reward = sum([entry['reward'] for entry in self.log])
        print(f"Total rewards: {total_reward}")
        print(f"Total trials: {len(self.log)}")

# --- Run the simulation ---
task = BanditTask()
task.run()
task.summarize()