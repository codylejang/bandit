# agent.py
# vanilla RNN model with PyTorch

import torch
import torch.nn as nn

class RNNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNAgent, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        # x shape: [batch, sequence_len, input_size]
        out, h = self.rnn(x, h)
        logits = self.fc(out)
        return logits, h