"""A two-layer MLP that takes a 794-dim hinted MNIST input."""

import torch.nn as nn


class HintedMLP(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784 + 10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.network(x)

