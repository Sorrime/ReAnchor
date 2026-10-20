"""Models for hinted MNIST: a plain MLP, and a LeNet that sees the hint late."""

import torch
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


class HintedLeNet(nn.Module):
    """LeNet whose convolutions see only the image; the hint is joined later.

    Input is the flat 794-dim vector from HintedMNIST. It gets split back into
    (N, 1, 28, 28) image + (N, 10) hint. The conv stack is exactly the LeNet
    from the practice notebook: 28 -> 14 -> 10 -> 5, flattened to 400. Only
    then is the hint concatenated, giving 410, and fed to the classifier.

    Keeping the hint out of the conv stack means the image features are learned
    independently of the hint, so a difference between groups A/B/C can only
    come from the classifier's use of the hint.
    """

    def __init__(self, hint_size: int = 10, hidden_size: int = 120):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 + hint_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 84), nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.apply(init_weights)

    def forward(self, x):
        image = x[:, :784].reshape(-1, 1, 28, 28)
        hint = x[:, 784:]

        features = self.conv(image)
        merged = torch.cat([features, hint], dim=1)
        return self.classifier(merged)


def init_weights(m):
    """xavier uniform for conv/linear layers, straight from the notebook."""
    if type(m) in (nn.Conv2d, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

