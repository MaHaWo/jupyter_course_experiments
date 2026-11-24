"""Minimal model for MNIST digit classification."""

from torch import nn


class SimpleNet(nn.Module):
    """A tiny fully connected network: 784 pixels -> 128 hidden units -> 10 classes."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)
