"""Minimal Lightning training setup for MNIST digit classification."""

import lightning as L
import torch
from torch import nn


class LitClassifier(L.LightningModule):
    """Wraps any classifier model; Lightning's Trainer runs the loop."""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = self.loss_fn(self(images), labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        acc = (self(images).argmax(dim=1) == labels).float().mean()
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
