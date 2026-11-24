"""Tests for the Lightning training wrapper."""

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from jupyter_ssc_compact_course.model import SimpleNet
from jupyter_ssc_compact_course.training import LitClassifier


def random_batch(n=16):
    """Random images and labels shaped like MNIST."""
    return torch.rand(n, 1, 28, 28), torch.randint(0, 10, (n,))


def test_training_step_returns_scalar_loss():
    lit_model = LitClassifier(SimpleNet())

    loss = lit_model.training_step(random_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_can_overfit_a_single_batch():
    """Sanity check: if the model can't memorise 16 examples, something is broken."""
    torch.manual_seed(0)
    lit_model = LitClassifier(SimpleNet(), lr=1e-2)
    optimizer = lit_model.configure_optimizers()
    batch = random_batch()

    first_loss = lit_model.training_step(batch, 0).item()
    for _ in range(50):
        optimizer.zero_grad()
        lit_model.training_step(batch, 0).backward()
        optimizer.step()
    last_loss = lit_model.training_step(batch, 0).item()

    assert last_loss < first_loss / 10


def test_lightning_trainer_runs_end_to_end(tmp_path):
    """fast_dev_run pushes a single batch through fit and test, which catches wiring errors quickly."""
    loader = DataLoader(TensorDataset(*random_batch()), batch_size=8)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        logger=False,
    )

    lit_model = LitClassifier(SimpleNet())
    trainer.fit(lit_model, loader)
    results = trainer.test(lit_model, loader, verbose=False)

    assert 0.0 <= results[0]["test_acc"] <= 1.0
