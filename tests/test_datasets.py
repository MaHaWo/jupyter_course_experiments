"""Tests for the MNIST dataset wrapper.

Unit tests should be fast and must not depend on the internet, so we replace the
real MNIST download with torchvision's FakeData (random images of the same shape).
"""

import pytest
import torch
from torchvision import datasets as tv_datasets

from jupyter_ssc_compact_course import datasets


def fake_mnist(root, train, download, transform):
    """Stand-in for torchvision.datasets.MNIST: 20 random 1x28x28 images, labels 0-9."""
    return tv_datasets.FakeData(size=20, image_size=(1, 28, 28), num_classes=10, transform=transform)


@pytest.fixture
def no_download(monkeypatch):
    """Swap out MNIST for the fake version for the duration of one test."""
    monkeypatch.setattr(datasets.datasets, "MNIST", fake_mnist)


def test_dataset_length(no_download):
    assert len(datasets.MNISTDataset()) == 20


def test_dataset_item_shape_and_label(no_download):
    image, label = datasets.MNISTDataset()[0]

    assert image.shape == (1, 28, 28)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0  # ToTensor scales pixels to [0, 1]
    assert 0 <= int(label) <= 9


def test_dataloaders_yield_batches(no_download):
    train_loader, test_loader = datasets.get_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))

    assert images.shape == (8, 1, 28, 28)
    assert labels.shape == (8,)
