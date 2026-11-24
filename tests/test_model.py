"""Tests for the SimpleNet model."""

import pytest
import torch

from jupyter_ssc_compact_course.model import SimpleNet


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_output_has_one_score_per_class(batch_size):
    model = SimpleNet()
    images = torch.rand(batch_size, 1, 28, 28)

    scores = model(images)

    assert scores.shape == (batch_size, 10)


def test_gradients_reach_all_parameters():
    model = SimpleNet()
    model(torch.rand(4, 1, 28, 28)).sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
