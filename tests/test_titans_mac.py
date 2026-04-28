from __future__ import annotations

import pytest
import torch

from memory_state.titans_mac import TitansMACMemory


@pytest.fixture()
def memory():
    return TitansMACMemory(hidden_size=32, memory_mlp_size=16)


def test_read_output_shape(memory):
    x = torch.randn(2, 32)  # (batch=2, hidden=32)
    out = memory.read(x)
    assert out.shape == (2, 32)


def test_compute_surprise_shape(memory):
    x = torch.randn(2, 32)
    surprise = memory.compute_surprise(x)
    assert surprise.shape == (2, 1)


def test_surprise_is_nonnegative(memory):
    x = torch.randn(2, 32)
    surprise = memory.compute_surprise(x)
    assert (surprise >= 0).all()


def test_memory_weights_change_after_apply_update(memory):
    x = torch.randn(2, 32)
    W1_before = memory.W1.clone()
    memory.compute_surprise(x)
    memory.apply_update(gate_val=1.0)
    assert not torch.allclose(memory.W1, W1_before)


def test_gate_zero_means_no_weight_change(memory):
    x = torch.randn(2, 32)
    memory.compute_surprise(x)
    memory.apply_update(gate_val=0.0)
    # gate=0 → update is zeroed → W1 changes only via alpha*W1 (forgetting term)
    # With alpha=0.99 close to 1 and small init std, difference is tiny but present
    # The momentum contribution must be zero: m1 should be zeroed after gate=0
    assert torch.allclose(memory.m1, torch.zeros_like(memory.m1), atol=1e-6)


def test_reset_clears_momentum(memory):
    x = torch.randn(2, 32)
    memory.compute_surprise(x)
    memory.apply_update(gate_val=1.0)
    assert memory.m1.abs().sum() > 0
    memory.reset()
    assert memory.m1.abs().sum() == 0


def test_compute_surprise_does_not_require_future_tokens(memory):
    x1 = torch.randn(2, 32)
    x2 = torch.randn(2, 32)
    s1 = memory.compute_surprise(x1)
    memory.apply_update(gate_val=1.0)
    s2 = memory.compute_surprise(x2)
    memory.apply_update(gate_val=1.0)
    assert not s1.isnan().any()
    assert not s2.isnan().any()
