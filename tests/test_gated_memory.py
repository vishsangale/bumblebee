from __future__ import annotations

import pytest
import torch

from memory_state.gated_memory import GatedTitansMAC


@pytest.fixture()
def gated():
    return GatedTitansMAC(hidden_size=32, memory_mlp_size=16)


def test_forward_output_shape(gated):
    hidden = torch.randn(2, 32)
    out = gated(hidden, step=0)
    assert out.shape == (2, 32)


def test_memory_not_updated_when_gate_zero(gated):
    """Force gate to 0 by zeroing its linear weights; verify momentum is zeroed."""
    torch.nn.init.zeros_(gated.gate.linear.weight)
    torch.nn.init.constant_(gated.gate.linear.bias, -10.0)  # σ(-10) ≈ 0

    hidden = torch.randn(2, 32)
    gated(hidden, step=0)
    # With gate=0, momentum should be zeroed
    assert torch.allclose(gated.memory.m1, torch.zeros_like(gated.memory.m1), atol=1e-5)
    assert torch.allclose(gated.memory.m2, torch.zeros_like(gated.memory.m2), atol=1e-5)


def test_memory_updated_when_gate_one(gated):
    """Force gate to 1; verify W1 changes (same as pure Titans write)."""
    torch.nn.init.zeros_(gated.gate.linear.weight)
    torch.nn.init.constant_(gated.gate.linear.bias, 10.0)  # σ(10) ≈ 1

    W1_before = gated.memory.W1.clone()
    hidden = torch.randn(2, 32)
    gated(hidden, step=0)
    assert not torch.allclose(gated.memory.W1, W1_before, atol=1e-5)


def test_reset_clears_memory(gated):
    hidden = torch.randn(2, 32)
    gated(hidden, step=0)
    W1_after_write = gated.memory.W1.clone()
    gated.reset()
    # After reset, W1 should differ from post-write state
    assert not torch.allclose(gated.memory.W1, W1_after_write, atol=1e-5)


def test_forward_is_differentiable(gated):
    hidden = torch.randn(2, 32, requires_grad=True)
    out = gated(hidden, step=0)
    loss = out.sum()
    loss.backward()
    assert hidden.grad is not None
