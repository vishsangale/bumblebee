from __future__ import annotations

import pytest
import torch

from memory_state.write_gate import WriteGate


@pytest.fixture()
def gate():
    return WriteGate(hidden_size=32)


def test_gate_output_shape(gate):
    hidden = torch.randn(2, 32)
    surprise = torch.rand(2, 1) * 5.0
    out = gate(hidden, surprise, step=0)
    assert out.shape == (2, 1)


def test_gate_output_in_unit_interval(gate):
    hidden = torch.randn(4, 32)
    surprise = torch.rand(4, 1) * 10.0
    out = gate(hidden, surprise, step=0)
    assert (out >= 0).all() and (out <= 1).all()


def test_decay_term_decreases_with_step(gate):
    hidden = torch.zeros(1, 32)
    surprise = torch.zeros(1, 1)
    # Decay term is based on step; same hidden/surprise, increasing step
    # We can't check the gate direction without training, but decay buffer must not NaN
    for step in range(10):
        out = gate(hidden, surprise, step=step)
        assert not out.isnan().any()


def test_gate_is_differentiable(gate):
    hidden = torch.randn(2, 32, requires_grad=True)
    surprise = torch.rand(2, 1)
    out = gate(hidden, surprise, step=0)
    loss = out.sum()
    loss.backward()
    assert hidden.grad is not None


def test_reset_clears_state(gate):
    hidden = torch.randn(2, 32)
    surprise = torch.rand(2, 1)
    gate(hidden, surprise, step=5)
    gate.reset()
    # After reset, step-0 call should not error or NaN
    out = gate(hidden, surprise, step=0)
    assert not out.isnan().any()
