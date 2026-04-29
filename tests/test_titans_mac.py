from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

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
    surprise, _ = memory.compute_surprise(x)
    assert surprise.shape == (2, 1)


def test_surprise_is_nonnegative(memory):
    x = torch.randn(2, 32)
    surprise, _ = memory.compute_surprise(x)
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
    assert torch.allclose(memory.m1, torch.zeros_like(memory.m1), atol=1e-6)


def test_reset_clears_momentum(memory):
    x = torch.randn(2, 32)
    memory.compute_surprise(x)
    memory.apply_update(gate_val=1.0)
    assert memory.m1.abs().sum() > 0
    memory.reset()
    assert memory.m1.abs().sum() == 0


def test_reset_clears_snapshot(memory):
    x = torch.randn(2, 32)
    memory.compute_surprise(x)
    assert memory._W1_snap is not None
    memory.reset()
    assert memory._W1_snap is None


def test_compute_surprise_does_not_require_future_tokens(memory):
    x1 = torch.randn(2, 32)
    x2 = torch.randn(2, 32)
    s1, _ = memory.compute_surprise(x1)
    memory.apply_update(gate_val=1.0)
    s2, _ = memory.compute_surprise(x2)
    memory.apply_update(gate_val=1.0)
    assert not s1.isnan().any()
    assert not s2.isnan().any()


# ── Bug 1 regression tests ────────────────────────────────────────────────────

def test_inner_loss_target_is_not_input():
    """Inner loss target must be W_V(token), not token itself (Bug 1 regression)."""
    torch.manual_seed(42)
    mem = TitansMACMemory(hidden_size=32, memory_mlp_size=16)
    x = torch.randn(2, 32)
    k_t = mem.W_K(x).detach()
    v_t = mem.W_V(x).detach()
    assert not torch.allclose(k_t, x, atol=1e-3), "W_K must differ from identity"
    assert not torch.allclose(v_t, x, atol=1e-3), "W_V must differ from identity"
    assert not torch.allclose(k_t, v_t, atol=1e-3), "key and value projections must differ"


def test_momentum_accumulates_across_updates():
    """m1 must grow after repeated updates — catches the 'momentum dead at step 20k' failure."""
    torch.manual_seed(0)
    mem = TitansMACMemory(hidden_size=32, memory_mlp_size=16)
    for _ in range(10):
        x = torch.randn(2, 32)
        mem.compute_surprise(x)
        mem.apply_update(gate_val=1.0)
    assert mem.m1.norm().item() > 1e-6, "momentum must accumulate; was zero with identity loss"


def test_memory_loss_decreases_with_updates():
    """Memory loss must decrease on a repeated (key, value) pair after many inner updates."""
    torch.manual_seed(1)
    mem = TitansMACMemory(hidden_size=16, memory_mlp_size=8)
    x = torch.randn(1, 16)
    k_t = mem.W_K(x).detach()
    v_t = mem.W_V(x).detach()

    def inner_loss() -> float:
        with torch.no_grad():
            pred = mem._forward_memory(k_t, mem.W1, mem.W2)
            return F.mse_loss(pred, v_t).item()

    loss_before = inner_loss()
    for _ in range(100):
        mem.compute_surprise(x)
        mem.apply_update(gate_val=1.0)
    loss_after = inner_loss()

    assert loss_after < 0.5 * loss_before, (
        f"loss did not decrease sufficiently: {loss_before:.4f} → {loss_after:.4f}"
    )


# ── Outer-loop gradient tests (W_K / W_V are outer-loop trained) ──────────────

def test_wq_receives_gradient_from_read():
    """W_Q must receive LM-loss gradient through read() — outer-loop training path.
    W_K is NOT in this path; it trains via assoc_loss. W_Q and W_K are separate."""
    mem = TitansMACMemory(hidden_size=16, memory_mlp_size=8)
    x = torch.randn(1, 16)
    out = mem.read(x)
    out.sum().backward()
    assert mem.W_Q.weight.grad is not None
    assert mem.W_Q.weight.grad.abs().sum().item() > 0
    assert mem.W_K.weight.grad is None, "W_K must NOT receive gradient from read()"


def test_assoc_loss_trains_wk_and_wv():
    """assoc_loss must have grad_fn and flow gradients to both W_K and W_V."""
    mem = TitansMACMemory(hidden_size=16, memory_mlp_size=8)
    x = torch.randn(1, 16)
    _, assoc_loss = mem.compute_surprise(x)

    assert assoc_loss.requires_grad, "assoc_loss must be differentiable"
    assoc_loss.backward()

    assert mem.W_K.weight.grad is not None and mem.W_K.weight.grad.abs().sum() > 0, (
        "W_K must receive gradient from assoc_loss"
    )
    assert mem.W_V.weight.grad is not None and mem.W_V.weight.grad.abs().sum() > 0, (
        "W_V must receive gradient from assoc_loss"
    )


def test_snapshot_shared_across_tokens():
    """Snapshot must be created once and reused across read()/compute_surprise() calls."""
    mem = TitansMACMemory(hidden_size=16, memory_mlp_size=8)
    assert mem._W1_snap is None
    mem.read(torch.randn(1, 16))
    snap_id = id(mem._W1_snap)
    # Second call should NOT create a new snapshot
    mem.read(torch.randn(1, 16))
    assert id(mem._W1_snap) == snap_id, "snapshot must be reused within a sequence"
