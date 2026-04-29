from __future__ import annotations

import pytest
import torch

from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig, _mac_causal_mask


@pytest.fixture()
def small_cfg():
    return MemoryTransformerConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ffn=128,
        max_seq_len=64,
        memory_mlp_size=16,
        memory_layer=0,
    )


def test_forward_output_shape_with_memory(small_cfg):
    model = MemoryTransformer(small_cfg, use_memory=True)
    ids = torch.randint(0, 256, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 256)


def test_forward_output_shape_without_memory(small_cfg):
    model = MemoryTransformer(small_cfg, use_memory=False)
    ids = torch.randint(0, 256, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 256)


def test_memory_state_changes_across_calls(small_cfg):
    model = MemoryTransformer(small_cfg, use_memory=True)
    ids = torch.randint(0, 256, (1, 8))
    W1_before = model.memory_modules[0].memory.W1.clone()
    model(ids)
    assert not torch.allclose(model.memory_modules[0].memory.W1, W1_before)


def test_reset_memory_clears_state(small_cfg):
    model = MemoryTransformer(small_cfg, use_memory=True)
    ids = torch.randint(0, 256, (1, 8))
    model(ids)
    W1_after = model.memory_modules[0].memory.W1.clone()
    model.reset_memory()
    # After reset, W1 should differ from post-forward state
    assert not torch.allclose(model.memory_modules[0].memory.W1, W1_after)


def test_causal_attention_mask(small_cfg):
    model = MemoryTransformer(small_cfg, use_memory=False)
    model.eval()
    # Changing token at position 5 should NOT affect logits at positions 0-4
    ids1 = torch.randint(0, 256, (1, 8))
    ids2 = ids1.clone()
    ids2[0, 5] = (ids2[0, 5] + 1) % 256
    logits1 = model(ids1)
    logits2 = model(ids2)
    assert torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-4)


def test_gate_activations_returned_when_requested(small_cfg):
    model = MemoryTransformer(small_cfg, use_memory=True)
    ids = torch.randint(0, 256, (1, 8))
    model(ids)
    activations = model.get_gate_activations()
    assert activations is not None
    assert len(activations) == 1  # single MAC module per paper baseline


def test_mac_causal_mask_structure():
    """Verify MAC causal mask has correct block structure (lucidrains approach).

    For [h_1..h_T, x_1..x_T]:
      h-h block: fully visible (h tokens see all h tokens)
      h-x block: fully masked (h tokens don't see x tokens)
      x-h block: fully visible (x tokens see all h tokens)
      x-x block: upper-triangular (x tokens attend causally)
    """
    T = 4
    mask = _mac_causal_mask(T, T, torch.device("cpu"))
    assert mask.shape == (2 * T, 2 * T)

    # h-h: rows 0..T-1, cols 0..T-1 — fully unmasked
    assert not mask[:T, :T].any(), "h tokens must see all h tokens"

    # h-x: rows 0..T-1, cols T..2T-1 — fully masked
    assert mask[:T, T:].all(), "h tokens must not see x tokens"

    # x-h: rows T..2T-1, cols 0..T-1 — fully unmasked
    assert not mask[T:, :T].any(), "x tokens must see all h tokens"

    # x-x: rows T..2T-1, cols T..2T-1 — upper triangular only
    x_x = mask[T:, T:]
    expected = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.equal(x_x, expected), "x-x block must be causal (upper triangular)"


def test_mac_memory_causality():
    """Token at position t must not affect logits at positions < t in MAC mode."""
    cfg = MemoryTransformerConfig(
        vocab_size=256, d_model=32, n_heads=4, n_layers=2,
        d_ffn=64, max_seq_len=32, memory_mlp_size=8, memory_layer=0,
    )
    model = MemoryTransformer(cfg, use_memory=True)
    model.eval()

    ids1 = torch.randint(0, 256, (1, 8))
    ids2 = ids1.clone()
    ids2[0, 5] = (ids2[0, 5] + 1) % 256  # change token at position 5

    model.reset_memory()
    logits1 = model(ids1)
    model.reset_memory()
    logits2 = model(ids2)

    # Positions 0-4 must be unaffected by the change at position 5
    assert torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-4), (
        "MAC forward pass violated causality: future token affected earlier logits"
    )


# ── Stage 4: Output gate tests ────────────────────────────────────────────────

def test_output_gate_read_current_called_once_per_token(small_cfg):
    """Stage 4: read_current must be called exactly T times (once per token after write)."""
    model = MemoryTransformer(small_cfg, use_memory=True)
    model.eval()

    ids = torch.randint(0, 256, (1, 8))
    T = ids.shape[1]
    call_count = [0]
    original = model.memory_modules[0].memory.read_current

    def counting(query):
        call_count[0] += 1
        return original(query)

    model.memory_modules[0].memory.read_current = counting

    with torch.no_grad():
        model(ids)

    assert call_count[0] == T, f"expected {T} read_current calls, got {call_count[0]}"


def test_output_gate_affects_logits(small_cfg):
    """Stage 4: output gate must affect logits; identity gate (ones) must give different output."""
    model = MemoryTransformer(small_cfg, use_memory=True)
    model.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 8))

    with torch.no_grad():
        logits_real = model(ids).clone()

    model.reset_memory()
    model.memory_modules[0].memory.read_current = lambda q: torch.ones_like(q)

    with torch.no_grad():
        logits_identity = model(ids)

    assert not torch.allclose(logits_real, logits_identity, atol=1e-4), (
        "output gate must affect logits; same result with ones implies gate is not applied"
    )


def test_output_gate_reads_post_write_memory(small_cfg):
    """Stage 4: read_current must be called after apply_update (W1 must differ from M_0 snapshot)."""
    model = MemoryTransformer(small_cfg, use_memory=True)
    model.eval()

    mem = model.memory_modules[0].memory
    w1_equals_snap_at_read: list[bool] = []
    original = mem.read_current

    def checking(query):
        snap = mem._W1_snap
        w1_equals_snap_at_read.append(snap is not None and torch.equal(mem.W1, snap))
        return original(query)

    mem.read_current = checking

    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        model(ids)

    assert len(w1_equals_snap_at_read) == ids.shape[1]
    assert not any(w1_equals_snap_at_read), (
        "read_current must be called after writes; W1 matched snapshot, meaning reads happened before writes"
    )
