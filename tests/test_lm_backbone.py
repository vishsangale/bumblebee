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
