from __future__ import annotations

import pytest
import torch

from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig


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
        memory_every_n_layers=2,
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
    assert len(activations) == 2  # memory_every_n_layers=2, n_layers=4 → 2 modules
