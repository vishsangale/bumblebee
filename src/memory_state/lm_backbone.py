from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from memory_state.gated_memory import GatedTitansMAC


@dataclass
class MemoryTransformerConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ffn: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1
    memory_mlp_size: int = 64
    memory_every_n_layers: int = 2  # insert GatedTitansMAC every N layers
    memory_decay_init: float = 0.99


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MemoryTransformerConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) / scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        att = self.dropout(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MemoryTransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ffn),
            nn.GELU(),
            nn.Linear(cfg.d_ffn, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MemoryTransformer(nn.Module):
    """
    GPT-style transformer with GatedTitansMAC memory inserted every N layers.

    Memory modules process one token at a time (sequential within the sequence
    dimension) to preserve causal semantics. The enriched representation
    is added to the hidden state before the next transformer block.
    """

    def __init__(self, cfg: MemoryTransformerConfig, use_memory: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_memory = use_memory

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Memory modules at every N-th layer (0-indexed: layers 0, N, 2N, ...)
        if use_memory:
            self.memory_modules = nn.ModuleList(
                [
                    GatedTitansMAC(cfg.d_model, cfg.memory_mlp_size, cfg.memory_decay_init)
                    for i in range(cfg.n_layers)
                    if i % cfg.memory_every_n_layers == 0
                ]
            )
            # Map layer index → memory module index
            self._layer_to_memory = {
                i: idx
                for idx, i in enumerate(
                    j for j in range(cfg.n_layers) if j % cfg.memory_every_n_layers == 0
                )
            }
        else:
            self.memory_modules = nn.ModuleList()
            self._layer_to_memory = {}

        # Storage for last-forward gate activations (for interpretability)
        self._gate_activations: list[Tensor] | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        input_ids: (B, T) — token indices
        Returns: logits (B, T, V)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        if self.use_memory:
            self._gate_activations = [[] for _ in self.memory_modules]

        for layer_idx, block in enumerate(self.blocks):
            # Memory read+write before attention (per-token, sequential)
            if self.use_memory and layer_idx in self._layer_to_memory:
                mem_idx = self._layer_to_memory[layer_idx]
                mem_module = self.memory_modules[mem_idx]
                enriched = torch.zeros_like(x)
                for t in range(T):
                    token_h = x[:, t, :]  # (B, d_model)
                    # forward() calls: read → compute_surprise → gate → apply_update
                    mem_out = mem_module(token_h, step=t)  # (B, d_model)
                    enriched[:, t, :] = mem_out
                    # Gate activations are already computed inside forward();
                    # re-read from the gate's last input for logging
                    if self._gate_activations is not None:
                        with torch.no_grad():
                            surprise = mem_module.memory.read(token_h).norm(dim=-1, keepdim=True)
                            gate_val = mem_module.gate(token_h.detach(), surprise, step=t)
                            self._gate_activations[mem_idx].append(gate_val.detach())
                x = x + enriched

            x = block(x)

        x = self.ln_f(x)
        return self.head(x)

    def reset_memory(self) -> None:
        for mem in self.memory_modules:
            mem.reset()
        self._gate_activations = None

    def get_gate_activations(self) -> list[Tensor] | None:
        """
        Returns list of gate activation tensors, one per memory module.
        Each tensor is (T, B, 1) — time steps stacked.
        Only populated after a forward() call with use_memory=True.
        """
        if self._gate_activations is None:
            return None
        return [torch.stack(acts, dim=0) if acts else None for acts in self._gate_activations]
