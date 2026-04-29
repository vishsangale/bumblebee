from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from memory_state.gated_memory import GatedTitansMAC
from memory_state.titans_mac import TitansMACMemory


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
    memory_layer: int = 0  # which transformer layer hosts the MAC module
    memory_decay_init: float = 0.99


def _mac_causal_mask(T_p: int, T: int, device: torch.device) -> Tensor:
    """MAC causal mask for [h_1..h_Tp, x_1..x_T] attention (lucidrains approach).

    h tokens see all h tokens and are blocked from all x tokens.
    x tokens see all h tokens and attend causally within the x block.
    """
    total = T_p + T
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[:T_p, T_p:] = True  # h tokens blocked from x tokens
    mask[T_p:, T_p:] = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )
    return mask


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MemoryTransformerConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, prefix: Tensor | None = None) -> Tensor:
        """
        x: (B, T, C) — content tokens (already layer-normed by the block).
        prefix: (B, T_p, C) — optional h-token prefix (no norm, no pos emb).
                If given, attention runs over [prefix, x] with MAC causal mask
                and returns only the x portion (B, T, C).
        """
        if prefix is None:
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
        else:
            # MAC mode: prepend h-token prefix, use MAC causal mask, return x portion
            B, T_p, C = prefix.shape
            T = x.shape[1]
            x_full = torch.cat([prefix, x], dim=1)  # (B, T_p + T, C)
            T_all = T_p + T
            q, k, v = self.qkv(x_full).split(C, dim=-1)
            q = q.view(B, T_all, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T_all, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T_all, self.n_heads, self.head_dim).transpose(1, 2)
            scale = math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) / scale
            mask = _mac_causal_mask(T_p, T, x.device)
            att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            att = self.dropout(F.softmax(att, dim=-1))
            out = (att @ v).transpose(1, 2).contiguous().view(B, T_all, C)
            out = self.out(out)
            return out[:, T_p:, :]  # (B, T, C) — content tokens only


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
    GPT-style transformer with a single Titans MAC module at cfg.memory_layer.

    MAC integration (Memory as a Context, Titans paper Sections 3-4):
      1. Parallel read: h_t = M_0(W_Q(x_t)) for all t from M_0 snapshot (no pos emb)
      2. MAC attention: causal attention over [h_1..h_T, x_1..x_T] → y_1..y_T
      3. Sequential write: for each t, write y_t to memory using data-dependent
         η_t/θ_t/α_t scalars and the learned write gate
      4. FFN: complete the transformer block

    Non-memory layers are standard causal transformer blocks.

    Parallel-approximation note: all reads use M_0 (sequence-start snapshot)
    rather than M_{t-1} as the paper specifies. This is the standard tractable
    approximation used by reference implementations.
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

        # Single MAC module at cfg.memory_layer
        if use_memory:
            self.memory_modules = nn.ModuleList(
                [GatedTitansMAC(cfg.d_model, cfg.memory_mlp_size, cfg.memory_decay_init)]
            )
        else:
            self.memory_modules = nn.ModuleList()

        self._gate_activations: list[list[Tensor]] | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        # Override scalar projection biases after global init zeros them.
        # sigmoid(0)=0.5 → α_t=0.5 per token = catastrophic forgetting.
        for module in self.modules():
            if isinstance(module, TitansMACMemory):
                module._init_scalar_projections()

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        input_ids: (B, T) — token indices
        Returns: logits (B, T, V)

        Side effect: sets self.last_assoc_loss — accumulated associative loss
        over the sequence. Add a weighted copy to the outer task loss to train
        W_K and W_V (paper outer-loop objective). Zero for use_memory=False models.
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        if self.use_memory:
            self._gate_activations = [[]]  # one list for the single memory module

        self.last_assoc_loss: Tensor = torch.zeros(1, device=input_ids.device)

        for layer_idx, block in enumerate(self.blocks):
            if self.use_memory and layer_idx == self.cfg.memory_layer:
                mem_module = self.memory_modules[0]

                # ── Step 1: Parallel read from M_0 snapshot ───────────────────
                # h_t = M_0(W_Q(x_t)) for all t; no positional embedding on h tokens.
                h_tokens = torch.stack(
                    [mem_module.memory.read(x[:, t, :]) for t in range(T)], dim=1
                )  # (B, T, H)

                # ── Step 2: MAC attention over [h_tokens, ln1(x)] ─────────────
                ln1_x = block.ln1(x)  # (B, T, H) — layer-normed content tokens
                attn_out = block.attn(ln1_x, prefix=h_tokens)  # (B, T, H)
                x = x + attn_out  # residual add

                # ── Step 3: Sequential write using post-attention y_t ─────────
                for t in range(T):
                    y_t = x[:, t, :]
                    surprise, assoc_loss = mem_module.memory.compute_surprise(y_t)
                    gate_val = mem_module.gate(y_t, surprise, t)
                    mem_module.last_gate_value = gate_val
                    mem_module.memory.apply_update(gate_val)
                    self.last_assoc_loss = self.last_assoc_loss + assoc_loss
                    if self._gate_activations is not None:
                        self._gate_activations[0].append(gate_val.detach())

                # ── Step 4: FFN portion of the block ──────────────────────────
                x = x + block.ffn(block.ln2(x))
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def reset_memory(self) -> None:
        """Reset all memory modules (called at sequence boundaries)."""
        if hasattr(self, "memory_modules"):
            for mem in self.memory_modules:
                mem.reset()

    def get_gate_activations(self) -> list[Tensor | None] | None:
        """
        Returns list of gate activation tensors, one per memory module.
        Each tensor is (T, B, 1) — time steps stacked.
        Only populated after a forward() call with use_memory=True.
        """
        if self._gate_activations is None:
            return None
        return [
            torch.stack(acts, dim=0) if acts else None
            for acts in self._gate_activations
        ]
