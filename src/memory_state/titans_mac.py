from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TitansMACMemory(nn.Module):
    """
    Titans MAC (Memory as a Context) memory module.

    Memory is a two-layer MLP whose weights are updated online by a
    gradient step on a local associative loss. Weights are register_buffer
    tensors so the outer AdamW optimizer does not touch them.

    The write is split into two steps so GatedTitansMAC can insert the gate
    between computing the surprise and applying the update:

        surprise = memory.compute_surprise(token)   # no weight change
        gate = write_gate(hidden, surprise, step)
        memory.apply_update(gate)                   # gated weight change

    This gives a faithful implementation of gate_t × surprise_update_t —
    the gate multiplies the update BEFORE it is applied to the weights.

    Reference: Titans (arxiv:2501.00663), MAC variant.
    """

    def __init__(self, hidden_size: int, memory_mlp_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_mlp_size = memory_mlp_size

        # Memory weights — buffers, updated by inner optimizer during forward
        self.register_buffer("W1", torch.empty(memory_mlp_size, hidden_size))
        self.register_buffer("W2", torch.empty(hidden_size, memory_mlp_size))
        # Momentum buffers
        self.register_buffer("m1", torch.zeros(memory_mlp_size, hidden_size))
        self.register_buffer("m2", torch.zeros(hidden_size, memory_mlp_size))

        # Learnable inner learning rate (trained by outer optimizer)
        self.log_eta = nn.Parameter(torch.tensor(-2.3))  # init: eta ≈ 0.1
        self.beta = 0.9  # momentum coefficient
        self.alpha = 0.99  # adaptive forgetting

        # Cached gradients between compute_surprise() and apply_update()
        self._cached_g1: Tensor | None = None
        self._cached_g2: Tensor | None = None

        self.reset()

    @property
    def eta(self) -> float:
        return float(self.log_eta.exp())

    def _forward_memory(self, query: Tensor, W1: Tensor, W2: Tensor) -> Tensor:
        """query: (..., H) → (..., H) using given weight tensors."""
        return F.gelu(query @ W1.T) @ W2.T

    def read(self, query: Tensor) -> Tensor:
        """Read from memory without updating it. query: (B, H) → (B, H)."""
        with torch.no_grad():
            return self._forward_memory(query, self.W1, self.W2)

    def compute_surprise(self, token: Tensor) -> Tensor:
        """
        Compute surprise norm and cache gradients. Does NOT update W1/W2.
        Must be followed by apply_update() to complete the write step.

        token: (B, H)
        Returns: surprise_norm (B, 1) — detached, for use as gate input.
        """
        W1 = self.W1.detach().requires_grad_(True)
        W2 = self.W2.detach().requires_grad_(True)

        pred = self._forward_memory(token.detach(), W1, W2)
        loss = F.mse_loss(pred, token.detach())
        g1, g2 = torch.autograd.grad(loss, [W1, W2])

        self._cached_g1 = g1.detach()
        self._cached_g2 = g2.detach()

        surprise = (g1.detach().norm() + g2.detach().norm()) / 2.0
        batch = token.shape[0]
        return surprise.expand(batch, 1).contiguous().detach()

    def apply_update(self, gate_val: float | Tensor = 1.0) -> None:
        """
        Apply cached gradients scaled by gate_val to memory weights.
        Call compute_surprise() first.

        gate_val: scalar or (B, 1) tensor in [0, 1].
                  gate_val=1.0 → pure Titans update.
                  gate_val=0.0 → no update (momentum zeroed).
        """
        assert self._cached_g1 is not None, "call compute_surprise() before apply_update()"
        scale = gate_val if isinstance(gate_val, float) else float(gate_val.mean().item())
        eta = self.eta
        with torch.no_grad():
            self.m1.mul_(self.beta).sub_(self._cached_g1, alpha=eta * scale)
            self.m2.mul_(self.beta).sub_(self._cached_g2, alpha=eta * scale)
            self.W1.mul_(self.alpha).add_(self.m1)
            self.W2.mul_(self.alpha).add_(self.m2)
        self._cached_g1 = None
        self._cached_g2 = None

    def reset(self) -> None:
        """Reset memory weights, momentum, and gradient cache."""
        nn.init.normal_(self.W1, std=0.02)
        nn.init.normal_(self.W2, std=0.02)
        self.m1.zero_()
        self.m2.zero_()
        self._cached_g1 = None
        self._cached_g2 = None
