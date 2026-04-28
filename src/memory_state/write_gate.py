from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class WriteGate(nn.Module):
    """
    Learned causal write gate for Titans MAC memory.

    gate_t = σ(W_g @ [h_t ; log(surprise_t + ε) ; decay_t])

    All three inputs are causally available at write time:
      h_t:         current token hidden state (content relevance)
      surprise_t:  ‖∇L_t‖ from TitansMACMemory.write() (write magnitude signal)
      decay_t:     exp(-λ * step) — exponential decay reset per-sequence

    The gate is a scalar in [0, 1] that multiplies the Titans surprise update.
    Setting gate=0 recovers no-write; gate=1 recovers pure Titans.

    Differentiation from LSTM input gate: LSTM gate has no surprise-norm input.
    Differentiation from Titans α_t/η_t: those are fixed scalars, not
    content-conditioned learned functions.
    """

    def __init__(self, hidden_size: int, decay_init: float = 0.99) -> None:
        super().__init__()
        # Input: [h_t (H), log_surprise (1), decay (1)] → scalar gate
        self.linear = nn.Linear(hidden_size + 2, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 1.0)  # init: gate ≈ 0.73 (σ(1))

        # Learnable decay rate (trained by outer optimizer)
        self.log_decay = nn.Parameter(torch.tensor(math.log(-math.log(decay_init))))
        self._eps = 1e-6
        self.force_open = False

    @property
    def decay_rate(self) -> float:
        return float(torch.exp(-self.log_decay.exp()))

    def forward(self, hidden: Tensor, surprise: Tensor, step: int) -> Tensor:
        """
        hidden:   (B, H)
        surprise: (B, 1) — from TitansMACMemory.write(), nonnegative
        step:     int, position in sequence (0-indexed)
        Returns:  gate (B, 1) in [0, 1]
        """
        batch = hidden.shape[0]
        if self.force_open:
            return torch.ones((batch, 1), dtype=hidden.dtype, device=hidden.device)

        log_surprise = torch.log(surprise + self._eps)  # (B, 1)
        decay_base = torch.exp(-self.log_decay.exp()).to(dtype=hidden.dtype, device=hidden.device)
        decay = torch.full(
            (batch, 1),
            1.0,
            dtype=hidden.dtype,
            device=hidden.device,
        ) * decay_base.pow(step)
        features = torch.cat([hidden, log_surprise, decay], dim=-1)  # (B, H+2)
        return torch.sigmoid(self.linear(features))  # (B, 1)

    def reset(self) -> None:
        """No persistent state to reset; provided for interface consistency."""
        pass

    def set_force_open(self, enabled: bool) -> None:
        """Force the gate to output 1.0 for Titans-only ablations."""
        self.force_open = enabled
        for parameter in self.parameters():
            parameter.requires_grad_(not enabled)
