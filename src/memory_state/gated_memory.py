from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from memory_state.titans_mac import TitansMACMemory
from memory_state.write_gate import WriteGate


class GatedTitansMAC(nn.Module):
    """
    Titans MAC memory module with a learned write gate.

    Forward pass per token:
      1. Read enriched context from memory (no side effects)
      2. Compute Titans surprise update for this token
      3. Apply gate: effective_update = gate_t × surprise_update_t
      4. Write gated update to memory
      5. Return hidden + memory_read as enriched representation

    The gate modulates the magnitude of the Titans write — it does NOT
    replace the surprise-based update direction.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_mlp_size: int = 64,
        decay_init: float = 0.99,
    ) -> None:
        super().__init__()
        self.memory = TitansMACMemory(hidden_size, memory_mlp_size)
        self.gate = WriteGate(hidden_size, decay_init)
        self.out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.last_gate_value: Tensor | None = None

    def forward(self, hidden: Tensor, step: int) -> tuple[Tensor, Tensor]:
        """
        hidden: (B, H) — current token representation
        step:   int — position in sequence, for decay term
        Returns:
            out: (B, H) — hidden enriched with memory context
            assoc_loss: scalar tensor — add to outer LM loss to train W_K / W_V

        Write order: compute_surprise → gate → apply_update(gate).
        The gate multiplies the update BEFORE it is applied to memory weights,
        giving a faithful implementation of: update_t = gate_t × surprise_update_t.
        """
        # Step 1: Read — W_K in outer grad path; W1/W2 via snapshot (safe from in-place)
        mem_ctx = self.memory.read(hidden)  # (B, H)

        # Step 2: Compute surprise, cache inner gradients, get outer assoc loss
        surprise, assoc_loss = self.memory.compute_surprise(hidden)  # (B, 1), scalar

        # Step 3: Gate — conditioned on content and surprise
        gate_val = self.gate(hidden, surprise, step)  # (B, 1) in [0, 1]
        self.last_gate_value = gate_val

        # Step 4: Apply gated update — gate multiplies the momentum contribution
        self.memory.apply_update(gate_val)

        combined = torch.cat([hidden, gate_val * mem_ctx], dim=-1)  # (B, 2H)
        return self.out_proj(combined), assoc_loss  # (B, H), scalar

    def reset(self) -> None:
        self.memory.reset()
        self.gate.reset()
        self.last_gate_value = None
