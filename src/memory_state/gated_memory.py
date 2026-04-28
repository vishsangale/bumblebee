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

    def forward(self, hidden: Tensor, step: int) -> Tensor:
        """
        hidden: (B, H) — current token representation
        step:   int — position in sequence, for decay term
        Returns: (B, H) — hidden enriched with memory context

        Write order: compute_surprise → gate → apply_update(gate).
        The gate multiplies the update BEFORE it is applied to memory weights,
        giving a faithful implementation of: update_t = gate_t × surprise_update_t.
        """
        # Step 1: Read — no side effects on memory
        mem_ctx = self.memory.read(hidden)  # (B, H)

        # Step 2: Compute surprise and cache gradients (no weight change yet)
        surprise = self.memory.compute_surprise(hidden)  # (B, 1)

        # Step 3: Gate — uses surprise norm before it is applied
        gate_val = self.gate(hidden.detach(), surprise, step)  # (B, 1) in [0, 1]
        self.last_gate_value = gate_val

        # Step 4: Apply gated update — gate multiplies the momentum contribution
        self.memory.apply_update(gate_val)

        # Enrich hidden with memory context
        combined = torch.cat([hidden, mem_ctx], dim=-1)  # (B, 2H)
        return self.out_proj(combined)  # (B, H)

    def reset(self) -> None:
        self.memory.reset()
        self.gate.reset()
        self.last_gate_value = None
