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

        surprise, assoc_loss = memory.compute_surprise(token)
        gate = write_gate(hidden, surprise, step)
        memory.apply_update(gate)

    W_K and W_V are outer-loop (AdamW) trained parameters, as per the paper:
      - W_K gets gradients through read() → mem_ctx → LM loss
      - W_V gets gradients through assoc_loss → outer LM total loss

    Both paths share a per-sequence snapshot of W1/W2 (taken on the first
    read/compute_surprise call after reset()). The snapshot is a clone so
    in-place mutations in apply_update() cannot cause version-counter errors
    during the outer backward pass.

    Reference: Titans (arxiv:2501.00663), MAC variant.
    """

    def __init__(self, hidden_size: int, memory_mlp_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_mlp_size = memory_mlp_size

        # Outer-loop trained projections (Titans Eq. 12): token → key / value.
        # W_K is used in read() so it receives LM-loss gradients.
        # W_V is trained via the assoc_loss term added to the outer loss.
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)

        # Memory weights — buffers, updated by inner optimizer during forward
        self.register_buffer("W1", torch.empty(memory_mlp_size, hidden_size))
        self.register_buffer("W2", torch.empty(hidden_size, memory_mlp_size))
        # Momentum buffers
        self.register_buffer("m1", torch.zeros(memory_mlp_size, hidden_size))
        self.register_buffer("m2", torch.zeros(hidden_size, memory_mlp_size))

        self.eta: float = 0.1   # inner learning rate
        self.beta: float = 0.9  # momentum coefficient
        self.alpha: float = 0.99  # adaptive forgetting

        # Per-sequence snapshot of W1/W2 used for outer-loss gradient paths.
        # Created lazily on first read()/compute_surprise() after reset().
        # Cleared by reset() so each sequence gets a fresh snapshot.
        self._W1_snap: Tensor | None = None
        self._W2_snap: Tensor | None = None

        # Cached gradients between compute_surprise() and apply_update()
        self._cached_g1: Tensor | None = None
        self._cached_g2: Tensor | None = None

        self.reset()

    def _forward_memory(self, query: Tensor, W1: Tensor, W2: Tensor) -> Tensor:
        """query: (..., H) → (..., H) using given weight tensors."""
        return F.gelu(query @ W1.T) @ W2.T

    def _get_snap(self) -> tuple[Tensor, Tensor]:
        """Return (W1_snap, W2_snap) for the current sequence.

        Created once per sequence (after reset()) as a detached clone so
        in-place apply_update() calls cannot invalidate the outer autograd
        graph. One clone pair per sequence, shared across all T tokens.
        """
        if self._W1_snap is None:
            self._W1_snap = self.W1.detach().clone()
            self._W2_snap = self.W2.detach().clone()
        return self._W1_snap, self._W2_snap  # type: ignore[return-value]

    def read(self, query: Tensor) -> Tensor:
        """Read from memory without updating it. query: (B, H) → (B, H).

        The query is projected through W_K so that W_K receives outer LM-loss
        gradients. W1/W2 are accessed via the sequence snapshot (detached clone)
        to prevent version-counter errors from in-place apply_update() mutations.
        """
        W1_snap, W2_snap = self._get_snap()
        k_q = self.W_K(query)
        return self._forward_memory(k_q, W1_snap, W2_snap)

    def compute_surprise(self, token: Tensor) -> tuple[Tensor, Tensor]:
        """Compute surprise norm, cache inner gradients, and return outer assoc loss.

        Does NOT update W1/W2. Must be followed by apply_update().

        token: (B, H)
        Returns:
            surprise_norm: (B, 1) — detached, used as write-gate input.
            assoc_loss: scalar tensor with grad_fn — add to outer LM loss to
                        train W_K and W_V (Titans paper outer-loop objective).

        Inner loss (Titans Eq. 12): ‖M(k_t) − v_t‖², differentiated w.r.t.
        W1/W2 only (k_t and v_t are detached for the inner grad path).

        Outer assoc loss: same equation, but k_t = W_K(token) and
        v_t = W_V(token) are NOT detached, so W_K and W_V receive gradients
        when the caller adds assoc_loss to the outer task loss.
        """
        W1_snap, W2_snap = self._get_snap()

        # ── Inner gradient path: only W1/W2 receive gradients ────────────────
        with torch.enable_grad():
            W1 = self.W1.detach().requires_grad_(True)
            W2 = self.W2.detach().requires_grad_(True)
            k_inner = self.W_K(token).detach()
            v_inner = self.W_V(token).detach()
            pred_inner = self._forward_memory(k_inner, W1, W2)
            loss_inner = F.mse_loss(pred_inner, v_inner)
            g1, g2 = torch.autograd.grad(loss_inner, [W1, W2])

        self._cached_g1 = g1.detach()
        self._cached_g2 = g2.detach()

        # ── Outer assoc loss: W_K and W_V receive gradients ──────────────────
        k_t = self.W_K(token)            # outer graph — W_K grad
        v_t = self.W_V(token)            # outer graph — W_V grad
        pred_outer = self._forward_memory(k_t, W1_snap, W2_snap)
        assoc_loss = F.mse_loss(pred_outer, v_t)

        surprise = (g1.norm() + g2.norm()) / 2.0
        batch = token.shape[0]
        return surprise.expand(batch, 1).contiguous().detach(), assoc_loss

    def apply_update(self, gate_val: float | Tensor = 1.0) -> None:
        """Apply cached gradients scaled by gate_val to memory weights.

        Call compute_surprise() first.

        gate_val: scalar or (B, 1) tensor in [0, 1].
                  gate_val=1.0 → pure Titans update.
                  gate_val=0.0 → no update (momentum zeroed).
        """
        assert self._cached_g1 is not None, "call compute_surprise() before apply_update()"
        if isinstance(gate_val, Tensor):
            scale = float(gate_val.detach().mean().item())
        else:
            scale = float(gate_val)
        g1 = self._cached_g1.to(device=self.W1.device, dtype=self.W1.dtype)
        g2 = self._cached_g2.to(device=self.W2.device, dtype=self.W2.dtype)
        g1_norm = g1.norm()
        if g1_norm > 1.0:
            g1 = g1 / g1_norm
        g2_norm = g2.norm()
        if g2_norm > 1.0:
            g2 = g2 / g2_norm
        with torch.no_grad():
            self.m1.mul_(self.beta).sub_(g1, alpha=self.eta * scale)
            self.m2.mul_(self.beta).sub_(g2, alpha=self.eta * scale)
            self.W1.mul_(self.alpha).add_(self.m1)
            self.W2.mul_(self.alpha).add_(self.m2)
        self._cached_g1 = None
        self._cached_g2 = None

    def reset(self) -> None:
        """Reset memory weights, momentum, snapshot, and gradient cache."""
        nn.init.normal_(self.W1, std=0.02)
        nn.init.normal_(self.W2, std=0.02)
        self.m1.zero_()
        self.m2.zero_()
        self._W1_snap = None
        self._W2_snap = None
        self._cached_g1 = None
        self._cached_g2 = None
