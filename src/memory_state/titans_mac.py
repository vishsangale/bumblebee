from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _softclamp_grad(g: Tensor, max_norm: float = 1.0) -> Tensor:
    """Smooth gradient norm clamp (Titans paper). Tanh saturation instead of hard clip."""
    norm = g.norm()
    if norm < 1e-8:
        return g
    return g * (max_norm * torch.tanh(norm / max_norm) / norm)


class TitansMACMemory(nn.Module):
    """
    Titans MAC (Memory as a Context) memory module — paper-faithful implementation.

    Memory is a two-layer MLP whose weights are updated online by a
    gradient step on the associative loss (Titans Eq. 12):

        ℓ(M_{t-1}; x_t) = ‖ M(k_t) − v_t ‖²   where k_t = W_K(x_t), v_t = W_V(x_t)

    Inner update (Eqs. 9-10) with data-dependent scalars:
        S_t = η_t · S_{t-1} − θ_t · ∇ℓ        (momentum update)
        M_t = (1 − α_t) · M_{t-1} + S_t         (weight update with forgetting)

    η_t, θ_t, α_t are sigmoid outputs of learned linear projections applied to the
    current token. Gradient norm uses softclamp (tanh-based, smooth) instead of
    a hard clip. Both follow the paper exactly.

    Outer-loop trained projections W_K, W_V:
      - W_K gradient path: read() → LM loss
      - W_V gradient path: assoc_loss auxiliary term → outer loss

    Both read() and compute_surprise() share a per-sequence W1/W2 snapshot
    (one clone pair per memory module per sequence) so the outer backward is
    safe from apply_update()'s in-place buffer mutations.

    Reference: Titans (arxiv:2501.00663), MAC variant, Sections 3–4.
    """

    def __init__(self, hidden_size: int, memory_mlp_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_mlp_size = memory_mlp_size

        # Outer-loop trained projections (Eq. 12): token → key / value
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)

        # Data-dependent inner-loop scalars (Eqs. 9-10):
        #   η_t = sigmoid(W_eta · x_t)   — momentum coefficient
        #   θ_t = sigmoid(W_theta · x_t) — inner learning rate
        #   α_t = sigmoid(W_alpha · x_t) — forgetting rate
        # Biases set by _init_scalar_projections() to match stable defaults.
        self.eta_proj = nn.Linear(hidden_size, 1, bias=True)    # η_t
        self.theta_proj = nn.Linear(hidden_size, 1, bias=True)  # θ_t
        self.alpha_proj = nn.Linear(hidden_size, 1, bias=True)  # α_t

        # Memory weights — buffers updated by inner optimizer during forward
        self.register_buffer("W1", torch.empty(memory_mlp_size, hidden_size))
        self.register_buffer("W2", torch.empty(hidden_size, memory_mlp_size))
        # Momentum buffers (S_t in the paper)
        self.register_buffer("m1", torch.zeros(memory_mlp_size, hidden_size))
        self.register_buffer("m2", torch.zeros(hidden_size, memory_mlp_size))

        # Per-sequence snapshot of W1/W2 for outer-loss gradient paths.
        # Created lazily on first read()/compute_surprise() after reset().
        self._W1_snap: Tensor | None = None
        self._W2_snap: Tensor | None = None

        # Cached values between compute_surprise() and apply_update()
        self._cached_g1: Tensor | None = None
        self._cached_g2: Tensor | None = None
        self._cached_eta: float = 0.9
        self._cached_theta: float = 0.1
        self._cached_alpha: float = 0.01

        # Set stable scalar biases before reset(). MemoryTransformer._init_weights()
        # calls this again after its global init resets all biases to 0.
        self._init_scalar_projections()
        self.reset()

    def _init_scalar_projections(self) -> None:
        """Initialize biases so η≈0.9, θ≈0.1, α≈0.01 at the start of training.

        Called from MemoryTransformer._init_weights() after the global std=0.02
        linear init, which would otherwise reset these biases to zero and give
        η=θ=α=0.5 — a catastrophically aggressive forgetting schedule.
        """
        nn.init.zeros_(self.eta_proj.weight)
        nn.init.constant_(self.eta_proj.bias, math.log(0.9 / 0.1))    # sigmoid ≈ 0.90
        nn.init.zeros_(self.theta_proj.weight)
        nn.init.constant_(self.theta_proj.bias, math.log(0.1 / 0.9))   # sigmoid ≈ 0.10
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, math.log(0.01 / 0.99)) # sigmoid ≈ 0.01

    def _forward_memory(self, query: Tensor, W1: Tensor, W2: Tensor) -> Tensor:
        """query: (..., H) → (..., H) using given weight tensors."""
        return F.gelu(query @ W1.T) @ W2.T

    def _get_snap(self) -> tuple[Tensor, Tensor]:
        """Return (W1_snap, W2_snap) for the current sequence.

        Created once per sequence (after reset()) as a detached clone so
        in-place apply_update() calls cannot invalidate the outer autograd graph.
        """
        if self._W1_snap is None:
            self._W1_snap = self.W1.detach().clone()
            self._W2_snap = self.W2.detach().clone()
        return self._W1_snap, self._W2_snap  # type: ignore[return-value]

    def read(self, query: Tensor) -> Tensor:
        """Read from memory without updating it. query: (B, H) → (B, H).

        W_K is in the outer LM-loss gradient path. W1/W2 accessed via snapshot.
        """
        W1_snap, W2_snap = self._get_snap()
        k_q = self.W_K(query)
        return self._forward_memory(k_q, W1_snap, W2_snap)

    def read_current(self, query: Tensor) -> Tensor:
        """Read from current (post-update) memory state. Detached — no grad path.

        Used for the MAC output gate: o_t = y_t ⊗ M*_t(y_t) after writes are done.
        """
        with torch.no_grad():
            k_q = self.W_K(query)
            return self._forward_memory(k_q, self.W1, self.W2).detach()

    def compute_surprise(self, token: Tensor) -> tuple[Tensor, Tensor]:
        """Compute surprise norm, cache inner gradients, and return outer assoc loss.

        Does NOT update W1/W2. Must be followed by apply_update().

        token: (B, H)
        Returns:
            surprise_norm: (B, 1) — detached, used as write-gate input.
            assoc_loss: scalar tensor with grad_fn — add to outer loss to
                        train W_K (Eq. 12 outer-loop objective).

        Also caches data-dependent η_t, θ_t, α_t for use in apply_update().
        """
        W1_snap, W2_snap = self._get_snap()

        # Data-dependent scalars — averaged over batch (memory is shared across batch)
        with torch.no_grad():
            self._cached_eta   = torch.sigmoid(self.eta_proj(token)).mean().item()
            self._cached_theta = torch.sigmoid(self.theta_proj(token)).mean().item()
            self._cached_alpha = torch.sigmoid(self.alpha_proj(token)).mean().item()

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
        k_t = self.W_K(token)
        v_t = self.W_V(token)
        pred_outer = self._forward_memory(k_t, W1_snap, W2_snap)
        assoc_loss = F.mse_loss(pred_outer, v_t)

        surprise = (g1.norm() + g2.norm()) / 2.0
        batch = token.shape[0]
        return surprise.expand(batch, 1).contiguous().detach(), assoc_loss

    def apply_update(self, gate_val: float | Tensor = 1.0) -> None:
        """Apply cached gradients to memory weights using data-dependent scalars.

        Call compute_surprise() first.

        Update rule (Titans Eqs. 9-10):
            S_t = η_t · S_{t-1} − gate_val · θ_t · softclamp(∇ℓ)
            M_t = (1 − α_t) · M_{t-1} + S_t

        gate_val: scalar or (B, 1) tensor in [0, 1].
                  gate_val=1.0 → pure Titans update (no write gate).
                  gate_val=0.0 → no update (used by GatedTitansMAC when gate closes).
        """
        assert self._cached_g1 is not None, "call compute_surprise() before apply_update()"
        scale = float(gate_val.detach().mean().item()) if isinstance(gate_val, Tensor) else float(gate_val)
        g1 = self._cached_g1.to(device=self.W1.device, dtype=self.W1.dtype)
        g2 = self._cached_g2.to(device=self.W2.device, dtype=self.W2.dtype)

        # Softclamp: smooth tanh saturation instead of hard clip (paper section 3.1)
        g1 = _softclamp_grad(g1)
        g2 = _softclamp_grad(g2)

        eta   = self._cached_eta
        theta = self._cached_theta * scale   # gate scales the effective learning rate
        alpha = self._cached_alpha           # forgetting rate (paper's α_t)

        with torch.no_grad():
            # S_t = η_t · S_{t-1} − θ_t · ∇ℓ
            self.m1.mul_(eta).sub_(g1, alpha=theta)
            self.m2.mul_(eta).sub_(g2, alpha=theta)
            # M_t = (1 − α_t) · M_{t-1} + S_t
            self.W1.mul_(1.0 - alpha).add_(self.m1)
            self.W2.mul_(1.0 - alpha).add_(self.m2)

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
        self._cached_eta   = 0.9
        self._cached_theta = 0.1
        self._cached_alpha = 0.01
