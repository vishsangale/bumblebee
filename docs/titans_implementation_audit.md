# Titans MAC — Implementation Audit

**Date:** 2026-04-28  
**Reference:** [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (arxiv:2501.00663)  
**Reference code:** lucidrains/titans-pytorch  
**Our code:** `src/memory_state/titans_mac.py`, `gated_memory.py`, `write_gate.py`, `lm_backbone.py`

---

## Summary

Two confirmed bugs and one architectural mismatch found after comparing our implementation against the paper's equations (Sections 3–4) and the lucidrains reference implementation.

| # | Issue | Severity | File |
|---|-------|----------|------|
| 1 | Wrong loss objective — identity reconstruction instead of key→value | **High / Bug** | `titans_mac.py:79-80` |
| 2 | MAC integration is actually MAL — memory not visible to attention | **High / Architecture** | `lm_backbone.py:140-155` |
| 3 | Fixed η/β/α instead of data-dependent learned scalars | **Medium / Simplification** | `titans_mac.py:42-44` |

---

## Bug 1 — Wrong Loss Objective (High)

### What the paper says

The associative memory loss (Eq. 12) is:

```
ℓ(M_{t-1}; x_t) = ‖ M(k_t) − v_t ‖²
  where  k_t = x_t · W_K
         v_t = x_t · W_V
```

`W_K` and `W_V` are **outer-loop trained** linear projections. The memory learns to map keys to values — an associative lookup table, not a reconstruction network.

### What our code does

`titans_mac.py:79-80`:
```python
pred = self._forward_memory(token.detach(), W1, W2)
loss = F.mse_loss(pred, token.detach())   # identity reconstruction
```

There is no `W_K` or `W_V`. The memory MLP is trained to reconstruct the raw token embedding from itself. This is a trivial identity mapping — attention can already do this without any memory at all.

### Why this explains our training results

The inner gradients from an identity reconstruction task are noisy and carry no useful associative signal. Over 256 token updates per batch, the momentum accumulates this noise and then decays through the `alpha` multiplier. In the Titans (gate_disabled) checkpoint at step 20k:

- `Module 0: m1 norm = 0.0000` — momentum has completely collapsed to zero
- `Module 0: W1 norm = 0.3394` ≈ `init_norm × 0.99^256` — weights barely moved from init, decayed through alpha

The rest of the model (attention layers 8–11, norms 157–213 vs baseline 40–46) compensates for the noise injected by a memory that writes garbage. The Gated variant is more stable only because the write gate damps the magnitude of these useless writes.

### Fix (implemented — `titans_mac.py`)

Added `W_K`, `W_V` as `nn.Linear` parameters; changed the inner loss to key→value:
```python
self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)

# in compute_surprise():
k_t = self.W_K(token).detach()   # detach: inner grad stays in W1/W2 only
v_t = self.W_V(token).detach()   # target
pred = self._forward_memory(k_t, W1, W2)
loss = F.mse_loss(pred, v_t)
```

**Caveat on outer-loop training:** `read()` still uses `torch.no_grad()` (required to keep
`W1.mul_()` in-place updates safe w.r.t. autograd). As a result, W_K and W_V are currently
fixed random projections — no outer LM gradient path exists. The memory still learns a
non-trivial `M(W_K(x)) ≈ W_V(x)` association. Full outer-loop training of W_K/W_V is a
follow-up concern (requires rethinking read() w.r.t. in-place buffer mutations).

**Verified:** After the fix, 256 token updates give `m1 norm = 0.003462` (was `0.000000`);
`W1 norm = 0.2906` vs pure-decay floor of `≈ 0.066` — momentum is alive and W1 is learning.

---

## Bug 2 — MAC Integration is Actually MAL (High)

### What the paper says

MAC (Memory as Context), Eqs. 21–25:

```
1. h_t   = M*_{t-1}(q_t)          READ memory (no weight update)
2. S̃^(t) = [p_1..p_Np | h_t | segment_tokens]   CONCAT as prefix
3. y_t   = Attn(S̃^(t))            ATTENTION over extended context
4. M_t   = update(M_{t-1}, y_t)   WRITE: update memory weights
5. o_t   = y_t ⊗ M*_t(y_t)        OUTPUT: elementwise gate with second read
```

Memory context is **concatenated as extra tokens** that attention can attend to. Attention has full visibility into what memory returned.

### What our code does

`lm_backbone.py:140-155`:
```python
for t in range(T):
    mem_out = mem_module(token_h, step=t)   # read+write
    enriched[:, t, :] = mem_out
x = x + enriched                            # residual add BEFORE attention
x = block(x)                               # attention runs on (x + memory)
```

Memory output is added to the hidden state as a residual, and then attention runs on the summed embedding. Attention has no explicit visibility into the memory contribution — it sees a single blended vector. This is **MAL (Memory as Layer)**, not MAC.

### Practical difference

In MAC, attention can learn to selectively attend to memory tokens vs. current-context tokens via QK scores. It can ignore memory when it's unhelpful and weight it when it's useful. In our MAL variant, the memory contribution is baked into the embedding unconditionally before attention sees anything.

### Status

This is an architectural mismatch, not a small bug. Fixing it requires:
1. Storing memory reads as separate token tensors
2. Expanding the attention mask to include memory prefix positions
3. Changing the write to use attention output `y_t` rather than input `token_h`

This is a larger refactor. We should decide whether to implement true MAC or continue with MAL explicitly (it is a valid research variant — just not what the paper calls MAC).

---

## Issue 3 — Fixed η/β/α Instead of Data-Dependent (Medium / Simplification)

### What the paper says

The momentum update (Eqs. 9–10) and forgetting (Eqs. 13–14) use **data-dependent** scalars:

```
S_t = η_t · S_{t-1} − θ_t · ∇ℓ(M_{t-1}; x_t)
M_t = (1 − α_t) · M_{t-1} + S_t
```

`η_t`, `θ_t`, `α_t` are each sigmoid outputs of a learned linear layer applied to `x_t`. Different tokens can request different write strengths and forgetting rates.

### What our code does

`titans_mac.py:42-44`:
```python
self.eta: float = 0.1   # inner learning rate (fixed)
self.beta: float = 0.9  # momentum coefficient (fixed)
self.alpha: float = 0.99  # adaptive forgetting (fixed)
```

All three are hardcoded scalars. Every token writes to memory with equal weight regardless of content.

### Assessment

This is a simplification, not a correctness bug — fixed scalars are the degenerate special case of the paper's learned scalars. It removes content-dependent selectivity. Given that Bug 1 (wrong loss) is more fundamental, fixing this should come after Bug 1 is resolved.

**Note on alpha naming:** Our `alpha=0.99` means "keep 99% of old weights". The paper's convention for `α_t` is the opposite — `α_t=0.99` would mean "erase 99%". Our behavior (little forgetting) is correct for a memory module; only the variable name convention is inverted relative to the paper.

---

## Things That Are NOT Bugs

| Item | Status |
|------|--------|
| **WriteGate (GatedTitansMAC)** | Our extension — not in the paper. Reasonable approximation of the paper's data-dependent η_t/α_t via a single gating scalar. |
| **Gradient clipping at norm 1.0** | Paper uses soft tanh clamp (`softclamp_grad_norm`). Our hard clip is cruder but not wrong. |
| **Memory modules every N layers** | Paper doesn't specify placement; our every-2-layers choice is a reasonable design decision. |
| **Surprise norm as gate input** | Our extension. The paper's "surprise" is the gradient vector itself used as update direction; we compute the norm separately as a feature for the gate. |

---

## Connection to Training Results (20k steps, bs=16, seq_len=256)

| Model | Final Loss | Key Observation |
|-------|-----------|-----------------|
| lm_baseline | 4.866 | Healthy, uniform layer norms (40–46 across all 12 layers) |
| lm_gated | 5.005 | Stable layer norms, write gate closing slightly (mean 0.42–0.45), partially suppressing Bug 1 damage |
| lm_titans | 6.392 | Layer norms 8–11 explode to 157–213; Module 0 momentum dead; model compensating for memory noise |

The +1.53 nat gap between Titans and baseline is almost entirely explained by Bug 1: the model is injecting noise from a broken inner loss objective, and the transformer layers are warping to compensate. Gated recovers +1.39 nats of that gap by damping (but not eliminating) the writes.

---

## Recommended Fix Order

1. **Fix Bug 1 first** (add W_K/W_V, change loss to key→value). This is ~15 lines and is the foundational correctness issue. Re-run diagnostic (5k steps) to confirm memory starts learning.
2. **Decide on Bug 2** (MAC vs MAL architecture). If we fix Bug 1 and memory still underperforms, implement true MAC. This is the larger refactor.
3. **Fix Issue 3 optionally** (data-dependent η_t/α_t). Only after Bug 1 + Bug 2 are resolved.
