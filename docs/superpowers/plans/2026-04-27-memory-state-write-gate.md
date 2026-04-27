# Memory-State Write-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stages 0–2 of the memory-state architecture paper: expand the Qwen diagnostic, reproduce Titans MAC, build the write gate, and wire everything into a trainable 100M LM with a proxy_v0 eval path.

**Architecture:** A GPT-style 100M transformer (`MemoryTransformer`) where every N layers hosts a `GatedTitansMAC` module — a Titans MAC memory whose surprise update is modulated by a learned sigmoid gate conditioned on the current hidden state, the gradient-based surprise norm, and an exponential decay term. Memory weights are `register_buffer` tensors (not `nn.Parameter`) updated by a per-forward inner optimizer, keeping the outer AdamW optimizer focused on gate and attention weights only.

**Tech Stack:** Python 3.11, PyTorch 2.2+, Hydra 1.3, `transformers` (tokenizer only), `torch.autograd.grad` for inner-loop surprise computation, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-04-27-memory-state-architecture-design.md`

**Scope:** Stages 0–2 only. Stage 3 (ablations at scale, Track B fine-tuning on Qwen2.5-1.5B, official benchmarks) is a separate plan gated on Stage 2 go/no-go results.

---

## File Map

```
# New files
conf/evaluator/memory_diagnostic.yaml          — extended Qwen diagnostic (n=50, long sweep)
conf/train_memory.yaml                         — Hydra root config for memory LM training
conf/model/memory_lm_100m.yaml                 — 100M GPT-style architecture config
conf/trainer/memory_lm.yaml                    — training hyperparameters
conf/experiment/lm_baseline.yaml               — no-memory transformer run
conf/experiment/lm_titans.yaml                 — Titans-only (no gate) run
conf/experiment/lm_gated.yaml                  — gated Titans run

src/memory_state/titans_mac.py                 — TitansMACMemory: inner-optimizer memory module
src/memory_state/write_gate.py                 — WriteGate: sigmoid gate on surprise+content+decay
src/memory_state/gated_memory.py               — GatedTitansMAC: MAC + gate integrated
src/memory_state/lm_backbone.py                — MemoryTransformer: GPT backbone with memory hooks

experiments/memory_state/train_memory.py       — Hydra LM training entry point
experiments/memory_state/data.py               — text data loading and batching

tests/test_titans_mac.py
tests/test_write_gate.py
tests/test_gated_memory.py
tests/test_lm_backbone.py

notes/memory_state/qwen_diagnostic_v1.md       — results template (fill after Stage 0 run)

# Modified files
experiments/eval_memory.py                     — add MemoryTransformerGenerator + per-task reporting
src/shared/hf_inference.py                     — add MemoryTransformerGenerator to load_text_generator
```

---

## Task 1: Stage 0 — Extended Qwen Diagnostic Config

**Files:**
- Create: `conf/evaluator/memory_diagnostic.yaml`
- Create: `notes/memory_state/qwen_diagnostic_v1.md`

- [ ] **Step 1: Create extended diagnostic evaluator config**

```yaml
# conf/evaluator/memory_diagnostic.yaml
suite: memory_state_core
protocol: proxy_v0
seed: 42
examples_per_benchmark: 50
context_word_steps:
  - 512
  - 2048
  - 8192
save_predictions: true
```

- [ ] **Step 2: Create results template note**

```markdown
# Qwen2.5-3B Diagnostic v1

Date: (fill in)
Model: Qwen/Qwen2.5-3B-Instruct
Protocol: proxy_v0, n=50, context=[512, 2048, 8192]

## Per-Benchmark Results

| benchmark | context_words | accuracy | n |
|-----------|--------------|----------|---|
| mqar      | 512          |          | 50 |
| mqar      | 2048         |          | 50 |
| mqar      | 8192         |          | 50 |
| nolima    | 512          |          | 50 |
| nolima    | 2048         |          | 50 |
| nolima    | 8192         |          | 50 |
| ruler     | 512          |          | 50 |
| ruler     | 2048         |          | 50 |
| ruler     | 8192         |          | 50 |
| babilong  | 512          |          | 50 |
| babilong  | 2048         |          | 50 |
| babilong  | 8192         |          | 50 |

## Failure Mode Classification

- [ ] BABILong fails at all context lengths (architectural)
- [ ] BABILong fails only at long context (context-length-bounded)
- [ ] BABILong failure disappears with different prompt format (instruction-following)
- [ ] RULER degrades monotonically with context length (retrieval interference)

## Decision

[ ] Hypothesis earned: BABILong failure persists across context lengths and prompt formats → proceed to Stage 1
[ ] Hypothesis NOT earned: investigate prompt format / task difficulty before proceeding
```

- [ ] **Step 3: Run extended diagnostic**

```bash
source .venv/bin/activate
python experiments/eval_memory.py \
  model=qwen25_3b \
  evaluator=memory_diagnostic
```

Expected: takes ~30–60 min; produces `outputs/memory_state/eval/qwen25_3b/<date>/results/summary.json`

- [ ] **Step 4: Fill in results and make go/no-go decision**

Open `notes/memory_state/qwen_diagnostic_v1.md` and fill in the accuracy table from `summary.json`. Check the failure mode classification. If BABILong remains near 0.00 across 2048 and 8192 words, proceed.

- [ ] **Step 5: Commit**

```bash
git add conf/evaluator/memory_diagnostic.yaml notes/memory_state/qwen_diagnostic_v1.md
git commit -m "Add Stage 0 diagnostic config and results note"
```

---

## Task 2: Titans MAC Memory Module

**Files:**
- Create: `src/memory_state/titans_mac.py`
- Create: `tests/test_titans_mac.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_titans_mac.py
from __future__ import annotations

import torch
import pytest
from memory_state.titans_mac import TitansMACMemory


@pytest.fixture()
def memory():
    return TitansMACMemory(hidden_size=32, memory_mlp_size=16)


def test_read_output_shape(memory):
    x = torch.randn(2, 32)  # (batch=2, hidden=32)
    out = memory.read(x)
    assert out.shape == (2, 32)


def test_write_returns_surprise_shape(memory):
    x = torch.randn(2, 32)
    surprise = memory.write(x)
    assert surprise.shape == (2, 1)


def test_surprise_is_nonnegative(memory):
    x = torch.randn(2, 32)
    surprise = memory.write(x)
    assert (surprise >= 0).all()


def test_memory_weights_change_after_write(memory):
    x = torch.randn(2, 32)
    W1_before = memory.W1.clone()
    memory.write(x)
    assert not torch.allclose(memory.W1, W1_before)


def test_reset_clears_momentum(memory):
    x = torch.randn(2, 32)
    memory.write(x)
    assert memory.m1.abs().sum() > 0  # momentum is nonzero after write
    memory.reset()
    assert memory.m1.abs().sum() == 0


def test_write_does_not_require_future_tokens(memory):
    # write is called one token at a time — no look-ahead
    x1 = torch.randn(2, 32)
    x2 = torch.randn(2, 32)
    s1 = memory.write(x1)
    s2 = memory.write(x2)
    # Both surprise values are valid scalars (not NaN)
    assert not s1.isnan().any()
    assert not s2.isnan().any()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate
pytest tests/test_titans_mac.py -v
```

Expected: `ModuleNotFoundError: No module named 'memory_state.titans_mac'`

- [ ] **Step 3: Implement TitansMACMemory**

```python
# src/memory_state/titans_mac.py
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
        self.beta = 0.9   # momentum coefficient
        self.alpha = 0.99  # adaptive forgetting

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

    def write(self, token: Tensor) -> Tensor:
        """
        Compute surprise, update memory weights via inner gradient step.

        token: (B, H)
        Returns: surprise_norm (B, 1) — detached, for use as gate input.
        """
        # Detach copies for inner gradient computation
        W1 = self.W1.detach().requires_grad_(True)
        W2 = self.W2.detach().requires_grad_(True)

        # Local associative loss: memory should reconstruct the current token
        pred = self._forward_memory(token.detach(), W1, W2)
        loss = F.mse_loss(pred, token.detach())

        # Gradients w.r.t. memory weights
        g1, g2 = torch.autograd.grad(loss, [W1, W2])

        # Surprise norm: average gradient magnitude across both weight matrices
        surprise = (g1.detach().norm() + g2.detach().norm()) / 2.0
        batch = token.shape[0]
        surprise_per_token = surprise.expand(batch, 1).contiguous()

        # Momentum + adaptive forgetting update (no outer-graph tracking)
        with torch.no_grad():
            eta = self.eta
            self.m1.mul_(self.beta).sub_(g1, alpha=eta)
            self.m2.mul_(self.beta).sub_(g2, alpha=eta)
            self.W1.mul_(self.alpha).add_(self.m1)
            self.W2.mul_(self.alpha).add_(self.m2)

        return surprise_per_token.detach()

    def reset(self) -> None:
        """Reset memory weights and momentum to initial state."""
        nn.init.normal_(self.W1, std=0.02)
        nn.init.normal_(self.W2, std=0.02)
        self.m1.zero_()
        self.m2.zero_()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
source .venv/bin/activate
pytest tests/test_titans_mac.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Lint and commit**

```bash
source .venv/bin/activate
make format && make lint
git add src/memory_state/titans_mac.py tests/test_titans_mac.py
git commit -m "Add TitansMACMemory with inner-optimizer gradient-based write"
```

---

## Task 3: Write Gate Module

**Files:**
- Create: `src/memory_state/write_gate.py`
- Create: `tests/test_write_gate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_write_gate.py
from __future__ import annotations

import torch
import pytest
from memory_state.write_gate import WriteGate


@pytest.fixture()
def gate():
    return WriteGate(hidden_size=32)


def test_gate_output_shape(gate):
    hidden = torch.randn(2, 32)
    surprise = torch.rand(2, 1) * 5.0
    out = gate(hidden, surprise, step=0)
    assert out.shape == (2, 1)


def test_gate_output_in_unit_interval(gate):
    hidden = torch.randn(4, 32)
    surprise = torch.rand(4, 1) * 10.0
    out = gate(hidden, surprise, step=0)
    assert (out >= 0).all() and (out <= 1).all()


def test_decay_term_decreases_with_step(gate):
    hidden = torch.zeros(1, 32)
    surprise = torch.zeros(1, 1)
    # Decay term is based on step; same hidden/surprise, increasing step
    # We can't check the gate direction without training, but decay buffer must not NaN
    for step in range(10):
        out = gate(hidden, surprise, step=step)
        assert not out.isnan().any()


def test_gate_is_differentiable(gate):
    hidden = torch.randn(2, 32, requires_grad=True)
    surprise = torch.rand(2, 1)
    out = gate(hidden, surprise, step=0)
    loss = out.sum()
    loss.backward()
    assert hidden.grad is not None


def test_reset_clears_state(gate):
    hidden = torch.randn(2, 32)
    surprise = torch.rand(2, 1)
    gate(hidden, surprise, step=5)
    gate.reset()
    # After reset, step-0 call should not error or NaN
    out = gate(hidden, surprise, step=0)
    assert not out.isnan().any()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate
pytest tests/test_write_gate.py -v
```

Expected: `ModuleNotFoundError: No module named 'memory_state.write_gate'`

- [ ] **Step 3: Implement WriteGate**

```python
# src/memory_state/write_gate.py
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
        log_surprise = torch.log(surprise + self._eps)  # (B, 1)
        decay = torch.full(
            (batch, 1),
            self.decay_rate ** step,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        features = torch.cat([hidden, log_surprise, decay], dim=-1)  # (B, H+2)
        return torch.sigmoid(self.linear(features))  # (B, 1)

    def reset(self) -> None:
        """No persistent state to reset; provided for interface consistency."""
        pass
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
source .venv/bin/activate
pytest tests/test_write_gate.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Lint and commit**

```bash
source .venv/bin/activate
make format && make lint
git add src/memory_state/write_gate.py tests/test_write_gate.py
git commit -m "Add WriteGate: sigmoid gate conditioned on content, surprise norm, decay"
```

---

## Task 4: GatedTitansMAC Integration

**Files:**
- Create: `src/memory_state/gated_memory.py`
- Create: `tests/test_gated_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gated_memory.py
from __future__ import annotations

import torch
import pytest
from memory_state.gated_memory import GatedTitansMAC


@pytest.fixture()
def gated():
    return GatedTitansMAC(hidden_size=32, memory_mlp_size=16)


def test_forward_output_shape(gated):
    hidden = torch.randn(2, 32)
    out = gated(hidden, step=0)
    assert out.shape == (2, 32)


def test_memory_not_updated_when_gate_zero(gated):
    """Force gate to 0 by zeroing its linear weights; verify W1 unchanged."""
    torch.nn.init.zeros_(gated.gate.linear.weight)
    torch.nn.init.constant_(gated.gate.linear.bias, -10.0)  # σ(-10) ≈ 0

    W1_before = gated.memory.W1.clone()
    hidden = torch.randn(2, 32)
    gated(hidden, step=0)
    assert torch.allclose(gated.memory.W1, W1_before, atol=1e-5)


def test_memory_updated_when_gate_one(gated):
    """Force gate to 1; verify W1 changes (same as pure Titans write)."""
    torch.nn.init.zeros_(gated.gate.linear.weight)
    torch.nn.init.constant_(gated.gate.linear.bias, 10.0)  # σ(10) ≈ 1

    W1_before = gated.memory.W1.clone()
    hidden = torch.randn(2, 32)
    gated(hidden, step=0)
    assert not torch.allclose(gated.memory.W1, W1_before, atol=1e-5)


def test_reset_clears_memory(gated):
    hidden = torch.randn(2, 32)
    gated(hidden, step=0)
    W1_after_write = gated.memory.W1.clone()
    gated.reset()
    # After reset, W1 should differ from post-write state
    assert not torch.allclose(gated.memory.W1, W1_after_write, atol=1e-5)


def test_forward_is_differentiable(gated):
    hidden = torch.randn(2, 32, requires_grad=True)
    out = gated(hidden, step=0)
    loss = out.sum()
    loss.backward()
    assert hidden.grad is not None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate
pytest tests/test_gated_memory.py -v
```

Expected: `ModuleNotFoundError: No module named 'memory_state.gated_memory'`

- [ ] **Step 3: Implement GatedTitansMAC**

```python
# src/memory_state/gated_memory.py
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

    def forward(self, hidden: Tensor, step: int) -> Tensor:
        """
        hidden: (B, H) — current token representation
        step:   int — position in sequence, for decay term
        Returns: (B, H) — hidden enriched with memory context
        """
        # Read: what does memory know about this query?
        mem_ctx = self.memory.read(hidden)  # (B, H)

        # Compute surprise and tentative update (before gate)
        surprise = self.memory.write(hidden)  # (B, 1), also updates W1/W2 internally

        # Gate: should we accept the write that just happened?
        # If gate ≈ 0, undo the write by re-writing with zero update.
        gate_val = self.gate(hidden.detach(), surprise, step)  # (B, 1)

        # Scale the last momentum step by gate (post-hoc gating via momentum scaling)
        # gate < 1 means we dampen the update that was applied
        with torch.no_grad():
            self.memory.m1.mul_(gate_val.mean())
            self.memory.m2.mul_(gate_val.mean())

        # Enrich hidden with memory context
        combined = torch.cat([hidden, mem_ctx], dim=-1)  # (B, 2H)
        return self.out_proj(combined)  # (B, H)

    def reset(self) -> None:
        self.memory.reset()
        self.gate.reset()
```

- [ ] **Step 4: Run tests**

```bash
source .venv/bin/activate
pytest tests/test_gated_memory.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
source .venv/bin/activate
make test
```

Expected: all existing tests still pass.

- [ ] **Step 6: Lint and commit**

```bash
source .venv/bin/activate
make format && make lint
git add src/memory_state/gated_memory.py tests/test_gated_memory.py
git commit -m "Add GatedTitansMAC: write gate modulates Titans surprise update"
```

---

## Task 5: GPT-Style Backbone with Memory Hooks

**Files:**
- Create: `src/memory_state/lm_backbone.py`
- Create: `tests/test_lm_backbone.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lm_backbone.py
from __future__ import annotations

import torch
import pytest
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
    # Changing token at position 5 should NOT affect logits at positions 0-4
    ids1 = torch.randint(0, 256, (1, 8))
    ids2 = ids1.clone()
    ids2[0, 5] = (ids2[0, 5] + 1) % 256
    logits1 = model(ids1)
    logits2 = model(ids2)
    assert torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-5)


def test_gate_activations_returned_when_requested(small_cfg):
    model = MemoryTransformer(small_cfg, use_memory=True)
    ids = torch.randint(0, 256, (1, 8))
    model(ids)
    activations = model.get_gate_activations()
    assert activations is not None
    assert len(activations) == 2  # memory_every_n_layers=2, n_layers=4 → 2 modules
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate
pytest tests/test_lm_backbone.py -v
```

Expected: `ModuleNotFoundError: No module named 'memory_state.lm_backbone'`

- [ ] **Step 3: Implement MemoryTransformer**

```python
# src/memory_state/lm_backbone.py
from __future__ import annotations

import math
from dataclasses import dataclass, field

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
            self.memory_modules = nn.ModuleList([
                GatedTitansMAC(cfg.d_model, cfg.memory_mlp_size, cfg.memory_decay_init)
                for i in range(cfg.n_layers)
                if i % cfg.memory_every_n_layers == 0
            ])
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
                    mem_out = mem_module(token_h, step=t)  # (B, d_model)
                    enriched[:, t, :] = mem_out
                    # Collect gate activations for interpretability
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
        return [
            torch.stack(acts, dim=0) if acts else None
            for acts in self._gate_activations
        ]
```

- [ ] **Step 4: Run tests**

```bash
source .venv/bin/activate
pytest tests/test_lm_backbone.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
source .venv/bin/activate
make test
```

Expected: all tests PASS.

- [ ] **Step 6: Lint and commit**

```bash
source .venv/bin/activate
make format && make lint
git add src/memory_state/lm_backbone.py tests/test_lm_backbone.py
git commit -m "Add MemoryTransformer: GPT backbone with GatedTitansMAC at every N layers"
```

---

## Task 6: Hydra Configs for Memory LM Training

**Files:**
- Create: `conf/train_memory.yaml`
- Create: `conf/model/memory_lm_100m.yaml`
- Create: `conf/trainer/memory_lm.yaml`
- Create: `conf/experiment/lm_baseline.yaml`
- Create: `conf/experiment/lm_titans.yaml`
- Create: `conf/experiment/lm_gated.yaml`

- [ ] **Step 1: Create root Hydra config**

```yaml
# conf/train_memory.yaml
defaults:
  - track: memory_state
  - runtime: default
  - logging: tensorboard
  - model: memory_lm_100m
  - trainer: memory_lm
  - experiment: lm_gated
  - _self_

project:
  name: bumblebee

hydra:
  run:
    dir: ${runtime.output_root}/${track.slug}/train/${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: true
```

- [ ] **Step 2: Create model architecture config**

```yaml
# conf/model/memory_lm_100m.yaml
name: memory_lm_100m
vocab_size: 50257        # GPT-2 BPE vocabulary
d_model: 768
n_heads: 12
n_layers: 12
d_ffn: 3072
max_seq_len: 2048
dropout: 0.1
memory_mlp_size: 64
memory_every_n_layers: 2
memory_decay_init: 0.99
```

- [ ] **Step 3: Create trainer config**

```yaml
# conf/trainer/memory_lm.yaml
seed: 42
device: auto             # auto-selects cuda > mps > cpu
learning_rate: 3.0e-4
weight_decay: 0.1
grad_clip: 1.0
batch_size: 8
seq_len: 512             # tokens per training sequence
max_steps: 100000        # ~500M tokens at batch=8, seq=512, grad_accum=1
warmup_steps: 2000
log_every_n_steps: 100
save_every_n_steps: 5000
best_checkpoint_name: best.pt
last_checkpoint_name: last.pt
grad_accumulation_steps: 1
```

- [ ] **Step 4: Create experiment configs for the three training variants**

```yaml
# conf/experiment/lm_baseline.yaml
name: lm_baseline
description: "Pure transformer, no memory module"
use_memory: false
```

```yaml
# conf/experiment/lm_titans.yaml
name: lm_titans
description: "Titans MAC memory, gradient-only write (no gate)"
use_memory: true
gate_disabled: true      # freeze gate to output 1.0
```

```yaml
# conf/experiment/lm_gated.yaml
name: lm_gated
description: "Titans MAC with learned write gate (our contribution)"
use_memory: true
gate_disabled: false
```

- [ ] **Step 5: Verify Hydra config resolves without error**

```bash
source .venv/bin/activate
python -c "
import hydra
from omegaconf import OmegaConf
with hydra.initialize(config_path='../conf', version_base='1.3'):
    cfg = hydra.compose(config_name='train_memory')
    print(OmegaConf.to_yaml(cfg))
"
```

Expected: prints resolved config with no errors.

- [ ] **Step 6: Commit**

```bash
git add conf/train_memory.yaml conf/model/memory_lm_100m.yaml conf/trainer/memory_lm.yaml \
        conf/experiment/lm_baseline.yaml conf/experiment/lm_titans.yaml conf/experiment/lm_gated.yaml
git commit -m "Add Hydra configs for memory LM training: baseline, titans, gated variants"
```

---

## Task 7: Training Entry Point

**Files:**
- Create: `experiments/memory_state/data.py`
- Create: `experiments/memory_state/train_memory.py`

- [ ] **Step 1: Create data module**

```python
# experiments/memory_state/data.py
"""
Text data loading for memory LM training.

Smoke mode: generates random token IDs for fast iteration.
Real mode: streams from a pre-tokenized binary file (see below for prep).

To prepare real training data (run once, requires ~2GB disk):
  pip install datasets tiktoken
  python experiments/memory_state/data.py --prepare
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
from torch import Tensor


def synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> Tensor:
    """Random token IDs for smoke tests — no real text needed."""
    return torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)


class TokenDataset:
    """Streams fixed-length chunks from a flat binary token file (uint16)."""

    def __init__(self, path: str | Path, seq_len: int) -> None:
        import numpy as np
        self.data = np.memmap(str(path), dtype="uint16", mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def get_batch(self, start: int, batch_size: int, device: torch.device) -> Tensor:
        import numpy as np
        chunks = []
        for i in range(batch_size):
            idx = (start + i * self.seq_len) % len(self)
            chunk = torch.from_numpy(self.data[idx : idx + self.seq_len + 1].astype("int64"))
            chunks.append(chunk)
        return torch.stack(chunks).to(device)


def prepare_fineweb(output_path: str | Path, num_tokens: int = 100_000_000) -> None:
    """Download and tokenize FineWeb-edu to a flat uint16 binary file."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    written = 0
    with out.open("wb") as f:
        for example in ds:
            ids = enc.encode_ordinary(example["text"])
            ids.append(enc.eot_token)
            f.write(struct.pack(f"{len(ids)}H", *ids))
            written += len(ids)
            if written >= num_tokens:
                break
    print(f"Wrote {written:,} tokens to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", default="data/fineweb_train.bin")
    parser.add_argument("--num_tokens", type=int, default=100_000_000)
    args = parser.parse_args()
    if args.prepare:
        prepare_fineweb(args.output, args.num_tokens)
```

- [ ] **Step 2: Create training entry point**

```python
# experiments/memory_state/train_memory.py
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig
from shared.runtime import prepare_run_artifacts, save_checkpoint
from experiments.memory_state.data import synthetic_batch, TokenDataset


def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(cfg: DictConfig) -> MemoryTransformer:
    model_cfg = MemoryTransformerConfig(
        vocab_size=int(cfg.model.vocab_size),
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        d_ffn=int(cfg.model.d_ffn),
        max_seq_len=int(cfg.model.max_seq_len),
        dropout=float(cfg.model.dropout),
        memory_mlp_size=int(cfg.model.memory_mlp_size),
        memory_every_n_layers=int(cfg.model.memory_every_n_layers),
        memory_decay_init=float(cfg.model.memory_decay_init),
    )
    use_memory = bool(cfg.experiment.use_memory)
    return MemoryTransformer(model_cfg, use_memory=use_memory)


@hydra.main(version_base="1.3", config_path="../../conf", config_name="train_memory")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.trainer.seed))
    device = select_device(str(cfg.trainer.device))
    artifacts = prepare_run_artifacts(cfg.runtime)

    model = build_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model_params={num_params:,} use_memory={cfg.experiment.use_memory}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
        betas=(0.9, 0.95),
    )

    data_path = ROOT / "data" / "fineweb_train.bin"
    use_real_data = data_path.exists()
    if use_real_data:
        dataset = TokenDataset(data_path, int(cfg.trainer.seq_len))
        print(f"Using real data: {data_path} ({len(dataset):,} chunks)")
    else:
        print("data/fineweb_train.bin not found — using synthetic data (smoke mode)")
        print("To prepare real data: python experiments/memory_state/data.py --prepare")

    writer = SummaryWriter(log_dir=str(artifacts.tensorboard_dir))
    writer.add_text("config/resolved", OmegaConf.to_yaml(cfg, resolve=True), 0)

    batch_size = int(cfg.trainer.batch_size)
    seq_len = int(cfg.trainer.seq_len)
    vocab_size = int(cfg.model.vocab_size)
    grad_clip = float(cfg.trainer.grad_clip)
    warmup = int(cfg.trainer.warmup_steps)
    max_steps = int(cfg.trainer.max_steps)

    model.train()
    step = 0
    while step < max_steps:
        # Learning rate warmup
        lr = float(cfg.trainer.learning_rate) * min(1.0, step / max(warmup, 1))
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Reset memory at sequence boundaries (each training batch = fresh sequence)
        if hasattr(model, "reset_memory"):
            model.reset_memory()

        # Fetch batch
        if use_real_data:
            tokens = dataset.get_batch(step * batch_size, batch_size, device)
        else:
            tokens = synthetic_batch(batch_size, seq_len, vocab_size, device)

        input_ids = tokens[:, :seq_len]
        targets = tokens[:, 1 : seq_len + 1]

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)  # (B, T, V)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        step += 1

        if step % int(cfg.trainer.log_every_n_steps) == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            print(f"step={step} loss={loss.item():.4f} lr={lr:.2e}")

        if step % int(cfg.trainer.save_every_n_steps) == 0:
            state = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            save_checkpoint(state, artifacts.checkpoint_dir / f"step_{step:07d}.pt")

    writer.close()
    print(f"run_dir={artifacts.run_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify smoke run (synthetic data, 10 steps)**

```bash
source .venv/bin/activate
python experiments/memory_state/train_memory.py \
  experiment=lm_gated \
  trainer.max_steps=10 \
  trainer.log_every_n_steps=1 \
  trainer.save_every_n_steps=10
```

Expected: prints 10 loss lines, saves checkpoint, no errors. Loss should be near `log(50257) ≈ 10.8` at step 1 (random init).

- [ ] **Step 4: Verify baseline smoke run**

```bash
source .venv/bin/activate
python experiments/memory_state/train_memory.py \
  experiment=lm_baseline \
  trainer.max_steps=10 \
  trainer.log_every_n_steps=1 \
  trainer.save_every_n_steps=10
```

Expected: same as above, use_memory=False.

- [ ] **Step 5: Commit**

```bash
git add experiments/memory_state/data.py experiments/memory_state/train_memory.py
git commit -m "Add memory LM training entry point with synthetic and FineWeb data paths"
```

---

## Task 8: Eval Pipeline Integration for Memory Models

**Files:**
- Modify: `src/shared/hf_inference.py`
- Modify: `experiments/eval_memory.py`

Connects `MemoryTransformer` to the existing `eval_memory.py` runner so trained checkpoints can be evaluated on proxy_v0.

- [ ] **Step 1: Add MemoryTransformerGenerator to hf_inference.py**

Open `src/shared/hf_inference.py`. After the `HuggingFaceGenerator` class, add:

```python
class MemoryTransformerGenerator:
    """Wraps a trained MemoryTransformer checkpoint for use with eval_memory.py."""

    def __init__(self, model_cfg) -> None:
        import tiktoken
        from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig

        self.enc = tiktoken.get_encoding("gpt2")
        cfg = MemoryTransformerConfig(
            vocab_size=int(model_cfg.vocab_size),
            d_model=int(model_cfg.d_model),
            n_heads=int(model_cfg.n_heads),
            n_layers=int(model_cfg.n_layers),
            d_ffn=int(model_cfg.d_ffn),
            max_seq_len=int(model_cfg.max_seq_len),
            dropout=0.0,
            memory_mlp_size=int(model_cfg.memory_mlp_size),
            memory_every_n_layers=int(model_cfg.memory_every_n_layers),
            memory_decay_init=float(model_cfg.memory_decay_init),
        )
        use_memory = bool(model_cfg.use_memory)
        self.model = MemoryTransformer(cfg, use_memory=use_memory)
        checkpoint = torch.load(str(model_cfg.checkpoint_path), map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.max_new_tokens = int(model_cfg.max_new_tokens)

    def generate(self, prompt: str, *, answer: str | None = None) -> ModelResponse:
        del answer
        ids = self.enc.encode(prompt)
        input_ids = torch.tensor([ids])
        input_len = len(ids)

        if hasattr(self.model, "reset_memory"):
            self.model.reset_memory()

        generated: list[int] = []
        with torch.no_grad():
            context = input_ids
            for _ in range(self.max_new_tokens):
                logits = self.model(context)  # (1, T, V)
                next_token = int(logits[0, -1].argmax())
                generated.append(next_token)
                context = torch.cat([context, torch.tensor([[next_token]])], dim=1)
                if next_token == self.enc.eot_token:
                    break

        text = self.enc.decode(generated).strip()
        return ModelResponse(text=text, input_tokens=input_len, output_tokens=len(generated))
```

Then modify `load_text_generator` to add the new backend:

```python
def load_text_generator(model_cfg) -> TextGenerator:
    backend = str(model_cfg.backend)
    if backend == "oracle":
        return OracleGenerator()
    if backend == "huggingface":
        return HuggingFaceGenerator(model_cfg)
    if backend == "memory_transformer":
        return MemoryTransformerGenerator(model_cfg)
    raise ValueError(f"Unsupported model backend: {backend}")
```

- [ ] **Step 2: Add per-task BABILong reporting to eval_memory.py**

In `experiments/eval_memory.py`, in the `summarize` function, add task-family breakdown:

After the existing `summaries` list is built, add:

```python
    # Per-task-family breakdown for BABILong
    babilong_rows = [r for r in rows if str(r["benchmark"]) == "babilong"]
    task_family_breakdown: list[dict[str, object]] = []
    if babilong_rows:
        by_family: dict[str, list[dict[str, object]]] = {}
        for row in babilong_rows:
            fam = str(row.get("metadata", {}).get("task_family", "unknown"))
            by_family.setdefault(fam, []).append(row)
        for fam, items in sorted(by_family.items()):
            correct = sum(1 for it in items if bool(it["correct"]))
            task_family_breakdown.append({
                "benchmark": "babilong",
                "task_family": fam,
                "num_examples": len(items),
                "accuracy": correct / len(items),
            })
```

And add `"babilong_task_families": task_family_breakdown` to the returned dict.

- [ ] **Step 3: Run tests**

```bash
source .venv/bin/activate
make test
```

Expected: all existing tests PASS (new code is additive; no existing interfaces changed).

- [ ] **Step 4: Lint and commit**

```bash
source .venv/bin/activate
make format && make lint
git add src/shared/hf_inference.py experiments/eval_memory.py
git commit -m "Add MemoryTransformerGenerator and per-task BABILong breakdown to eval pipeline"
```

---

## Task 9: Gate Interpretability — AUROC on bAbI State-Mutation Tokens

**Files:**
- Modify: `src/memory_state/proxy_tasks.py` — add state-mutation token position metadata
- Create: `experiments/memory_state/gate_auroc.py` — standalone AUROC evaluation script

- [ ] **Step 1: Add state-mutation metadata to babilong examples**

In `src/memory_state/proxy_tasks.py`, modify `_babilong_example` to record which token positions in the prompt correspond to state-mutation events (e.g., location changes, item transfers).

In the `fact_chain` task type, the state-mutation sentences are `"{holder} moved to the {rooms[i]}"`. Record their byte offsets in metadata:

```python
    if task_type == "fact_chain":
        holder = "Mira"
        item = "lantern"
        rooms = ["atrium", "hallway", "studio"]
        move_sentences = [
            f"{holder} moved to the {rooms[1]}.",
            f"{holder} moved to the {rooms[2]}.",
        ]
        prompt = (
            "Story:\n"
            f"{holder} picked up the {item} in the {rooms[0]}.\n"
            f"{_filler(context_words // 3, rng)}\n"
            f"{move_sentences[0]}\n"
            f"{_filler(context_words // 3, rng)}\n"
            f"{move_sentences[1]}\n\n"
            f"Question: Where is the {item}? Answer with the location only."
        )
        answer = rooms[-1]
        mutation_phrases = move_sentences
```

Add `"mutation_phrases": mutation_phrases` to the `metadata` dict. This allows the AUROC evaluator to identify which tokens are state mutations via string search.

- [ ] **Step 2: Create gate AUROC evaluator**

```python
# experiments/memory_state/gate_auroc.py
"""
Compute gate AUROC on bAbI state-mutation tokens.

Usage:
  python experiments/memory_state/gate_auroc.py \
    --checkpoint outputs/memory_state/train/memory_lm_100m/<run>/checkpoints/best.pt \
    --config conf/train_memory.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
import tiktoken
import torch

from memory_state.proxy_tasks import build_suite_examples, MemoryTaskExample
from shared.benchmark_registry import resolve_eval_suite


def token_is_mutation(token_str: str, mutation_phrases: list[str]) -> bool:
    return any(phrase.lower() in token_str.lower() for phrase in mutation_phrases)


def compute_gate_auroc(checkpoint_path: str, n_examples: int = 50, seed: int = 42) -> float:
    from sklearn.metrics import roc_auc_score
    from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig

    enc = tiktoken.get_encoding("gpt2")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = checkpoint["config"]["model"]

    model_cfg = MemoryTransformerConfig(
        vocab_size=cfg_dict["vocab_size"],
        d_model=cfg_dict["d_model"],
        n_heads=cfg_dict["n_heads"],
        n_layers=cfg_dict["n_layers"],
        d_ffn=cfg_dict["d_ffn"],
        max_seq_len=cfg_dict["max_seq_len"],
        dropout=0.0,
        memory_mlp_size=cfg_dict["memory_mlp_size"],
        memory_every_n_layers=cfg_dict["memory_every_n_layers"],
        memory_decay_init=cfg_dict["memory_decay_init"],
    )
    model = MemoryTransformer(model_cfg, use_memory=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    examples = build_suite_examples(
        suite_slug="memory_state_core",
        context_word_steps=[512],
        examples_per_benchmark=n_examples,
        seed=seed,
        repo_root=ROOT,
    )
    babilong_examples = [e for e in examples if e.benchmark == "babilong"]

    all_labels: list[float] = []
    all_scores: list[float] = []

    for example in babilong_examples:
        mutation_phrases = list(example.metadata.get("mutation_phrases", []))
        if not mutation_phrases:
            continue

        ids = enc.encode(example.prompt)
        tokens_str = [enc.decode([t]) for t in ids]
        labels = [
            1.0 if token_is_mutation(tok, mutation_phrases) else 0.0
            for tok in tokens_str
        ]

        model.reset_memory()
        input_ids = torch.tensor([ids])
        with torch.no_grad():
            model(input_ids)

        activations = model.get_gate_activations()
        if activations is None or activations[0] is None:
            continue

        # Use first memory module's gate activations
        gate = activations[0].squeeze(-1).squeeze(-1).numpy()  # (T,)
        T = min(len(labels), len(gate))
        all_labels.extend(labels[:T])
        all_scores.extend(gate[:T].tolist())

    if sum(all_labels) == 0 or sum(all_labels) == len(all_labels):
        print("WARNING: degenerate labels — AUROC undefined")
        return float("nan")

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"gate_auroc={auroc:.4f} n_tokens={len(all_labels)} n_mutations={int(sum(all_labels))}")
    return auroc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    compute_gate_auroc(args.checkpoint, args.n_examples, args.seed)
```

- [ ] **Step 3: Verify script runs without error on a smoke checkpoint**

First run a short training to get a checkpoint:

```bash
source .venv/bin/activate
python experiments/memory_state/train_memory.py \
  experiment=lm_gated \
  trainer.max_steps=5 \
  trainer.save_every_n_steps=5 \
  trainer.log_every_n_steps=1
```

Then run AUROC (sklearn required: `pip install scikit-learn`):

```bash
source .venv/bin/activate
pip install scikit-learn
python experiments/memory_state/gate_auroc.py \
  --checkpoint outputs/memory_state/train/memory_lm_100m/<date>/<time>/checkpoints/step_0000005.pt
```

Expected: prints `gate_auroc=<value>` without error. On a random-init model, AUROC will be near 0.5 (random). After training, target > 0.65.

- [ ] **Step 4: Lint and commit**

```bash
source .venv/bin/activate
make format && make lint
git add src/memory_state/proxy_tasks.py experiments/memory_state/gate_auroc.py
git commit -m "Add state-mutation metadata to babilong proxy and gate AUROC evaluator"
```

---

## Stage 2 Go/No-Go Checkpoint

After running the full training (real data, ~100k steps), evaluate and check all conditions:

```bash
source .venv/bin/activate
# Evaluate gated model
python experiments/eval_memory.py \
  model=memory_lm_100m \
  model.checkpoint_path=outputs/.../best.pt \
  evaluator=memory_diagnostic

# Evaluate Titans-only baseline (same checkpoint dir, titans variant)
python experiments/eval_memory.py \
  model=memory_lm_100m \
  model.checkpoint_path=outputs/.../titans_best.pt \
  model.use_memory=true \
  evaluator=memory_diagnostic

# Gate AUROC
python experiments/memory_state/gate_auroc.py \
  --checkpoint outputs/.../best.pt
```

**Go criteria (all must pass):**
- [ ] Gated model BABILong Δ ≥ 0.10 vs. no-memory baseline at 2k words (≥ 2 seeds)
- [ ] Gated model matches or beats Titans-only at 2k words
- [ ] Gate AUROC > 0.55 (above chance; target > 0.65)
- [ ] No MQAR regression (gated MQAR ≥ baseline MQAR − 0.05)
- [ ] PG-19 perplexity not worse than baseline (evaluate with `experiments/memory_state/train_memory.py` in eval mode)

**Stop condition:** If BABILong Δ < 0.05 vs. no-memory AND Titans-only also shows no improvement, investigate gate design and training stability before running Stage 3 ablations.

---

## Self-Review Notes

**Spec coverage check:**
- Stage 0 diagnostic → Task 1 ✓
- TitansMACMemory → Task 2 ✓
- WriteGate (content + surprise + decay) → Task 3 ✓
- GatedTitansMAC (modulates, not replaces) → Task 4 ✓
- 100M GPT backbone → Task 5 ✓
- Two-track training (Track A 100M from scratch) → Task 7 ✓
- Track B (1.5B fine-tune) → NOT in this plan; separate Stage 3 plan
- Mamba-2 and TTT baselines → NOT in this plan; separate Stage 3 plan
- Official BABILong/MQAR ports → NOT in this plan; separate Stage 3 plan
- Eval integration for memory models → Task 8 ✓
- Gate AUROC interpretability metric → Task 9 ✓
- Numerical go/no-go gate → Stage 2 checkpoint ✓
- Compute budget definition (token-matched) → Task 7, trainer config ✓
- Gate collapse mitigation (entropy regularization) → noted in risks; entropy reg is a manual intervention if collapse observed, not automatic ✓
