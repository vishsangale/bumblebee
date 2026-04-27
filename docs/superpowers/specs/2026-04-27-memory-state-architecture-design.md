# Memory-State Architecture Paper: Design Spec

Date: 2026-04-27
Track: `memory_state`

## Summary

An architecture paper proposing a write-gated neural memory module as a replacement/augmentation for the standard read-only KV cache. Anchored on Titans (`arxiv:2501.00663`), extended with an explicit interpretable write gate. Target venues: ICLR / NeurIPS / ICML.

---

## 1. Anchor Paper and Hypothesis

**Anchor:** Titans (`arxiv:2501.00663`), with ATLAS (`arxiv:2505.23735`) as a secondary reference.

Titans is the cleaner reproduction target: one module, clear read/write semantics, and ATLAS explicitly builds on it. Understanding Titans is prerequisite to ATLAS anyway.

**Hypothesis:** Standard transformers fail at BABILong-style state tracking (confirmed: Qwen2.5-3B scores 0.00) because the KV cache is read-only and context-length-bounded. A Titans-style neural memory module with a white-box write gate will recover this failure mode while maintaining or improving RULER multi-needle performance.

**Proposed contribution:** An explicit write gate layered on top of the Titans memory module. The gate is a learned scalar or vector per token conditioned on content relevance and a recency signal. It replaces Titans' implicit gradient-flow write with an inspectable, differentiable decision, enabling visualization of which tokens trigger memory writes.

---

## 2. Evaluation Plan

**Existing baseline:** Qwen2.5-3B-Instruct proxy_v0
- MQAR: 1.00
- NoLiMa: 1.00
- RULER: 0.50
- BABILong: 0.00

**Primary metrics:**
- BABILong: recover from 0.00 — headline result
- RULER (multi-needle, tracing, aggregation): improve from 0.50
- MQAR, NoLiMa: maintain ≥ 1.00 — regression here invalidates the architecture

**Context length sweep:** 128, 512, 2048 words minimum. The key metric is the degradation curve as context grows; the goal is a shallower slope versus the baseline.

**Benchmark fidelity:** Use proxy_v0 during development for fast iteration. Before submission, port BABILong (publicly available) and MQAR (from Zoology) to official datasets. Paper numbers must be official-benchmark grounded.

**Positive result definition:** BABILong > 0.50 at 512 words, shallower degradation curve with context length versus the Qwen2.5-3B baseline. RULER improvement on multi-needle and tracing subtasks.

---

## 3. Architecture Design

**Base model:** Small transformer backbone, 100M–300M params, trained from scratch on a text corpus (FineWeb or The Pile subset). Training from scratch gives clean ablations and full control over the architecture.

**Titans memory module (base):**
- A separate neural memory module alongside the attention layers
- Receives the current token representation; decides what to write into persistent memory state
- Reads memory via content-based query (attention over memory slots)
- Gradient-based write: memory updated by a gradient step proportional to a surprise/novelty signal

**Write gate (our contribution):**
- Replaces the implicit gradient-flow write with an explicit learned gate
- Gate is a differentiable scalar or vector per token
- Conditioned on: (a) content relevance — does this token look worth remembering? (b) recency signal
- Trains end-to-end with the rest of the model
- Inspectable at inference: visualize which tokens trigger writes

**Integration:** One memory module per transformer block. Read happens before attention; write happens after. Number of layers with memory is a hyperparameter to ablate.

**Ablation plan:**
1. Pure transformer baseline (no memory module, identical training)
2. Titans memory, gradient-only write (no gate)
3. Our write-gated variant (content + recency gate)
4. Gate ablation: content-only gate vs. content + recency gate

---

## 4. Path to Paper

### Stage 1 — Reproduction (2–3 weeks)
- Read Titans in detail; implement gradient-based memory module
- Train 100M transformer baseline on text subset; verify stable training
- Run proxy_v0 suite on this baseline to confirm BABILong ≈ 0, RULER degrading with context
- **Done when:** clean failure numbers from our own trained model (not just Qwen)

### Stage 2 — Memory module integration (3–4 weeks)
- Implement Titans memory module; integrate into backbone
- Add write gate on top
- Train memory-augmented model with identical data/compute budget as baseline
- Run proxy_v0 suite; check BABILong and RULER numbers
- **Go/No-go gate:** If BABILong shows no improvement at all, investigate before proceeding to Stage 3. Do not run ablations on a flat result.

### Stage 3 — Ablations and official benchmarks (2–3 weeks)
- Run all four ablations
- Port BABILong and MQAR to official datasets
- Extend context sweep to ≥ 2048 words
- Record all failures and negative results in `notes/memory_state/`

### Stage 4 — Paper writing
- Intro: failure characterization (Qwen baseline as motivation)
- Method: write-gated memory architecture
- Experiments: ablation results + official benchmark results
- Discussion: interpretability of write gate, failure cases, scope

---

## 5. Risks and Out of Scope

**Risks:**
- Overfitting on retrieval proxies without real BABILong improvement (check official benchmarks early)
- Training instability from gradient-based memory write (Titans reports this; plan for gradient clipping)
- Write gate collapse (gate learns to always write or never write — monitor gate sparsity during training)

**Out of scope for this paper:**
- Other tracks (adaptive_inference, hierarchical_programs)
- Large-scale training (>1B params) before ablations are clean at small scale
- Broad downstream leaderboard evaluation
- Claims about general intelligence from synthetic task wins
