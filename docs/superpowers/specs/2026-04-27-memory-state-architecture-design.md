# Memory-State Architecture Paper: Design Spec

Date: 2026-04-27 (revised after Gemini / Codex / Opus review)
Track: `memory_state`

## Summary

An architecture paper proposing a learned causal write controller for Titans-style neural memory (`arxiv:2501.00663`). The gate modulates the magnitude of Titans' surprise-driven test-time update — it does not replace the update rule. The contribution is an explicit, inspectable write-budget signal on top of the strongest existing memory mechanism, with a quantitative interpretability claim and long-context extrapolation results. Target venues: ICLR / NeurIPS / ICML.

---

## 1. Anchor Paper and Hypothesis

**Anchor:** Titans MAC variant (`arxiv:2501.00663`, Memory as a Context). ATLAS (`arxiv:2505.23735`) and TTT (`arxiv:2407.04620`) are primary related work, not secondary references — the paper must position against both.

Titans MAC is the specific variant to reproduce: persistent long-term memory used as a context prefix, updated at each step via a surprise-based gradient rule. The MAC variant has the cleanest separation of memory read from attention and is the closest to the spec's integration design.

**Pre-architecture diagnostic (required before hypothesis is earned):**
The proxy_v0 run (n=2, 128–512 words) is insufficient evidence that BABILong failure is architectural. Before committing to architecture work, run a controlled diagnostic on Qwen2.5-3B:
- n ≥ 50 examples per benchmark
- Per-task breakdown (bAbI tasks 1–20 individually)
- Multiple prompt format variants (to rule out instruction-following / format failures)
- Context lengths: 512, 2048, 8192 words

Only after this diagnostic do we have license to attribute failure to architecture rather than prompt format, output control, or task difficulty at short context.

**Hypothesis (to be confirmed by diagnostic):** Standard transformers degrade on state-tracking tasks (bAbI task 4, 15, 16, 19 families) as context grows because they lack a write-capable persistent state. At context lengths beyond 8k words, a Titans-style memory with a learned write controller will show a shallower degradation curve than a parameter-matched transformer baseline, Mamba-2, and TTT.

**Proposed contribution:** A learned causal write gate that modulates the Titans surprise update:
```
memory_update_t = gate_t × titans_surprise_update_t
```
The gate is a differentiable sigmoid scalar per token. It is strictly causal: conditioned on the current token's local features and the surprise norm `‖∇L_t‖` from the Titans inner loop — both are available at inference time without lookahead. This differentiates it from LSTM input gates (no surprise signal) and from Titans' α_t/η_t hyperparameters (fixed, not content-conditioned). The recency component is an exponential decay term on the write budget, resetting per chunk boundary.

**Interpretability commitment:** The gate must support a falsifiable claim, not just a heatmap. Specifically: gate activation should be predictably higher on state-mutation tokens (entity transfers, location changes in bAbI) than on filler. This is evaluated by AUROC on gold-labeled state-mutation token positions from bAbI.

---

## 2. Evaluation Plan

**Existing diagnostic:** Qwen2.5-3B-Instruct proxy_v0 (motivation only, not ablation target)
- MQAR: 1.00 / NoLiMa: 1.00 / RULER: 0.50 / BABILong: 0.00
- Note: n=2, 512-word max — insufficient for causal claims; used as intro motivation only

**Required baselines (all parameter-matched and token-matched):**
1. Transformer (no memory) — trained from scratch identically
2. Sliding-window Transformer — same backbone, fixed window
3. Titans MAC reproduction — gradient-only write, no gate
4. **Mamba-2** — matched params, matched training tokens
5. **TTT** (`arxiv:2407.04620`) — matched params, matched training tokens
6. Our write-gated Titans variant

"Identical compute budget" means identical training tokens seen, not wall-clock time. Memory-augmented models run slower per step; the comparison is token-for-token, not step-for-step.

**Primary metrics:**
- BABILong per-task accuracy (tasks 1, 4, 5, 14, 15, 16, 19 reported separately — pooled accuracy is insufficient)
- RULER subtask breakdown: single-needle, multi-needle, tracing, aggregation
- Language modeling perplexity on PG-19 or ProofPile — required to show the memory module does not regress LM quality
- Gate AUROC on bAbI state-mutation tokens (interpretability metric)

**Context length sweep:** 2k, 8k, 32k, 128k words. Train at 2k; evaluate OOD at 8k, 32k, 128k. The degradation curve slope is the headline metric, not any single-point accuracy.

**Positive result definition:** Write-gated Titans shows a statistically significant shallower BABILong degradation curve versus matched-compute transformer baseline at ≥ 32k words context (p < 0.05, n ≥ 50 per cell, ≥ 3 seeds). Gate AUROC > 0.65 on bAbI state-mutation tokens. No perplexity regression on PG-19. BABILong at 512 words is an internal smoke check only, not a paper result.

**Benchmark fidelity:** Port BABILong and MQAR to official datasets before Stage 3 ablations begin — not just before submission. Official numbers must exist before we lock the ablation design.

---

## 3. Architecture Design

**Two-track training strategy:**

- **Track A (ablations):** 100M–150M param transformer trained from scratch on FineWeb (2B tokens, Chinchilla-approximately-optimal). Gives clean ablations with full architecture control. Used for all gate-design ablations and interpretability experiments.
- **Track B (headline):** Freeze a Qwen2.5-1.5B backbone; train only the memory module (and gate) inserted between layers. Used for long-context extrapolation headline results and comparison against Mamba-2/TTT at scale. This matches the approach taken by TTT and Titans themselves.

Track A results support the method; Track B results support the paper's headline claim.

**Titans MAC memory module (base):**
- Persistent long-term memory: a neural module whose weights are updated online during the forward pass via a surprise-based gradient step
- Surprise signal: `‖∇L_t‖` where `L_t` is a local associative-memory loss measuring how well current memory predicts the current token
- Memory read: query the memory module's weights via a forward pass (not slot-based attention — the memory *is* a small MLP, not a slot bank)
- Memory write: `M_t = M_{t-1} - η · ∇_{M_{t-1}} L_t` (momentum and adaptive forgetting per Titans)

**Write gate (our contribution):**
```
gate_t = σ(W_g · [h_t ; ‖∇L_t‖ ; decay_t])
memory_update_t = gate_t × (−η · ∇_{M_{t-1}} L_t)
M_t = M_{t-1} + memory_update_t
```
- `h_t`: current token hidden state (content relevance signal)
- `‖∇L_t‖`: surprise norm (magnitude of Titans' intended write — causal, available at inference)
- `decay_t`: exponential decay term, reset at chunk boundaries (recency signal)
- `σ`: sigmoid, producing scalar gate in [0, 1]
- Trains end-to-end; gate can be zeroed to recover pure Titans behavior

Differentiation from prior gating:
- LSTM input gate: `σ(W · [h_t, c_{t-1}])` — no surprise signal, no decay
- GRU update gate: convex combination of states — no persistent external memory
- Titans α_t/η_t: scalar hyperparameters, not content-conditioned learned functions
- GLA / RetNet decay: data-independent or input-dependent decay on linear attention — no Titans-style inner-loop learning

**Integration:** Titans memory module inserted every N transformer layers (N ablated; default N=2 for Track A). Read happens before the layer's self-attention; write happens after. Memory state is carried across the sequence; detached at chunk boundaries for training stability.

**Ablation plan:**
1. No memory, matched transformer (Track A baseline)
2. Titans MAC, gradient-only write (no gate) — pure reproduction
3. Gate as multiplier on surprise update (our contribution — scalar gate)
4. Gate as replacement of surprise update (ablation to confirm modulation > replacement)
5. Scalar gate vs. vector gate (gate dimensionality)
6. Content-only gate (no surprise norm, no decay)
7. Surprise-only gate (no content signal, no decay)
8. Fixed/random gate with matched write rate (confirms gate learns something)
9. Memory layer count sweep: N=1, 2, 4, every layer
10. Memory capacity sweep (MLP width)

Most important comparison: ablation 2 vs. 3. If gate does not improve over pure Titans reproduction under matched compute, the paper does not exist.

---

## 4. Path to Paper

### Stage 0 — Diagnostic (1 week, before architecture work)
- Run Qwen2.5-3B on expanded proxy_v0: n ≥ 50, per-task, prompt variants, 512/2k/8k words
- Record per-task failure modes; check whether BABILong failure persists beyond 2k words
- **Done when:** we can attribute failure to architecture (persistent at long context, across prompt formats) vs. instruction-following (disappears with prompt fix)
- **Stop condition:** If BABILong failure is resolved by prompt formatting, revisit hypothesis before proceeding

### Stage 1 — Reproduction (2–3 weeks)
- Read Titans paper; implement MAC variant from scratch
- Train 100M transformer baseline (Track A) on FineWeb 2B tokens
- Reproduce Titans MAC on Track A backbone; verify surprise signal and gradient-based write are stable
- Run proxy_v0 + official BABILong on Track A baseline; confirm failure pattern matches Stage 0 diagnostic
- **Done when:** Titans MAC trains stably with no gradient explosion; baseline failure numbers match Qwen diagnostic direction

### Stage 2 — Write gate integration (3–4 weeks)
- Implement write gate; integrate with Titans MAC update rule
- Train Track A gated model; monitor gate activation statistics daily (collapse = always 0 or always 1)
- If gate collapses: add entropy regularization to loss before escalating architecture changes
- Run ablations 1, 2, 3 (baseline, Titans, gated-Titans) on proxy_v0 for fast iteration
- **Go/No-go gate (numerical):** Gated model must beat Titans reproduction by ΔBABILong ≥ 0.10 at 2k words across ≥ 2 seeds. If not, investigate gate design before Stage 3. Do not run full ablations on a flat result.

### Stage 3 — Ablations, scale, official benchmarks (3–4 weeks)
- Run ablations 4–10 on Track A
- Start Track B: freeze Qwen2.5-1.5B, train memory module only; run BABILong and RULER at 8k–128k
- Port BABILong and MQAR to official datasets; run all baselines (Mamba-2, TTT, Transformer, Sliding-window)
- Add PG-19 perplexity evaluation
- Compute gate AUROC on bAbI gold state-mutation labels
- Run ≥ 3 seeds for all main results; report confidence intervals
- Record all failures in `notes/memory_state/`

### Stage 4 — Paper writing
- Intro: diagnostic evidence (Stage 0) motivates the architectural gap; Qwen results as one data point
- Method: gate formulation, differentiation from LSTM/GLA/Titans-α
- Experiments: Track A ablations + Track B long-context results + interpretability (AUROC)
- Discussion: when does the gate help vs. not; failure cases; scope

---

## 5. Risks and Out of Scope

**Risks:**
- **Titans reproduction difficulty:** No canonical OSS implementation. Weeks may be consumed on stable training before gate work begins. Mitigation: start from an open Mamba-style repo and add the Titans inner loop incrementally.
- **Gate collapse:** Monitor write-gate mean and entropy per step. Have entropy regularization ready as a fallback from day one of Stage 2.
- **Training instability (gradient-based write):** Gradient norms through the memory update can spike. Use gradient clipping; log memory value statistics throughout training.
- **Track B fine-tune failing to improve:** Frozen backbone may not adapt well to memory reads. Mitigation: unfreeze the top 2 transformer layers if needed.
- **Concurrent work:** ATLAS (2505) and likely 3–5 papers in this space by submission date. The gate's specific formulation (surprise-norm-conditioned sigmoid) must be genuinely new or demonstrably better.
- **Compute-matched challenge:** Reviewer 2 will ask for a FLOP-matched baseline. Track compute carefully from Stage 1.
- **Statistical significance:** Commit to ≥ 3 seeds and n ≥ 50 per cell before submitting.

**Out of scope for this paper:**
- Other tracks (adaptive_inference, hierarchical_programs)
- Training beyond 3B params before ablations are clean
- Broad downstream leaderboard evaluation
- Claims about general intelligence from synthetic task wins
- Full official RULER or NoLiMa reproduction (proxy is sufficient for these; focus official effort on BABILong)
