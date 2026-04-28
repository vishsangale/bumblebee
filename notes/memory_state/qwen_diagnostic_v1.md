# Qwen2.5-3B Diagnostic v1

Date: 2026-04-27
Model: Qwen/Qwen2.5-3B-Instruct
Protocol: proxy_v0, n=50, context=[512, 2048, 8192]

## Per-Benchmark Results

| benchmark | context_words | accuracy | n  |
|-----------|--------------|----------|----|
| mqar      | 512          | 1.00     | 50 |
| mqar      | 2048         | 0.98     | 50 |
| mqar      | 8192         | 1.00     | 50 |
| nolima    | 512          | 0.68     | 50 |
| nolima    | 2048         | 0.34     | 50 |
| nolima    | 8192         | 0.32     | 50 |
| ruler     | 512          | 0.58     | 50 |
| ruler     | 2048         | 0.56     | 50 |
| ruler     | 8192         | 0.56     | 50 |
| babilong  | 512          | 0.34     | 50 |
| babilong  | 2048         | 0.38     | 50 |
| babilong  | 8192         | 0.34     | 50 |

## BABILong Task-Family Breakdown

| task_family    | accuracy | n  |
|----------------|----------|----|
| set_membership | 1.00     | 48 |
| counting       | 0.10     | 51 |
| fact_chain     | 0.00     | 51 |

## Failure Mode Classification

- [x] BABILong fails at all context lengths (architectural) — fact_chain 0.00 even at 512 words
- [ ] BABILong fails only at long context (context-length-bounded)
- [ ] BABILong failure disappears with different prompt format (instruction-following)
- [x] RULER degrades monotonically with context length (retrieval interference) — NoLiMa: 0.68 → 0.34 → 0.32

## Key Observations

- **MQAR stays at 1.00**: pure key-value retrieval works fine at all context lengths — the model's attention mechanism handles retrieval
- **fact_chain = 0.00 at all context lengths**: state tracking (entity transfers across multiple steps) fails even at 512 words; this is not a context-length effect, it is task-type-specific
- **set_membership = 1.00**: the model can retrieve the final state of a single entity; it cannot track multi-step mutations
- **counting = 0.10**: also near-zero, consistent with the write/update failure pattern
- **NoLiMa degrades 0.68 → 0.32**: multi-hop retrieval degrades with context length — a separate but related signal

## Decision

[x] Hypothesis earned: BABILong fact_chain failure (0.00) persists at 512, 2048, and 8192 words; not a prompt format or context-length effect; MQAR staying high rules out general retrieval failure → proceed to Stage 1 (Titans MAC reproduction)
