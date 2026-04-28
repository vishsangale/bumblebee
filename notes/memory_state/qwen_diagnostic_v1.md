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
