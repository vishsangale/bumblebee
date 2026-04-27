# Qwen2.5-3B Proxy V0 Baseline

Date: April 20, 2026

## Model
- `Qwen/Qwen2.5-3B-Instruct`
- first runnable open-weight baseline for the `memory_state` suite

## Protocol
- suite: `memory_state_core`
- protocol: `proxy_v0`
- examples per benchmark: `2`
- context steps: `128`, `512`

Command:
```bash
source .venv/bin/activate
python experiments/eval_memory.py model=qwen25_3b evaluator.examples_per_benchmark=2 evaluator.context_word_steps=[128,512]
```

Artifacts:
- summary: `outputs/memory_state/eval/qwen25_3b/2026-04-20/23-39-12/results/summary.json`
- predictions: `outputs/memory_state/eval/qwen25_3b/2026-04-20/23-39-12/results/predictions.jsonl`

## Result
- overall accuracy: `0.625` over `16` examples
- `mqar`: `1.00`
- `nolima`: `1.00`
- `ruler`: `0.50`
- `babilong`: `0.00`

## Initial Read
This baseline cleanly separates retrieval from reasoning-over-state. The model solved direct associative recall and the current latent-retrieval proxy, but it degraded on ordered multi-needle outputs and fully failed the `babilong`-style state tracking tasks. The main near-term question is whether stronger memory architectures will mostly help the `babilong` failure mode, or whether these misses are still dominated by general instruction-following and output-format control.

## Caveat
This is a lightweight benchmark-family proxy, not a claim of full official benchmark reproduction.
