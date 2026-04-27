# Memory-State Evaluation Suite

## Why Start Here
Before building a new memory architecture, we should fix the evaluation target. The first step in this track is to run a small, stable benchmark suite on at least one pretrained long-context model and learn where current models actually fail.

The first runnable version in this repo is a `proxy_v0` suite. It is benchmark-family aligned, but it is not yet a claim of full official benchmark reproduction. That distinction matters: the point of `proxy_v0` is to cheaply separate retrieval failures, interference failures, and reasoning-over-retrieved-state failures before we invest in heavier benchmark integrations.

## Core Suite
- `MQAR`: a controlled associative-recall diagnostic from [Zoology](https://arxiv.org/abs/2312.04927). This is the cleanest test of whether a model can recover stored key-value associations across distance and interference.
- `RULER`: a broader synthetic long-context suite from [RULER](https://arxiv.org/abs/2404.06654). It goes beyond single-needle retrieval into tracing and aggregation, which makes it more useful than a plain needle test.
- `NoLiMa`: a stress test from [NoLiMa](https://arxiv.org/abs/2502.05167) for retrieval without obvious lexical overlap. This matters because strong needle scores can hide shortcut behavior.
- `BABILong`: a reasoning-in-haystack benchmark from [BABILong](https://arxiv.org/abs/2406.10149). It checks whether recalled facts can actually be composed into answers.

## First Evaluation Policy
Use one pretrained open model first, not a new architecture. Keep the protocol fixed:
- same prompt template across models
- same length sweep across tasks
- exact-match or benchmark-native metrics only
- per-task and per-length reporting, not just a pooled score

The current runner evaluates lightweight proxies for `MQAR`, `RULER`, `NoLiMa`, and `BABILong`. Once we know which failure modes dominate, we should replace these proxies with official datasets or faithful benchmark ports one family at a time.

If possible, compare one long-context open model against one conventional transformer baseline. The objective of this pass is not to win a leaderboard. It is to identify whether current failures are mostly retrieval failures, interference failures, or reasoning-over-retrieved-state failures.

## Decision Rule
Do not start architecture work until this suite exists and at least one pretrained baseline has been run on it. If the suite does not separate failure modes clearly, improve the suite before adding new model ideas.
