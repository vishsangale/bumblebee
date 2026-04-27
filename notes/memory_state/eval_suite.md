# Memory-State Eval Suite

Working question: which failures are genuinely memory failures rather than generic reasoning or instruction-following failures?

## Initial Suite
- `mqar`
- `ruler`
- `nolima`
- `babilong`

## What To Record For Every Run
- model name and claimed context length
- actual tested context lengths
- prompt template and decoding settings
- benchmark metric
- qualitative failure mode notes

## Out Of Scope For The First Pass
- training a new memory architecture
- broad downstream leaderboard chasing
- large hyperparameter sweeps
- claims about general intelligence from synthetic wins
