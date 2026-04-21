# Bumblebee

Bumblebee is a clean-slate research repo for discovering a credible successor to the standard Transformer stack. The working thesis is that the next architecture will not come from a single attention trick; it will come from changing one or more of the model's core axes:

1. `memory_state`: write-capable, persistent memory instead of a read-only KV cache
2. `adaptive_inference`: variable-compute reasoning instead of a fixed single forward pass
3. `hierarchical_programs`: learned abstraction and explicit scratchpads instead of a flat token stream

## Why These Three Directions
They are the strongest combination of recent momentum and long-term upside. Recent papers point to memory-centric models such as Mamba-2, TTT, Titans, and ATLAS; adaptive inference ideas such as COCONUT, LLaDA, and Mercury; and abstraction-centric models such as BLT plus earlier modular and workspace-style systems. We will treat white-box design and explicit objectives as a cross-cutting requirement rather than a separate track.

## Repository Layout
- `docs/`: research program, roadmap, and seed paper list
- `configs/tracks/`: YAML definitions for the three research tracks
- `experiments/`: experiment entry points and run organization
- `notes/`: paper notes, ablations, and failure logs
- `src/bumblebee/`: shared Python code
- `tests/`: lightweight regression checks for repo utilities

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make check
```

## Near-Term Plan
- reproduce a narrow baseline for each track
- build shared evaluation harnesses for long-context, reasoning, and compositional tasks
- publish track-specific results before attempting a full hybrid architecture

Detailed rationale lives in [docs/research_program.md](docs/research_program.md) and [docs/seed_papers.md](docs/seed_papers.md).
