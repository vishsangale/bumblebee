# Bumblebee

Bumblebee is a clean-slate research repository for discovering a credible successor to the standard Transformer stack. The goal is not to collect incremental efficiency tricks. The goal is to identify architecture changes that could support multiple serious research papers and eventually converge into a new model family.

## Research Thesis
The next architecture will likely come from changing one or more of the model's core axes:

1. `memory_state`: write-capable, persistent memory instead of a read-only KV cache
2. `adaptive_inference`: variable-compute reasoning instead of a fixed single forward pass
3. `hierarchical_programs`: learned abstraction and explicit scratchpads instead of a flat token stream

## Why These Three Directions
These directions are the best combination of recent empirical momentum and long-term upside. Recent papers point to memory-centric models such as Mamba-2, TTT, Titans, and ATLAS; adaptive inference ideas such as COCONUT, LLaDA, and Mercury; and abstraction-centric models such as BLT plus earlier modular and workspace-style systems.

White-box design is a cross-cutting constraint across all three tracks. New blocks should expose interpretable state, defensible hypotheses, and ablations that tell us what the architecture is actually doing.

## Current Scope
This repo is in the foundation phase. The current goals are:

- lock the research agenda and paper queue
- build a clean evaluation and experiment scaffold
- reproduce one narrow baseline per track before proposing hybrids
- preserve notes, failures, and benchmark caveats as first-class artifacts

This is currently a single-owner research repo. Optimize for clarity, reproducibility, and fast iteration rather than collaboration overhead.

## Current Status
The repo is still in the foundation and reproduction stage. The immediate focus is on paper notes, benchmark selection, small reproductions, and experiment scaffolding. Full architecture implementations should follow only after a baseline and evaluation plan exist for the relevant track.

## Repository Layout
- `conf/`: Hydra config root, including `conf/track/` as the source of truth for track definitions
- `docs/`: research program, roadmap, and seed paper list
- `src/<track>/`: reusable code that belongs to one research track
- `src/shared/`: cross-track runtime, logging, and experiment helpers
- `experiments/<track>/`: track-specific experiment entry points and stubs
- `notes/<track>/`: paper notes, ablations, and failure logs organized by track
- `tests/`: regression checks for config loading, runtime helpers, and smoke runs

Start with these docs:
- [docs/research_program.md](docs/research_program.md): the thesis and track definitions
- [docs/seed_papers.md](docs/seed_papers.md): the initial reading and reproduction queue
- [docs/roadmap.md](docs/roadmap.md): the staged plan from foundation to synthesis

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make check
```

## Local Development Workflow
Always use the repository virtual environment before running Python, Hydra, Ruff, or Pytest commands.

Create it once:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Reuse it in every new shell:
```bash
source .venv/bin/activate
```

Common commands:
```bash
source .venv/bin/activate
make show-config
make lint
make test
make check
make smoke
```

If a tool like `pytest` or `ruff` is not found, the usual fix is that `.venv` is not active yet. If a Hydra command fails unexpectedly, first run `make show-config` and inspect the resolved config.

## How To Start New Work
Use this sequence for new research threads:

```bash
source .venv/bin/activate
mkdir -p notes/<track> experiments/<track> src/<track>
# add a paper note or experiment stub
make check
```

Expected pattern:
- start with a note in `notes/<track>/`
- place Hydra-owned track definitions in `conf/track/`
- place new experiment code under `experiments/memory_state/`, `experiments/adaptive_inference/`, or `experiments/hierarchical_programs/`
- move reusable code into `src/<track>/`
- only move utilities into `src/shared/` once they are reused across tracks

## Working Style
This is a research repo, not a benchmark-chasing sandbox. New work should usually begin with:

- a paper note in `notes/<track>/`
- a clear hypothesis tied to one of the three tracks
- an experiment plan with success and failure criteria
- a small baseline or reproduction before architectural variation

If a change does not clarify a hypothesis, improve the scaffold, or tighten evaluation, it is probably premature.

## Near-Term Deliverables
- a minimal experiment entry point for each track
- shared benchmark utilities for long-context, reasoning, and compositional tasks
- one reproducible baseline result per track
- a publishable record of negative results, not just wins
