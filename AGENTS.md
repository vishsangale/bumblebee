# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python-first research codebase for post-transformer architecture work. It is currently a single-owner repo, so prefer lightweight local workflow rules over multi-contributor process. Keep the project organized around the three active tracks: `memory_state`, `adaptive_inference`, and `hierarchical_programs`. Hydra config lives in `conf/`, reusable code lives in `src/<track>/` or `src/shared/`, experiment entry points live in `experiments/<track>/`, and regression tests live in `tests/`.

Use these directories consistently:
- `docs/` for the research program, roadmap, and reading list
- `conf/benchmark/` for benchmark definitions tied to a track hypothesis
- `conf/eval_suite/` for curated benchmark bundles such as `memory_state_core`
- `conf/track/` for canonical track definitions
- `conf/` for Hydra defaults, runtime settings, logging, and experiment configs
- `experiments/<track>/` for track-specific entry points and run structure
- `notes/<track>/` for paper notes, failure logs, and ablation takeaways
- `src/<track>/` for reusable code within one research track
- `src/shared/` for cross-track Python utilities
- `tests/` for lightweight checks on config, runtime, and experiment plumbing

## Build, Test, and Development Commands
Create an environment, install the package in editable mode, then use the top-level targets:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make show-config
make test
make lint
make format
make smoke
```

Before running `python`, `pytest`, `ruff`, or any `make` target that depends on them, activate the repo venv:

```bash
source .venv/bin/activate
```

If `.venv` does not exist yet, create it first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Do not rely on the system Python or globally installed `pytest` and `ruff` for this repo.

- `make test`: runs `pytest -q`
- `make lint`: runs `ruff check src tests`
- `make format`: runs `ruff format src tests`
- `make check`: runs linting and tests together
- `make show-config`: renders the resolved Hydra config for the smoke runner
- `make smoke`: runs a 1-epoch synthetic PyTorch smoke experiment with TensorBoard and checkpoints
- `make show-eval-config`: renders the resolved Hydra config for the memory evaluation runner
- `make eval-memory-smoke`: validates the memory suite with the `oracle` backend before downloading a real model

For the first real memory baseline, use:
```bash
source .venv/bin/activate
python experiments/eval_memory.py model=qwen25_3b evaluator.examples_per_benchmark=2 evaluator.context_word_steps=[128,512]
```

## Coding Style & Naming Conventions
Target Python 3.11+ and use 4-space indentation. Prefer small, composable modules over notebook-only logic. Name Python files and functions in `snake_case`, classes in `PascalCase`, and configuration files with descriptive lowercase names such as `memory_state.yaml`.

Keep research direction names stable across docs, configs, and code:
- `memory_state`
- `adaptive_inference`
- `hierarchical_programs`

When adding experiments, name directories and files after the hypothesis or mechanism, not a vague version tag. Good examples: `write_gated_memory`, `latent_halting`, `dynamic_patching`.
Keep Hydra group names aligned with directory names under `conf/`.

## Testing Guidelines
Use `pytest` for all tests. Put unit tests in `tests/` and mirror the source layout when modules grow. Name files `test_<module>.py`.

Every nontrivial change should include:
- a unit test for new behavior
- a regression test for bug fixes
- a short note in the commit message about what was validated

For research code, validation also includes documentation:
- record paper notes and reproduction ambiguities in `notes/<track>/`
- state the benchmark, metric, and baseline before adding a new model variant
- prefer small reproducible runs before large training plans

## New Experiment Rule
Every new experiment should make four things explicit:
- which track it belongs to
- what hypothesis it tests
- which benchmark and metric it uses
- what baseline it is compared against

If one of those is missing, the experiment is probably not ready to add.

For `memory_state` work, prefer the fixed evaluation suite before new model code. If a proposal has not been checked against `MQAR`, `RULER`, `NoLiMa`, or `BABILong`, treat it as a hypothesis note, not a result.

## Commit Guidelines
Use short imperative commit subjects, for example `Add Hydra smoke runner` or `Move track configs into conf`. Keep commits scoped to one change: scaffolding, docs, or experiment logic.

Each commit should make it obvious:
- what changed
- which research track it affects
- how it was validated

Before committing, run:

```bash
source .venv/bin/activate
make format
make check
```

If the change is research-driven, update `notes/<track>/`, `conf/track/`, or `docs/` in the same commit unless there is a clear reason not to.

## Research Hygiene
Do not add speculative architecture code without attaching it to a track, a paper, or a stated hypothesis. Favor grounded reproductions, ablations, and benchmark quality over rapid idea sprawl. Negative results are valuable here; preserve them in `notes/` instead of deleting evidence that a direction failed.

## Do Not Add Yet
- large training pipelines without a baseline plan
- orphan experiment scripts without notes or docs
- benchmark claims without a recorded setup, metric, and comparison target
