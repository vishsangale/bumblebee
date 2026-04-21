# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python-first research codebase for post-transformer architecture work. It is currently a single-owner repo, so prefer lightweight local workflow rules over multi-contributor process. Keep the project organized around the three active tracks: `memory_state`, `adaptive_inference`, and `hierarchical_programs`. Research plans live in `docs/`, machine-readable track definitions live in `configs/tracks/`, reusable code lives in `src/bumblebee/`, and regression tests live in `tests/`.

Use these directories consistently:
- `docs/` for the research program, roadmap, and reading list
- `configs/tracks/` for YAML briefs describing each architecture direction
- `experiments/` for experiment entry points and run structure
- `notes/` for paper notes, failure logs, and ablation takeaways
- `src/bumblebee/` for shared Python utilities and registries
- `tests/` for lightweight checks on registries, loaders, and experiment plumbing

## Build, Test, and Development Commands
Create an environment, install the package in editable mode, then use the top-level targets:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test
make lint
make format
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

## Coding Style & Naming Conventions
Target Python 3.11+ and use 4-space indentation. Prefer small, composable modules over notebook-only logic. Name Python files and functions in `snake_case`, classes in `PascalCase`, and configuration files with descriptive lowercase names such as `memory_state.yaml`.

Keep research direction names stable across docs, configs, and code:
- `memory_state`
- `adaptive_inference`
- `hierarchical_programs`

When adding experiments, name directories and files after the hypothesis or mechanism, not a vague version tag. Good examples: `write_gated_memory`, `latent_halting`, `dynamic_patching`.

## Testing Guidelines
Use `pytest` for all tests. Put unit tests in `tests/` and mirror the source layout when modules grow. Name files `test_<module>.py`.

Every nontrivial change should include:
- a unit test for new behavior
- a regression test for bug fixes
- a short note in the PR or commit message about what was validated

For research code, validation also includes documentation:
- record paper notes and reproduction ambiguities in `notes/`
- state the benchmark, metric, and baseline before adding a new model variant
- prefer small reproducible runs before large training plans

## New Experiment Rule
Every new experiment should make four things explicit:
- which track it belongs to
- what hypothesis it tests
- which benchmark and metric it uses
- what baseline it is compared against

If one of those is missing, the experiment is probably not ready to add.

## Commit Guidelines
Use short imperative commit subjects, for example `Add research track registry` or `Document seed paper list`. Keep commits scoped to one change: scaffolding, docs, or experiment logic.

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

If the change is research-driven, update `notes/` or `docs/` in the same commit unless there is a clear reason not to.

## Research Hygiene
Do not add speculative architecture code without attaching it to a track, a paper, or a stated hypothesis. Favor grounded reproductions, ablations, and benchmark quality over rapid idea sprawl. Negative results are valuable here; preserve them in `notes/` instead of deleting evidence that a direction failed.

## Do Not Add Yet
- large training pipelines without a baseline plan
- orphan experiment scripts without notes or docs
- benchmark claims without a recorded setup, metric, and comparison target
