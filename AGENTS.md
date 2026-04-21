# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python-first research codebase for post-transformer architecture work. Keep research plans in `docs/`, machine-readable track definitions in `configs/tracks/`, reusable code in `src/bumblebee/`, and lightweight tests in `tests/`.

Use these directories consistently:
- `docs/` for the research program, roadmap, and reading list
- `configs/tracks/` for YAML briefs describing each architecture direction
- `experiments/` for experiment entry points and run structure
- `notes/` for paper notes, failure logs, and ablation takeaways
- `src/bumblebee/` for shared Python utilities and registries

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

## Testing Guidelines
Use `pytest` for all tests. Put unit tests in `tests/` and mirror the source layout when modules grow. Name files `test_<module>.py`.

Every nontrivial change should include:
- a unit test for new behavior
- a regression test for bug fixes
- a short note in the PR or commit message about what was validated

## Commit & Pull Request Guidelines
Use short imperative commit subjects, for example `Add research track registry` or `Document seed paper list`. Keep commits scoped to one change: scaffolding, docs, or experiment logic.

Pull requests should include:
- a brief summary
- affected research track or benchmark
- validation notes such as `pytest -q`
- links to papers or issues when the change is research-driven
