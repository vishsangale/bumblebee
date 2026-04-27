from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def benchmark_config_dir(*, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / "conf" / "benchmark"


def eval_suite_config_dir(*, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / "conf" / "eval_suite"


def list_benchmark_specs(*, repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    benchmarks: list[dict[str, Any]] = []
    for path in sorted(benchmark_config_dir(repo_root=repo_root).glob("*.yaml")):
        cfg = OmegaConf.load(path)
        benchmarks.append(OmegaConf.to_container(cfg, resolve=True))
    return benchmarks


def get_benchmark_spec(slug: str, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    for benchmark in list_benchmark_specs(repo_root=repo_root):
        if benchmark["slug"] == slug:
            return benchmark
    raise KeyError(f"Unknown benchmark: {slug}")


def list_eval_suites(*, repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    suites: list[dict[str, Any]] = []
    for path in sorted(eval_suite_config_dir(repo_root=repo_root).glob("*.yaml")):
        cfg = OmegaConf.load(path)
        suites.append(OmegaConf.to_container(cfg, resolve=True))
    return suites


def get_eval_suite(slug: str, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    for suite in list_eval_suites(repo_root=repo_root):
        if suite["slug"] == slug:
            return suite
    raise KeyError(f"Unknown eval suite: {slug}")


def resolve_eval_suite(slug: str, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    suite = get_eval_suite(slug, repo_root=repo_root)
    benchmarks = [
        get_benchmark_spec(benchmark_slug, repo_root=repo_root)
        for benchmark_slug in suite["benchmarks"]
    ]
    return {"suite": suite, "benchmarks": benchmarks}
