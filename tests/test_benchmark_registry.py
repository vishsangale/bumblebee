from pathlib import Path

from shared.benchmark_registry import list_benchmark_specs, resolve_eval_suite

ROOT = Path(__file__).resolve().parents[1]


def test_memory_state_benchmark_specs_exist() -> None:
    slugs = {spec["slug"] for spec in list_benchmark_specs(repo_root=ROOT)}
    assert {"mqar", "ruler", "nolima", "babilong"} <= slugs


def test_memory_state_core_suite_resolves_to_known_benchmarks() -> None:
    resolved = resolve_eval_suite("memory_state_core", repo_root=ROOT)

    assert resolved["suite"]["track"] == "memory_state"
    assert [benchmark["slug"] for benchmark in resolved["benchmarks"]] == [
        "mqar",
        "ruler",
        "nolima",
        "babilong",
    ]
    urls = [benchmark["source"]["url"] for benchmark in resolved["benchmarks"]]
    assert all(url.startswith("https://arxiv.org/abs/") for url in urls)
