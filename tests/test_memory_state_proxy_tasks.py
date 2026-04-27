from pathlib import Path

from memory_state.proxy_tasks import build_suite_examples, normalize_answer

ROOT = Path(__file__).resolve().parents[1]


def test_build_suite_examples_covers_all_memory_benchmarks() -> None:
    examples = build_suite_examples(
        suite_slug="memory_state_core",
        context_word_steps=[64, 128],
        examples_per_benchmark=2,
        seed=0,
        repo_root=ROOT,
    )

    assert len(examples) == 16
    assert {example.benchmark for example in examples} == {"mqar", "ruler", "nolima", "babilong"}


def test_normalize_answer_keeps_simple_exact_match_stable() -> None:
    assert normalize_answer(" Cedar Locker.\n") == "cedar locker"
    assert normalize_answer("TAG-11, TAG-12, TAG-13") == "tag-11, tag-12, tag-13"
