# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import hydra
from omegaconf import DictConfig, OmegaConf

from memory_state.proxy_tasks import (
    build_suite_examples,
    example_to_dict,
    normalize_answer,
    write_predictions_jsonl,
)
from shared.hf_inference import load_text_generator
from shared.runtime import current_run_dir, ensure_dir


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["benchmark"]), int(row["context_words"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (benchmark, context_words), items in sorted(grouped.items()):
        correct = sum(1 for item in items if bool(item["correct"]))
        summaries.append(
            {
                "benchmark": benchmark,
                "context_words": context_words,
                "num_examples": len(items),
                "accuracy": correct / len(items),
                "avg_input_tokens": sum(int(item["input_tokens"]) for item in items) / len(items),
                "avg_output_tokens": sum(int(item["output_tokens"]) for item in items) / len(items),
            }
        )

    overall_correct = sum(1 for row in rows if bool(row["correct"]))
    return {
        "num_examples": len(rows),
        "overall_accuracy": overall_correct / len(rows) if rows else 0.0,
        "slices": summaries,
    }


@hydra.main(version_base="1.3", config_path="../conf", config_name="evaluate_memory")
def main(cfg: DictConfig) -> None:
    run_dir = ensure_dir(current_run_dir())
    results_dir = ensure_dir(run_dir / "results")

    examples = build_suite_examples(
        suite_slug=str(cfg.evaluator.suite),
        context_word_steps=[int(value) for value in cfg.evaluator.context_word_steps],
        examples_per_benchmark=int(cfg.evaluator.examples_per_benchmark),
        seed=int(cfg.evaluator.seed),
        repo_root=ROOT,
    )
    generator = load_text_generator(cfg.model)

    rows: list[dict[str, object]] = []
    for example in examples:
        response = generator.generate(example.prompt, answer=example.answer)
        prediction = normalize_answer(response.text)
        target = normalize_answer(example.answer)
        row = {
            **example_to_dict(example),
            "prediction": response.text,
            "normalized_prediction": prediction,
            "normalized_answer": target,
            "correct": prediction == target,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "protocol": str(cfg.evaluator.protocol),
            "model_name": str(cfg.model.name),
            "model_id": str(cfg.model.model_id),
        }
        rows.append(row)
        print(
            f"benchmark={example.benchmark} context_words={example.context_words} "
            f"target={example.answer!r} prediction={response.text!r} correct={row['correct']}"
        )

    summary = {
        "suite": str(cfg.evaluator.suite),
        "protocol": str(cfg.evaluator.protocol),
        "model_name": str(cfg.model.name),
        "model_id": str(cfg.model.model_id),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": summarize(rows),
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if bool(cfg.evaluator.save_predictions):
        predictions_path = write_predictions_jsonl(results_dir / "predictions.jsonl", rows)
        print(f"predictions_path={predictions_path}")

    print(f"summary_path={summary_path}")
    print(f"overall_accuracy={summary['metrics']['overall_accuracy']:.3f}")


if __name__ == "__main__":
    main()
