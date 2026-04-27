from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from shared.benchmark_registry import resolve_eval_suite

FILLER_SENTENCES = [
    "The audit log recorded a routine maintenance event with no bearing on the answer.",
    "Several unrelated notes described ordinary scheduling details and low-priority updates.",
    "A background memo summarized prior experiments, but none of those numbers matter here.",
    "The archive also contained bland procedural remarks intended only to increase context length.",
    "A reviewer added neutral commentary about formatting, naming, and archival conventions.",
]


@dataclass(frozen=True)
class MemoryTaskExample:
    benchmark: str
    context_words: int
    prompt: str
    answer: str
    metadata: dict[str, str | int]


def _filler(target_words: int, rng: random.Random) -> str:
    words: list[str] = []
    while len(words) < target_words:
        words.extend(rng.choice(FILLER_SENTENCES).split())
    return " ".join(words[:target_words])


def _append_context(base: str, extra_words: int, rng: random.Random) -> str:
    if extra_words <= 0:
        return base
    return f"{base}\n\n{_filler(extra_words, rng)}"


def _mqar_example(context_words: int, rng: random.Random, case_id: int) -> MemoryTaskExample:
    keys = [f"K{i:02d}" for i in range(12)]
    values = [f"V{i * 7 + 11}" for i in range(12)]
    rng.shuffle(keys)
    target_index = rng.randrange(len(keys))
    target_key = keys[target_index]
    target_value = values[target_index]

    lines = ["Mapping records:"]
    for key, value in zip(keys, values, strict=True):
        lines.append(f"Record: key {key} maps to value {value}.")
        lines.append(_filler(max(6, context_words // 24), rng))
    lines.append(
        f"Question: In the mapping records, what value corresponds to key {target_key}? "
        "Answer with the exact value only."
    )
    prompt = "\n".join(lines)
    return MemoryTaskExample(
        benchmark="mqar",
        context_words=context_words,
        prompt=prompt,
        answer=target_value,
        metadata={"case_id": case_id, "task_family": "associative_recall"},
    )


def _ruler_example(context_words: int, rng: random.Random, case_id: int) -> MemoryTaskExample:
    task_type = ["single_needle", "multi_needle", "tracing", "aggregation"][case_id % 4]
    if task_type == "single_needle":
        answer = f"CODE-{rng.randrange(100, 999)}"
        prompt = (
            "Long document excerpt:\n"
            f"{_filler(context_words, rng)}\n"
            f"Target memo: the verification code is {answer}.\n"
            f"{_filler(context_words // 2, rng)}\n\n"
            "Question: What is the verification code? Answer with the code only."
        )
    elif task_type == "multi_needle":
        answers = [f"TAG-{rng.randrange(10, 99)}" for _ in range(3)]
        prompt = (
            "Scattered records:\n"
            f"First target tag: {answers[0]}.\n"
            f"{_filler(context_words // 2, rng)}\n"
            f"Second target tag: {answers[1]}.\n"
            f"{_filler(context_words // 2, rng)}\n"
            f"Third target tag: {answers[2]}.\n"
            f"{_filler(context_words // 2, rng)}\n\n"
            "Question: Return the three target tags in order, separated by commas and spaces."
        )
        answer = ", ".join(answers)
    elif task_type == "tracing":
        holders = ["Ava", "Ben", "Cora", "Dax"]
        item = "packet"
        steps = [
            f"{holders[i]} passed the {item} to {holders[i + 1]}."
            for i in range(len(holders) - 1)
        ]
        prompt = (
            "Transfer log:\n"
            f"{steps[0]}\n{_filler(context_words // 3, rng)}\n"
            f"{steps[1]}\n{_filler(context_words // 3, rng)}\n"
            f"{steps[2]}\n\n"
            f"Question: Who holds the {item} at the end? Answer with the name only."
        )
        answer = holders[-1]
    else:
        values = [rng.randrange(2, 10) for _ in range(4)]
        total = sum(values)
        prompt = (
            "Distributed notes:\n"
            f"Shard A reports {values[0]} alerts.\n{_filler(context_words // 4, rng)}\n"
            f"Shard B reports {values[1]} alerts.\n{_filler(context_words // 4, rng)}\n"
            f"Shard C reports {values[2]} alerts.\n{_filler(context_words // 4, rng)}\n"
            f"Shard D reports {values[3]} alerts.\n\n"
            "Question: What is the total number of alerts? Answer with the number only."
        )
        answer = str(total)

    return MemoryTaskExample(
        benchmark="ruler",
        context_words=context_words,
        prompt=prompt,
        answer=answer,
        metadata={"case_id": case_id, "task_family": task_type},
    )


def _nolima_example(context_words: int, rng: random.Random, case_id: int) -> MemoryTaskExample:
    facts = [
        ("physician", "medical file", "dossier", "cedar locker"),
        ("chef", "recipe sheet", "cook note", "blue binder"),
        ("engineer", "design note", "schematic brief", "steel drawer"),
        ("archivist", "history packet", "heritage folder", "oak cabinet"),
    ]
    actor, query_object, context_object, answer = facts[case_id % len(facts)]
    distractors = [
        "The analyst moved an invoice into the glass case.",
        "The curator placed a poster inside the canvas tube.",
        "The librarian stored a catalog card in the red tray.",
    ]
    rng.shuffle(distractors)
    body = "\n".join(
        [
            distractors[0],
            _filler(context_words // 3, rng),
            f"The {actor} secured the {context_object} inside the {answer}.",
            _filler(context_words // 3, rng),
            distractors[1],
            _filler(context_words // 3, rng),
            distractors[2],
        ]
    )
    prompt = (
        f"Archive notes:\n{body}\n\n"
        f"Question: Which container holds the {query_object}? "
        "Answer with the container only."
    )
    return MemoryTaskExample(
        benchmark="nolima",
        context_words=context_words,
        prompt=prompt,
        answer=answer,
        metadata={"case_id": case_id, "task_family": "latent_retrieval"},
    )


def _babilong_example(context_words: int, rng: random.Random, case_id: int) -> MemoryTaskExample:
    task_type = ["fact_chain", "counting", "set_membership"][case_id % 3]
    if task_type == "fact_chain":
        holder = "Mira"
        item = "lantern"
        rooms = ["atrium", "hallway", "studio"]
        prompt = (
            "Story:\n"
            f"{holder} picked up the {item} in the {rooms[0]}.\n"
            f"{_filler(context_words // 3, rng)}\n"
            f"{holder} moved to the {rooms[1]}.\n"
            f"{_filler(context_words // 3, rng)}\n"
            f"{holder} moved to the {rooms[2]}.\n\n"
            f"Question: Where is the {item}? Answer with the location only."
        )
        answer = rooms[-1]
    elif task_type == "counting":
        counts = [1, 2, 1]
        items = ["apple", "map", "coin"]
        actor = "Noah"
        lines = [
            f"{actor} collected the {item}."
            for item, repetitions in zip(items, counts, strict=True)
            for _ in range(repetitions)
        ]
        rng.shuffle(lines)
        story = f"\n{_filler(context_words // 2, rng)}\n".join(lines)
        prompt = (
            f"Story:\n{story}\n\n"
            f"Question: How many items is {actor} carrying in total? "
            "Answer with the number only."
        )
        answer = str(sum(counts))
    else:
        owners = {"compass": "Lena", "notebook": "Iris", "scarf": "Jude"}
        facts = [f"{owner} holds the {item}." for item, owner in owners.items()]
        rng.shuffle(facts)
        prompt = (
            "Story:\n"
            + f"\n{_filler(context_words // 3, rng)}\n".join(facts)
            + "\n\nQuestion: Who holds the compass? Answer with the name only."
        )
        answer = owners["compass"]

    return MemoryTaskExample(
        benchmark="babilong",
        context_words=context_words,
        prompt=prompt,
        answer=answer,
        metadata={"case_id": case_id, "task_family": task_type},
    )


GENERATORS: dict[str, Callable[[int, random.Random, int], MemoryTaskExample]] = {
    "mqar": _mqar_example,
    "ruler": _ruler_example,
    "nolima": _nolima_example,
    "babilong": _babilong_example,
}


def normalize_answer(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed.rstrip(".!?")


def build_suite_examples(
    *,
    suite_slug: str,
    context_word_steps: list[int],
    examples_per_benchmark: int,
    seed: int,
    repo_root: str | Path | None = None,
) -> list[MemoryTaskExample]:
    resolved = resolve_eval_suite(suite_slug, repo_root=repo_root)
    rng = random.Random(seed)
    examples: list[MemoryTaskExample] = []
    for benchmark in resolved["benchmarks"]:
        slug = benchmark["slug"]
        generator = GENERATORS[slug]
        for context_words in context_word_steps:
            for case_id in range(examples_per_benchmark):
                examples.append(generator(context_words, rng, case_id))
    return examples


def write_predictions_jsonl(path: str | Path, rows: list[dict[str, object]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return output_path


def example_to_dict(example: MemoryTaskExample) -> dict[str, object]:
    return asdict(example)
