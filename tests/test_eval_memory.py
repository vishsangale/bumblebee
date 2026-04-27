import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eval_memory_oracle_backend_writes_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "eval_memory_run"
    command = [
        sys.executable,
        "experiments/eval_memory.py",
        "model=oracle",
        "evaluator.examples_per_benchmark=1",
        "evaluator.context_word_steps=[64]",
        f"hydra.run.dir={run_dir}",
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = run_dir / "results" / "summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["overall_accuracy"] == 1.0
    assert summary["protocol"] == "proxy_v0"
