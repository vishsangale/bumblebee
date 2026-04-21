import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_train_creates_hydra_run_dir_and_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "smoke_run"
    command = [
        sys.executable,
        "experiments/train.py",
        "track=memory_state",
        "trainer.max_epochs=1",
        "trainer.batch_size=8",
        "trainer.log_every_n_steps=1",
        "experiment.num_train=32",
        "experiment.num_val=16",
        "experiment.hidden_dim=8",
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
    assert (run_dir / ".hydra" / "config.yaml").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert list((run_dir / "tensorboard").glob("events.out.tfevents.*"))
