from pathlib import Path

import torch
from omegaconf import OmegaConf

from shared.runtime import prepare_run_artifacts, save_checkpoint


def test_prepare_run_artifacts_creates_expected_dirs(tmp_path: Path) -> None:
    runtime_cfg = OmegaConf.create(
        {
            "checkpoint_dirname": "checkpoints",
            "tensorboard_dirname": "tensorboard",
        }
    )

    artifacts = prepare_run_artifacts(runtime_cfg, run_dir=tmp_path / "run")

    assert artifacts.run_dir == tmp_path / "run"
    assert artifacts.checkpoint_dir == tmp_path / "run" / "checkpoints"
    assert artifacts.tensorboard_dir == tmp_path / "run" / "tensorboard"
    assert artifacts.checkpoint_dir.is_dir()
    assert artifacts.tensorboard_dir.is_dir()


def test_save_checkpoint_writes_serialized_payload(tmp_path: Path) -> None:
    checkpoint_path = save_checkpoint(
        {"step": 1, "weights": torch.tensor([1.0])},
        tmp_path / "ckpt.pt",
    )

    payload = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint_path == tmp_path / "ckpt.pt"
    assert payload["step"] == 1
