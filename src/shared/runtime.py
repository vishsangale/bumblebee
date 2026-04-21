from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    checkpoint_dir: Path
    tensorboard_dir: Path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def current_run_dir() -> Path:
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        return Path.cwd()


def prepare_run_artifacts(runtime_cfg, *, run_dir: str | Path | None = None) -> RunArtifacts:
    base_dir = ensure_dir(run_dir or current_run_dir())
    checkpoint_dir = ensure_dir(base_dir / runtime_cfg.checkpoint_dirname)
    tensorboard_dir = ensure_dir(base_dir / runtime_cfg.tensorboard_dirname)
    return RunArtifacts(
        run_dir=base_dir,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
    )


def save_checkpoint(state: dict, path: str | Path) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    return checkpoint_path
