from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def track_config_dir(*, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / "conf" / "track"


def list_track_specs(*, repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for path in sorted(track_config_dir(repo_root=repo_root).glob("*.yaml")):
        cfg = OmegaConf.load(path)
        tracks.append(OmegaConf.to_container(cfg, resolve=True))
    return tracks


def get_track_spec(slug: str, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    for track in list_track_specs(repo_root=repo_root):
        if track["slug"] == slug:
            return track
    raise KeyError(f"Unknown track: {slug}")
