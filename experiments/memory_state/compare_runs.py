"""
Compare loss curves across memory experiment runs.

Usage:
    python experiments/memory_state/compare_runs.py                  # auto-discover today's runs
    python experiments/memory_state/compare_runs.py --date 2026-04-28
    python experiments/memory_state/compare_runs.py --dirs <dir1> <dir2> ...
    python experiments/memory_state/compare_runs.py --out my_plot.png
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TRAIN_ROOT = REPO_ROOT / "outputs" / "memory_state" / "train" / "memory_lm_100m"

EXPERIMENT_COLORS = {
    "lm_baseline": "#4e79a7",
    "lm_titans": "#f28e2b",
    "lm_gated": "#59a14f",
}


def _read_experiment_name(run_dir: Path) -> str:
    overrides = run_dir / ".hydra" / "overrides.yaml"
    if overrides.exists():
        for line in overrides.read_text().splitlines():
            line = line.strip().lstrip("- ")
            if line.startswith("experiment="):
                return line.split("=", 1)[1]
    return run_dir.parent.name + "/" + run_dir.name


def _load_scalars(run_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        return [], []
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def discover_runs(date: str | None = None) -> list[Path]:
    if date is None:
        date = datetime.date.today().isoformat()
    date_dir = TRAIN_ROOT / date
    if not date_dir.exists():
        return []
    runs = sorted(date_dir.iterdir())
    return [r for r in runs if (r / "tensorboard").exists()]


def plot_compare(
    run_dirs: list[Path],
    tag: str = "train/loss",
    out_path: Path | None = None,
    smoothing: float = 0.0,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    plotted = 0
    for run_dir in run_dirs:
        name = _read_experiment_name(run_dir)
        steps, values = _load_scalars(run_dir, tag)
        if not steps:
            print(f"  [skip] no data for {tag} in {run_dir}")
            continue

        color = EXPERIMENT_COLORS.get(name, None)

        if smoothing > 0:
            # exponential moving average
            smoothed = []
            ema = values[0]
            for v in values:
                ema = smoothing * ema + (1 - smoothing) * v
                smoothed.append(ema)
            ax.plot(steps, smoothed, label=name, color=color, linewidth=2)
            ax.plot(steps, values, alpha=0.25, color=color, linewidth=0.8)
        else:
            ax.plot(steps, values, label=name, color=color, linewidth=2)

        last_step, last_val = steps[-1], values[-1]
        ax.annotate(
            f"{last_val:.3f}",
            xy=(last_step, last_val),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=color or "black",
            va="center",
        )
        print(f"  {name}: {len(steps)} pts, step {steps[0]}–{last_step}, "
              f"loss {values[0]:.3f}→{last_val:.3f}")
        plotted += 1

    if plotted == 0:
        print("No data found.")
        return

    ax.set_xlabel("Step")
    ax.set_ylabel(tag.replace("/", " / "))
    ax.set_title("Memory experiment — training loss comparison")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = REPO_ROOT / "outputs" / "memory_state" / "compare_loss.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--dirs", nargs="+", type=Path, help="Explicit run dirs")
    parser.add_argument("--tag", default="train/loss")
    parser.add_argument("--smooth", type=float, default=0.6,
                        help="EMA smoothing coefficient (0=off, 0.9=heavy)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.dirs:
        run_dirs = [Path(d) for d in args.dirs]
    else:
        run_dirs = discover_runs(args.date)
        if not run_dirs:
            print(f"No runs found under {TRAIN_ROOT}/{args.date or 'today'}")
            sys.exit(1)

    print(f"Found {len(run_dirs)} run(s):")
    for d in run_dirs:
        print(f"  {d}")
    print()

    plot_compare(run_dirs, tag=args.tag, out_path=args.out, smoothing=args.smooth)


if __name__ == "__main__":
    main()
