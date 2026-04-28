"""
Inspect a memory experiment checkpoint: weight norms, memory MLP stats.

Usage:
    python experiments/memory_state/checkpoint_inspect.py <checkpoint.pt>
    python experiments/memory_state/checkpoint_inspect.py <checkpoint.pt> --out report.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch


def _load(ckpt_path: Path) -> dict:
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _tensor_stats(t: torch.Tensor) -> dict:
    t = t.float()
    return {
        "norm": t.norm().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "max": t.abs().max().item(),
    }


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def inspect_checkpoint(ckpt_path: Path, out_path: Path | None = None) -> None:
    ckpt = _load(ckpt_path)
    sd = ckpt["model_state_dict"]
    cfg = ckpt.get("config", {})
    exp_cfg = cfg.get("experiment", {})

    _print_section("Run info")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  step       : {ckpt['step']}")
    print(f"  experiment : {exp_cfg.get('name', '?')}")
    print(f"  use_memory : {exp_cfg.get('use_memory', '?')}")

    # ── Per-layer attention and FFN norms ─────────────────────────────────────
    _print_section("Per-layer weight norms")
    layer_indices = sorted({
        int(k.split(".")[1])
        for k in sd if k.startswith("blocks.")
    })

    attn_norms, ffn_norms = [], []
    fmt = "{:>4}  {:>10}  {:>10}  {:>10}  {:>10}"
    print(fmt.format("lyr", "qkv", "attn_out", "ffn_up", "ffn_down"))
    print("-" * 52)
    for i in layer_indices:
        qkv  = sd[f"blocks.{i}.attn.qkv.weight"].float().norm().item()
        aout = sd[f"blocks.{i}.attn.out.weight"].float().norm().item()
        fu   = sd[f"blocks.{i}.ffn.0.weight"].float().norm().item()
        fd   = sd[f"blocks.{i}.ffn.2.weight"].float().norm().item()
        print(fmt.format(i, f"{qkv:.3f}", f"{aout:.3f}", f"{fu:.3f}", f"{fd:.3f}"))
        attn_norms.append((qkv + aout) / 2)
        ffn_norms.append((fu + fd) / 2)

    # ── Embedding and head ────────────────────────────────────────────────────
    _print_section("Embeddings / head")
    for k in ["tok_emb.weight", "pos_emb.weight", "ln_f.weight", "head.weight"]:
        if k in sd:
            s = _tensor_stats(sd[k])
            print(f"  {k:30s}  norm={s['norm']:.3f}  std={s['std']:.4f}")

    # ── Memory modules (Titans / Gated) ───────────────────────────────────────
    memory_keys = [k for k in sd if "memory_modules" in k]
    if memory_keys:
        _print_section("Memory modules")
        # group by module index
        mem_indices = sorted({
            int(k.split(".")[1])
            for k in memory_keys
        })
        for mi in mem_indices:
            prefix = f"memory_modules.{mi}"
            print(f"\n  Module {mi}:")
            for k, v in sd.items():
                if k.startswith(prefix):
                    short = k[len(prefix) + 1:]
                    s = _tensor_stats(v)
                    print(f"    {short:50s}  norm={s['norm']:.4f}  "
                          f"std={s['std']:.6f}  max={s['max']:.4f}")

        # gate bias vs init (decay_init ≈ raw_bias before sigmoid)
        gate_bias_keys = [k for k in memory_keys if "gate" in k and "bias" in k]
        if gate_bias_keys:
            print("\n  Gate sigmoid outputs (from biases):")
            for k in sorted(gate_bias_keys):
                bias = sd[k].float()
                open_frac = torch.sigmoid(bias).mean().item()
                print(f"    {k:55s}  mean_gate={open_frac:.4f}")
    else:
        print("\n  (no memory modules — baseline checkpoint)")

    # ── Total parameter count ─────────────────────────────────────────────────
    total_params = sum(v.numel() for v in sd.values())
    print(f"\n  Total stored tensors : {len(sd)}")
    print(f"  Total parameters     : {total_params:,}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_attn = fig.add_subplot(gs[0, 0])
    ax_ffn  = fig.add_subplot(gs[0, 1])

    layers = list(layer_indices)
    ax_attn.bar(layers, attn_norms, color="#4e79a7", alpha=0.85)
    ax_attn.set_title("Avg attn weight norm (qkv + out) / 2")
    ax_attn.set_xlabel("Layer")
    ax_attn.set_ylabel("L2 norm")
    ax_attn.grid(True, axis="y", alpha=0.3)

    ax_ffn.bar(layers, ffn_norms, color="#f28e2b", alpha=0.85)
    ax_ffn.set_title("Avg FFN weight norm (up + down) / 2")
    ax_ffn.set_xlabel("Layer")
    ax_ffn.set_ylabel("L2 norm")
    ax_ffn.grid(True, axis="y", alpha=0.3)

    exp_name = exp_cfg.get("name", ckpt_path.parent.parent.name)
    fig.suptitle(f"{exp_name}  —  step {ckpt['step']:,}", fontsize=12)
    fig.tight_layout()

    if out_path is None:
        out_path = ckpt_path.parent / (ckpt_path.stem + "_inspect.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"File not found: {args.checkpoint}")
        sys.exit(1)

    inspect_checkpoint(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
