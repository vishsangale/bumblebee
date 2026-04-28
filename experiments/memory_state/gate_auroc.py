"""
Compute gate AUROC on bAbI state-mutation tokens.

Usage:
  python experiments/memory_state/gate_auroc.py \
    --checkpoint outputs/memory_state/train/memory_lm_100m/<run>/checkpoints/best.pt \
    --config conf/train_memory.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from sklearn.metrics import roc_auc_score

from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig
from experiments.memory_state.data import TokenDataset


def load_model(checkpoint_path: str | Path, config_path: str | Path) -> MemoryTransformer:
    import hydra
    from omegaconf import OmegaConf

    with hydra.initialize(config_path=str(Path(config_path).parent), version_base="1.3"):
        cfg = hydra.compose(config_name=Path(config_path).name)

    model_cfg = MemoryTransformerConfig(
        vocab_size=int(cfg.model.vocab_size),
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        d_ffn=int(cfg.model.d_ffn),
        max_seq_len=int(cfg.model.max_seq_len),
        dropout=0.0,
        memory_mlp_size=int(cfg.model.memory_mlp_size),
        memory_every_n_layers=int(cfg.model.memory_every_n_layers),
        memory_decay_init=float(cfg.model.memory_decay_init),
    )
    use_memory = bool(cfg.experiment.use_memory)
    model = MemoryTransformer(model_cfg, use_memory=use_memory)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_auroc_for_example(model: MemoryTransformer, example_text: str) -> float | None:
    """Compute AUROC for gate activations on a single example."""
    import tiktoken
    from sklearn.metrics import roc_auc_score

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(example_text)
    input_ids = torch.tensor([ids])

    if hasattr(model, "reset_memory"):
        model.reset_memory()

    with torch.no_grad():
        model(input_ids)

    gate_activations = model.get_gate_activations()
    if gate_activations is None:
        return None

    # Combine all memory module activations (shape: [num_modules, T, 1])
    combined = torch.cat(gate_activations, dim=0)  # (num_modules * T, 1)
    gate_vals = combined.squeeze(-1).cpu().numpy()

    # Identify mutation token positions via string search
    labels = []
    for phrase in ["moved to the"]:
        if phrase in example_text:
            phrase_ids = enc.encode(phrase)
            # Find all occurrences
            for i in range(len(ids) - len(phrase_ids) + 1):
                if ids[i:i + len(phrase_ids)] == phrase_ids:
                    # Mark tokens in the phrase as mutations
                    for j in range(len(phrase_ids)):
                        labels.append(1)
                else:
                    labels.append(0)

    if len(labels) == 0 or sum(labels) == 0:
        return None

    # Sample same number of non-mutation tokens
    num_mutations = sum(labels)
    non_mutation_indices = [i for i, l in enumerate(labels) if l == 0]
    if len(non_mutation_indices) < num_mutations:
        return None

    sampled_indices = non_mutation_indices[:num_mutations]
    all_indices = [i for i, l in enumerate(labels) if l == 1] + sampled_indices
    all_labels = [labels[i] for i in all_indices]
    all_gates = [gate_vals[i] for i in all_indices]

    if len(set(all_labels)) == 1:
        return None

    try:
        auroc = roc_auc_score(all_labels, all_gates)
        return auroc
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to train_memory.yaml config")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.config)

    # Load test examples from eval_memory
    from memory_state.proxy_tasks import build_suite_examples

    examples = build_suite_examples(
        suite_slug="memory_state_core",
        context_word_steps=[512],
        examples_per_benchmark=10,
        seed=42,
        repo_root=ROOT,
    )

    aurocs = []
    for example in examples:
        if example.benchmark != "babilong":
            continue
        auroc = compute_auroc_for_example(model, example.prompt)
        if auroc is not None:
            aurocs.append(auroc)
            print(f"{example.metadata.get('task_family', 'unknown')}: AUROC={auroc:.3f}")

    if aurocs:
        print(f"\nMean AUROC: {sum(aurocs) / len(aurocs):.3f}")
    else:
        print("No valid AUROCs computed (no mutation phrases found or gate activations empty)")


if __name__ == "__main__":
    main()
