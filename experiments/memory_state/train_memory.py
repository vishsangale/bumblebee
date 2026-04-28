# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports from memory_state and shared packages
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_original_expand_help = argparse.HelpFormatter._expand_help


def _patched_expand_help(self, action):
    if isinstance(action.help, (str, bytes)):
        return _original_expand_help(self, action)
    return None


argparse.HelpFormatter._expand_help = _patched_expand_help

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig
from shared.runtime import prepare_run_artifacts, save_checkpoint
from experiments.memory_state.data import synthetic_batch, TokenDataset


def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(cfg: DictConfig) -> MemoryTransformer:
    model_cfg = MemoryTransformerConfig(
        vocab_size=int(cfg.model.vocab_size),
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        d_ffn=int(cfg.model.d_ffn),
        max_seq_len=int(cfg.model.max_seq_len),
        dropout=float(cfg.model.dropout),
        memory_mlp_size=int(cfg.model.memory_mlp_size),
        memory_every_n_layers=int(cfg.model.memory_every_n_layers),
        memory_decay_init=float(cfg.model.memory_decay_init),
    )
    use_memory = bool(cfg.experiment.use_memory)
    model = MemoryTransformer(model_cfg, use_memory=use_memory)
    gate_disabled = bool(cfg.experiment.get("gate_disabled", False))
    if use_memory and gate_disabled:
        for memory_module in model.memory_modules:
            memory_module.gate.set_force_open(True)
    return model


@hydra.main(version_base="1.3", config_path="../../conf", config_name="train_memory")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.trainer.seed))
    device = select_device(str(cfg.trainer.device))
    artifacts = prepare_run_artifacts(cfg.runtime)

    model = build_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model_params={num_params:,} use_memory={cfg.experiment.use_memory}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
        betas=(0.9, 0.95),
    )

    data_path = REPO_ROOT / "data" / "fineweb_train.bin"
    use_real_data = data_path.exists()
    if use_real_data:
        dataset = TokenDataset(data_path, int(cfg.trainer.seq_len))
        print(f"Using real data: {data_path} ({len(dataset):,} chunks)")
    else:
        print("data/fineweb_train.bin not found — using synthetic data (smoke mode)")
        print("To prepare real data: python experiments/memory_state/data.py --prepare")

    writer = SummaryWriter(log_dir=str(artifacts.tensorboard_dir))
    writer.add_text("config/resolved", OmegaConf.to_yaml(cfg, resolve=True), 0)

    batch_size = int(cfg.trainer.batch_size)
    seq_len = int(cfg.trainer.seq_len)
    vocab_size = int(cfg.model.vocab_size)
    grad_clip = float(cfg.trainer.grad_clip)
    warmup = int(cfg.trainer.warmup_steps)
    max_steps = int(cfg.trainer.max_steps)

    # Resume from checkpoint if specified
    start_step = 0
    start_token = 0
    resume_from = cfg.runtime.get("resume_from", None)
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        start_token = int(ckpt.get("tokens_consumed", start_step * batch_size * seq_len))
        print(f"resumed from {resume_from} at step={start_step} token={start_token:,}")

    model.train()
    step = start_step
    while step < max_steps:
        # Learning rate warmup
        lr = float(cfg.trainer.learning_rate) * min(1.0, step / max(warmup, 1))
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Reset memory at sequence boundaries (each training batch = fresh sequence)
        if hasattr(model, "reset_memory"):
            model.reset_memory()

        # Fetch batch — cursor is absolute token position, safe to resume with different batch_size
        if use_real_data:
            token_offset = start_token + (step - start_step) * batch_size * seq_len
            tokens = dataset.get_batch(token_offset, batch_size, device)
        else:
            tokens = synthetic_batch(batch_size, seq_len, vocab_size, device)

        input_ids = tokens[:, :seq_len]
        targets = tokens[:, 1 : seq_len + 1]

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)  # (B, T, V)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), targets.reshape(-1)
        )
        assoc_weight = float(cfg.trainer.get("assoc_loss_weight", 0.0))
        if assoc_weight > 0 and hasattr(model, "last_assoc_loss"):
            loss = loss + assoc_weight * model.last_assoc_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        step += 1

        if step % int(cfg.trainer.log_every_n_steps) == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            print(f"step={step} loss={loss.item():.4f} lr={lr:.2e}")

        if step % int(cfg.trainer.save_every_n_steps) == 0:
            tokens_consumed = start_token + (step - start_step) * batch_size * seq_len
            state = {
                "step": step,
                "tokens_consumed": tokens_consumed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            save_checkpoint(state, artifacts.checkpoint_dir / f"step_{step:07d}.pt")

    writer.close()
    print(f"run_dir={artifacts.run_dir}")


if __name__ == "__main__":
    main()
