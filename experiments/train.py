# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import argparse

_original_expand_help = argparse.HelpFormatter._expand_help


def _patched_expand_help(self, action):
    if isinstance(action.help, (str, bytes)):
        _original_expand_help(self, action)


argparse.HelpFormatter._expand_help = _patched_expand_help

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from shared.runtime import prepare_run_artifacts, save_checkpoint
from shared.smoke import TinyClassifier, build_synthetic_loaders, select_device


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += labels.size(0)
    return total_loss / total_examples, total_correct / total_examples


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.trainer.seed))
    device = select_device(str(cfg.trainer.device))
    artifacts = prepare_run_artifacts(cfg.runtime)

    train_loader, val_loader = build_synthetic_loaders(cfg.experiment, cfg.trainer)
    model = TinyClassifier(
        input_dim=int(cfg.experiment.input_dim),
        hidden_dim=int(cfg.experiment.hidden_dim),
        num_classes=int(cfg.experiment.num_classes),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(
        log_dir=str(artifacts.tensorboard_dir),
        flush_secs=int(cfg.logging.flush_secs),
    )
    if bool(cfg.logging.log_config):
        writer.add_text("config/resolved", OmegaConf.to_yaml(cfg, resolve=True), 0)
    writer.add_text("meta/track", str(cfg.track.slug), 0)
    writer.add_text("meta/track_name", str(cfg.track.name), 0)

    best_val_acc = -1.0
    checkpoint_state: dict = {}
    global_step = 0
    for epoch in range(1, int(cfg.trainer.max_epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_examples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_examples += batch_size
            global_step += 1

            if global_step % int(cfg.trainer.log_every_n_steps) == 0:
                writer.add_scalar("train/step_loss", loss.item(), global_step)

        train_loss = train_loss_sum / train_examples
        train_acc = train_correct / train_examples
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)

        print(
            f"epoch={epoch} "
            f"track={cfg.track.slug} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}"
        )

        checkpoint_state = {
            "epoch": epoch,
            "track": str(cfg.track.slug),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if epoch % int(cfg.trainer.save_every_n_epochs) == 0:
            save_checkpoint(checkpoint_state, artifacts.checkpoint_dir / f"epoch_{epoch:03d}.pt")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_path = save_checkpoint(
                checkpoint_state,
                artifacts.checkpoint_dir / str(cfg.runtime.best_checkpoint_name),
            )
            print(f"saved best checkpoint: {best_path}")

    last_path = save_checkpoint(
        checkpoint_state,
        artifacts.checkpoint_dir / str(cfg.runtime.last_checkpoint_name),
    )
    writer.close()

    print(f"run_dir={artifacts.run_dir}")
    print(f"tensorboard_dir={artifacts.tensorboard_dir}")
    print(f"last_checkpoint={last_path}")


if __name__ == "__main__":
    main()
