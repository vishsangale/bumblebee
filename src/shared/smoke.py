from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_synthetic_loaders(experiment_cfg, trainer_cfg) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(int(trainer_cfg.seed))
    total = int(experiment_cfg.num_train) + int(experiment_cfg.num_val)
    features = torch.randn(int(total), int(experiment_cfg.input_dim), generator=generator)

    weights = torch.randn(
        int(experiment_cfg.input_dim),
        int(experiment_cfg.num_classes),
        generator=generator,
    )
    bias = torch.randn(int(experiment_cfg.num_classes), generator=generator)
    labels = (features @ weights + bias).argmax(dim=-1)

    train_size = int(experiment_cfg.num_train)
    train_ds = TensorDataset(features[:train_size], labels[:train_size])
    val_ds = TensorDataset(features[train_size:], labels[train_size:])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(trainer_cfg.batch_size),
        shuffle=True,
        num_workers=int(trainer_cfg.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(trainer_cfg.batch_size),
        shuffle=False,
        num_workers=int(trainer_cfg.num_workers),
    )
    return train_loader, val_loader
