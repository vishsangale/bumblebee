from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from experiments.memory_state.train_memory import build_model
from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig

ROOT = Path(__file__).resolve().parents[1]


def _compose_config(config_name: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
            return compose(
                config_name=config_name,
                overrides=overrides or [],
                return_hydra_config=True,
            )
    finally:
        GlobalHydra.instance().clear()


def _tiny_memory_cfg(**overrides: int | float) -> MemoryTransformerConfig:
    values = {
        "vocab_size": 128,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "d_ffn": 32,
        "max_seq_len": 8,
        "dropout": 0.0,
        "memory_mlp_size": 4,
        "memory_layer": 0,
        "memory_decay_init": 0.99,
    }
    values.update(overrides)
    return MemoryTransformerConfig(**values)


def test_memory_transformer_language_loss_trains_wq() -> None:
    # In MAC mode, W_Q is in the LM-loss gradient path via the h-token prefix
    # fed into attention. W1/W2 are buffers (not outer-loop params); the write
    # gate receives no direct LM-loss gradient (it acts through future memory state).
    model = MemoryTransformer(_tiny_memory_cfg(), use_memory=True)
    input_ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    targets = torch.randint(0, model.cfg.vocab_size, (2, 8))

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.cfg.vocab_size),
        targets.reshape(-1),
    )
    loss.backward()

    memory_module = model.memory_modules[0]
    wq_grad = memory_module.memory.W_Q.weight.grad
    assert wq_grad is not None, "W_Q must receive LM-loss gradient via MAC attention prefix"
    assert wq_grad.abs().sum().item() > 0


def test_train_memory_config_uses_memory_model_and_memory_trainer() -> None:
    cfg = _compose_config("train_memory")

    assert cfg.model.name == "memory_lm_100m"
    assert cfg.trainer.save_every_n_steps > 0
    assert cfg.trainer.max_steps > 0
    hydra_run = OmegaConf.to_container(cfg.hydra.run, resolve=False)
    assert "${track.slug}/train/${model.name}" in hydra_run["dir"]


def test_train_memory_script_runs_documented_file_entrypoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "memory_train_run"
    command = [
        sys.executable,
        "experiments/memory_state/train_memory.py",
        "trainer.max_steps=1",
        "trainer.log_every_n_steps=1",
        "trainer.save_every_n_steps=1",
        "trainer.batch_size=1",
        "trainer.seq_len=8",
        "model.vocab_size=128",
        "model.d_model=16",
        "model.n_heads=4",
        "model.n_layers=1",
        "model.d_ffn=32",
        "model.max_seq_len=8",
        "model.memory_mlp_size=4",
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
    assert (run_dir / "checkpoints" / "step_0000001.pt").exists()


def test_lm_titans_experiment_forces_gate_open_and_frozen() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "vocab_size": 128,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "d_ffn": 32,
                "max_seq_len": 8,
                "dropout": 0.0,
                "memory_mlp_size": 4,
                "memory_layer": 0,
                "memory_decay_init": 0.99,
            },
            "experiment": {
                "use_memory": True,
                "gate_disabled": True,
            },
        }
    )

    model = build_model(cfg)
    gate = model.memory_modules[0].gate
    hidden = torch.randn(3, 16)
    surprise = torch.rand(3, 1)

    gate_value = gate(hidden, surprise, step=3)

    assert torch.allclose(gate_value, torch.ones_like(gate_value))
    assert all(not param.requires_grad for param in gate.parameters())


def test_eval_memory_model_config_supports_documented_checkpoint_override() -> None:
    cfg = _compose_config(
        "evaluate_memory",
        overrides=[
            "model=memory_lm_100m",
            "model.checkpoint_path=/tmp/best.pt",
            "evaluator=memory_diagnostic",
        ],
    )

    assert cfg.model.backend == "memory_transformer"
    assert cfg.model.checkpoint_path == "/tmp/best.pt"
    assert cfg.model.use_memory is True
    assert cfg.model.max_new_tokens > 0
    assert cfg.model.model_id == "memory_lm_100m"


def test_gate_activations_are_actual_write_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MemoryTransformer(_tiny_memory_cfg(), use_memory=True)
    memory_module = model.memory_modules[0]
    with torch.no_grad():
        memory_module.gate.linear.weight.zero_()
        memory_module.gate.linear.bias.zero_()
        memory_module.gate.linear.weight[0, model.cfg.d_model] = 0.5

    gate_calls: list[torch.Tensor] = []
    original_forward = memory_module.gate.forward

    def recording_forward(hidden: torch.Tensor, surprise: torch.Tensor, step: int) -> torch.Tensor:
        gate_value = original_forward(hidden, surprise, step)
        gate_calls.append(gate_value.detach().clone())
        return gate_value

    monkeypatch.setattr(memory_module.gate, "forward", recording_forward)

    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    model(input_ids)

    assert len(gate_calls) == input_ids.shape[1]
    actual_write_gates = torch.stack(gate_calls, dim=0)
    logged_gates = model.get_gate_activations()[0]

    assert torch.allclose(logged_gates, actual_write_gates, rtol=1e-6, atol=0.0)


def test_gate_auroc_runtime_dependencies_are_declared() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    dependency_names = {
        dependency.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0]
        for dependency in dependencies
    }

    assert "tiktoken" in dependency_names
    assert "scikit-learn" in dependency_names


def test_gate_auroc_uses_babilong_mutation_metadata_instead_of_hardcoded_phrase() -> None:
    source = (ROOT / "experiments" / "memory_state" / "gate_auroc.py").read_text(
        encoding="utf-8"
    )

    assert "mutation_phrases" in source
    assert '["moved to the"]' not in source
