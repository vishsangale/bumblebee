from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, answer: str | None = None) -> ModelResponse: ...


def _resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


class OracleGenerator:
    def generate(self, prompt: str, *, answer: str | None = None) -> ModelResponse:
        del prompt
        assert answer is not None
        return ModelResponse(text=answer, input_tokens=0, output_tokens=0)


class HuggingFaceGenerator:
    def __init__(self, model_cfg) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_cfg = model_cfg
        tokenizer_kwargs = {"trust_remote_code": bool(model_cfg.trust_remote_code)}
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_cfg.model_id), **tokenizer_kwargs)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "dtype": _resolve_dtype(str(model_cfg.dtype)),
            "device_map": str(model_cfg.device_map),
            "trust_remote_code": bool(model_cfg.trust_remote_code),
            "low_cpu_mem_usage": True,
        }
        attn_impl = str(model_cfg.attn_implementation)
        if attn_impl not in {"", "none"}:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(str(model_cfg.model_id), **model_kwargs)
        self.device = next(self.model.parameters()).device

    def _format_prompt(self, prompt: str) -> str:
        if bool(self.model_cfg.use_chat_template) and hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            messages = [
                {"role": "system", "content": str(self.model_cfg.system_prompt)},
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self.model_cfg.system_prompt}\n\n{prompt}"

    def generate(self, prompt: str, *, answer: str | None = None) -> ModelResponse:
        del answer
        prompt_text = self._format_prompt(prompt)
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        model_inputs = {name: tensor.to(self.device) for name, tensor in model_inputs.items()}

        generate_kwargs = {
            "max_new_tokens": int(self.model_cfg.max_new_tokens),
            "do_sample": bool(self.model_cfg.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if bool(self.model_cfg.do_sample):
            generate_kwargs["temperature"] = float(self.model_cfg.temperature)
            generate_kwargs["top_p"] = float(self.model_cfg.top_p)

        generated = self.model.generate(**model_inputs, **generate_kwargs)
        prompt_len = int(model_inputs["input_ids"].shape[-1])
        output_ids = generated[0][prompt_len:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return ModelResponse(
            text=text,
            input_tokens=prompt_len,
            output_tokens=int(output_ids.shape[-1]),
        )


class MemoryTransformerGenerator:
    """Wraps a trained MemoryTransformer checkpoint for use with eval_memory.py."""

    def __init__(self, model_cfg) -> None:
        from memory_state.lm_backbone import MemoryTransformer, MemoryTransformerConfig

        self.enc = tiktoken.get_encoding("gpt2")
        cfg = MemoryTransformerConfig(
            vocab_size=int(model_cfg.vocab_size),
            d_model=int(model_cfg.d_model),
            n_heads=int(model_cfg.n_heads),
            n_layers=int(model_cfg.n_layers),
            d_ffn=int(model_cfg.d_ffn),
            max_seq_len=int(model_cfg.max_seq_len),
            dropout=0.0,
            memory_mlp_size=int(model_cfg.memory_mlp_size),
            memory_layer=int(getattr(model_cfg, "memory_layer", 0)),
            memory_decay_init=float(model_cfg.memory_decay_init),
        )
        use_memory = bool(model_cfg.use_memory)
        self.model = MemoryTransformer(cfg, use_memory=use_memory)
        checkpoint = torch.load(str(model_cfg.checkpoint_path), map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.max_new_tokens = int(model_cfg.max_new_tokens)

    def generate(self, prompt: str, *, answer: str | None = None) -> ModelResponse:
        del answer
        ids = self.enc.encode(prompt)
        input_ids = torch.tensor([ids])
        input_len = len(ids)

        if hasattr(self.model, "reset_memory"):
            self.model.reset_memory()

        generated: list[int] = []
        with torch.no_grad():
            context = input_ids
            for _ in range(self.max_new_tokens):
                logits = self.model(context)  # (1, T, V)
                next_token = int(logits[0, -1].argmax())
                generated.append(next_token)
                context = torch.cat([context, torch.tensor([[next_token]])], dim=1)
                if next_token == self.enc.eot_token:
                    break

        text = self.enc.decode(generated).strip()
        return ModelResponse(text=text, input_tokens=input_len, output_tokens=len(generated))


def load_text_generator(model_cfg) -> TextGenerator:
    backend = str(model_cfg.backend)
    if backend == "oracle":
        return OracleGenerator()
    if backend == "huggingface":
        return HuggingFaceGenerator(model_cfg)
    if backend == "memory_transformer":
        return MemoryTransformerGenerator(model_cfg)
    raise ValueError(f"Unsupported model backend: {backend}")
