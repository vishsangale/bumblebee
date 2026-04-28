"""
Text data loading for memory LM training.

Smoke mode: generates random token IDs for fast iteration.
Real mode: streams from a pre-tokenized binary file (see below for prep).

To prepare real training data (run once, requires ~2GB disk):
  pip install datasets tiktoken
  python experiments/memory_state/data.py --prepare
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
from torch import Tensor


def synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> Tensor:
    """Random token IDs for smoke tests — no real text needed."""
    return torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)


class TokenDataset:
    """Streams fixed-length chunks from a flat binary token file (uint16)."""

    def __init__(self, path: str | Path, seq_len: int) -> None:
        import numpy as np
        self.data = np.memmap(str(path), dtype="uint16", mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def get_batch(self, start: int, batch_size: int, device: torch.device) -> Tensor:
        import numpy as np
        chunks = []
        for i in range(batch_size):
            idx = (start + i * self.seq_len) % len(self)
            chunk = torch.from_numpy(self.data[idx : idx + self.seq_len + 1].astype("int64"))
            chunks.append(chunk)
        return torch.stack(chunks).to(device)


def prepare_fineweb(output_path: str | Path, num_tokens: int = 100_000_000) -> None:
    """Download and tokenize FineWeb-edu to a flat uint16 binary file."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    written = 0
    with out.open("wb") as f:
        for example in ds:
            ids = enc.encode_ordinary(example["text"])
            ids.append(enc.eot_token)
            f.write(struct.pack(f"{len(ids)}H", *ids))
            written += len(ids)
            if written >= num_tokens:
                break
    print(f"Wrote {written:,} tokens to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", default="data/fineweb_train.bin")
    parser.add_argument("--num_tokens", type=int, default=100_000_000)
    args = parser.parse_args()
    if args.prepare:
        prepare_fineweb(args.output, args.num_tokens)
