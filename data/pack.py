"""Tokenize the cleaned corpus and pack it into flat memmap-able .bin files.

Layout matches the nanoGPT convention: a single contiguous array of token ids per
split (train/val), each document terminated by the end-of-text token, written as raw
little-endian integers so training can `np.memmap` them with zero parsing. A
manifest.json records token counts and dtype so the loader needs no guesswork.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

import numpy as np

from .sources import Document
from .tokenizer import token_id


def _dtype(name: str):
    return {"uint16": np.uint16, "uint32": np.uint32}[name]


class _ShardWriter:
    """Buffers token ids and flushes to <split>.bin; tracks total tokens written."""

    def __init__(self, path: str, dtype) -> None:
        self.path = path
        self.dtype = dtype
        self._fh = open(path, "wb")
        self._buf: list[int] = []
        self.tokens = 0

    def add(self, ids: list[int]) -> None:
        self._buf.extend(ids)
        self.tokens += len(ids)
        if len(self._buf) >= 1_000_000:
            self._flush()

    def _flush(self) -> None:
        if self._buf:
            np.asarray(self._buf, dtype=self.dtype).tofile(self._fh)
            self._buf.clear()

    def close(self) -> None:
        self._flush()
        self._fh.close()


def pack_documents(
    docs: Iterable[Document],
    tokenizer,
    out_dir: str,
    dtype: str = "uint16",
    eot_token: str = "<|endoftext|>",
    val_fraction: float = 0.0005,
    seed: int = 0,
) -> dict:
    """Encode every document, append <eot>, and split into train/val .bin files."""
    os.makedirs(out_dir, exist_ok=True)
    eot_id = token_id(tokenizer, eot_token)
    np_dtype = _dtype(dtype)
    max_id = np.iinfo(np_dtype).max

    rng = np.random.default_rng(seed)
    train = _ShardWriter(os.path.join(out_dir, "train.bin"), np_dtype)
    val = _ShardWriter(os.path.join(out_dir, "val.bin"), np_dtype)

    n_docs = 0
    for doc in docs:
        ids = tokenizer.encode(doc.text).ids
        if not ids:
            continue
        if max(ids) > max_id or eot_id > max_id:
            raise ValueError(f"token id exceeds {dtype}; use a larger dtype")
        ids.append(eot_id)
        (val if rng.random() < val_fraction else train).add(ids)
        n_docs += 1

    train.close()
    val.close()

    manifest = {
        "dtype": dtype,
        "eot_id": eot_id,
        "vocab_size": tokenizer.get_vocab_size(),
        "n_docs": n_docs,
        "train_tokens": train.tokens,
        "val_tokens": val.tokens,
        "files": {"train": "train.bin", "val": "val.bin"},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_split(out_dir: str, split: str = "train") -> np.ndarray:
    """Memmap a packed split back as a 1-D array of token ids (for the trainer)."""
    with open(os.path.join(out_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    path = os.path.join(out_dir, manifest["files"][split])
    return np.memmap(path, dtype=_dtype(manifest["dtype"]), mode="r")
