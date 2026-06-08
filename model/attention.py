"""Attention backend selector.

The whole point of this project is that the *architecture* reuses my own research:
the gated-attention block (NeurIPS 2025 repro) and the NSA sparse-attention block.
Rather than copy that code, we import it from the sibling repos so there is a single
source of truth. Run pretraining from a checkout where these siblings exist, or set
REASONING_SLM_SIBLINGS to a directory that contains `gated-attention/` and `nsa-mini/`.

This module is a thin adapter; the vanilla fallback lives in transformer.py so the
model always builds even without the siblings present (handy for CI / unit tests).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_siblings_to_path() -> None:
    root = os.environ.get("REASONING_SLM_SIBLINGS")
    candidates = []
    if root:
        candidates.append(Path(root))
    # default: assume this repo sits next to gated-attention/ and nsa-mini/
    candidates.append(Path(__file__).resolve().parents[2])
    for base in candidates:
        for name in ("gated-attention", "nsa-mini"):
            p = base / name
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))


def build_attention(config):
    """Return an nn.Module mapping (B, T, n_embd) -> (B, T, n_embd)."""
    backend = config.attention
    if backend == "vanilla":
        from .transformer import VanillaCausalSelfAttention
        return VanillaCausalSelfAttention(config)

    _add_siblings_to_path()

    if backend == "gated":
        # ../gated-attention/model.py
        from model import GPTConfig, CausalSelfAttention  # type: ignore
        gcfg = GPTConfig(
            block_size=config.block_size,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            gate_attn=True,
        )
        return CausalSelfAttention(gcfg)

    if backend == "nsa":
        # ../nsa-mini/nsa/module.py
        from nsa.reference import NSAConfig  # type: ignore
        from nsa.module import NSAttention  # type: ignore
        ncfg = NSAConfig(
            dim=config.n_embd,
            n_heads=config.n_head,
            n_kv_heads=max(1, config.n_head // 4),
            head_dim=config.n_embd // config.n_head,
            bias=config.bias,
        )
        return NSAttention(ncfg)

    raise ValueError(f"unknown attention backend: {backend!r}")
