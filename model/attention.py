"""Attention backend selector.

The architecture reuses my own attention research:
  - "gated": a query-dependent sigmoid gate on the attention output (NeurIPS 2025
    "Gated Attention" repro). Vendored directly into transformer.py so this repo is
    self-contained and runs anywhere — no sibling-repo dependency.
  - "nsa":   DeepSeek Native Sparse Attention (sparse, long-context). This one is
    Triton-kernel-heavy, so rather than vendor it we optionally import it from the
    nsa-mini project if present (next to this repo, or via REASONING_SLM_SIBLINGS).
  - "vanilla": plain causal SDPA, the always-available fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_siblings_to_path() -> None:
    root = os.environ.get("REASONING_SLM_SIBLINGS")
    bases = [Path(root)] if root else []
    bases.append(Path(__file__).resolve().parents[2])  # repo's parent dir
    for base in bases:
        p = base / "nsa-mini"
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def build_attention(config):
    """Return an nn.Module mapping (B, T, n_embd) -> (B, T, n_embd)."""
    backend = config.attention

    if backend == "vanilla":
        from .transformer import VanillaCausalSelfAttention
        return VanillaCausalSelfAttention(config)

    if backend == "gated":
        from .transformer import GatedCausalSelfAttention
        return GatedCausalSelfAttention(config)

    if backend == "nsa":
        _add_siblings_to_path()
        try:
            from nsa.reference import NSAConfig          # type: ignore
            from nsa.module import NSAttention            # type: ignore
        except ImportError as e:
            raise ImportError(
                "attention='nsa' needs the nsa-mini project importable "
                "(place it next to this repo or set REASONING_SLM_SIBLINGS). "
                "Use attention='gated' or 'vanilla' otherwise."
            ) from e
        # NSAConfig derives head_dim = dim // n_heads itself (it's a property, not an arg).
        ncfg = NSAConfig(
            dim=config.n_embd,
            n_heads=config.n_head,
            n_kv_heads=max(1, config.n_head // 4),
            bias=config.bias,
        )
        return NSAttention(ncfg)

    raise ValueError(f"unknown attention backend: {backend!r}")
