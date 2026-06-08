from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Config for the reasoning-slm decoder.

    `attention` selects the block:
      - "vanilla": built-in causal SDPA (defined in transformer.py)
      - "gated":   reuse ../gated-attention CausalSelfAttention(gate_attn=True)
      - "nsa":     reuse ../nsa-mini NSAttention (sparse, long-context)
    """

    vocab_size: int = 32000
    block_size: int = 2048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    attention: str = "gated"

    @property
    def n_params_estimate(self) -> int:
        # rough: embeddings + 12 * n_embd^2 per layer (attn+mlp) — for MFU/log only
        emb = self.vocab_size * self.n_embd
        per_layer = 12 * self.n_embd * self.n_embd
        return emb + self.n_layer * per_layer
