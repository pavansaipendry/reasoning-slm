# reasoning-slm

A math/code **reasoning small language model, trained end-to-end on a stack I built myself** —
from raw web text to a served reasoning API.

> Data pipeline → pretraining (with custom gated + NSA sparse attention) → SFT →
> GRPO RL post-training → rigorous eval → served on a from-scratch inference engine.

This repo is the capstone that ties together three earlier projects:

| Piece | Where it comes from |
|-------|--------------------|
| **Gated attention** block | [`../gated-attention`](../gated-attention) — NeurIPS 2025 repro (+2.12% / attention-sink-free) |
| **NSA sparse attention** block | [`../nsa-mini`](../nsa-mini) — Triton impl of DeepSeek Native Sparse Attention |
| **Serving engine** | [`../mini-vllm`](../mini-vllm) — PagedAttention + continuous batching + custom Triton kernel |

## The honest framing

On a **single A100 for a few hours** (~$25–40 total), a from-scratch ~100M model will *not*
post state-of-the-art benchmark numbers — and any small-model repo that claims it does is
leaking test data. So the artifact here is not a number. It is a **complete, reproducible,
fully-owned pipeline that demonstrably improves the model at every stage** — base → SFT → RL —
on a *verifiable* domain (math/code, where answers are checkable). The headline results are
**relative lift** and a **real GRPO learning curve**, reported honestly, including failures.

## The three acts

### Act I — Data + Pretrain  (`data/`, `model/`, `pretrain/`)
- Curate a math/code-heavy corpus (OpenWebMath + The Stack + a little FineWeb-Edu)
- Quality-filter (Gopher-style heuristics) → **MinHash near-dedup** → **train a 32k BPE tokenizer**
- Pretrain a ~100M decoder using the **gated + NSA** attention blocks; log val-loss, **MFU**, tokens/sec

### Act II — Post-train  (`posttrain/`)
- **SFT** on math/code chain-of-thought traces
- **GRPO implemented from scratch** with **verifiable rewards** (numeric-answer check, code execute + unit test, format reward)

### Act III — Eval + Serve  (`eval/`, `serve/`)
- `lm-eval-harness` (GSM8K / MMLU-subset) + a custom code-execution harness; report base vs SFT vs RL
- Serve on `mini-vllm` with an OpenAI-compatible endpoint + live tokens/sec demo

## Results (filled in as stages land)

| Stage | GSM8K (val) | Countdown reward | HumanEval-subset | Notes |
|-------|-------------|------------------|------------------|-------|
| Base (pretrain) | — | — | — | |
| + SFT | — | — | — | |
| + GRPO | — | — | — | |

## Quickstart

```bash
pip install -r requirements.txt

# Act I — build a tiny corpus offline (no network, no GPU) to verify the pipeline:
python -m data.pipeline --sample --out data/build_sample

# Act I — the real thing (streams from HuggingFace):
python -m data.pipeline --config configs/data.yaml --out data/build
```

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the stage-by-stage plan and current status.

## Status
- [x] Repo scaffold
- [ ] **Act I — data pipeline**  ← in progress
- [ ] Act I — pretrain
- [ ] Act II — SFT
- [ ] Act II — GRPO
- [ ] Act III — eval
- [ ] Act III — serve
