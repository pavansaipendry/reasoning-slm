# reasoning-slm

A **math-reasoning small language model, trained end-to-end on a stack I built myself** —
from raw web text, through pretraining with custom gated attention, to RL post-training
that teaches the model to reason.

> Data pipeline → pretraining (custom **gated attention**) → SFT → **GRPO RL post-training
> (implemented from scratch)** → verifiable-reward eval.

The headline: starting from a model pretrained from scratch, **GRPO RL lifts accuracy on a
verifiable arithmetic-reasoning task from 35.7% (after SFT) to 91.0%** — the DeepSeek-R1-zero
/ TinyZero "RL learns to reason" curve, reproduced on a model *and* a GRPO implementation
written from scratch, on a single A100.

## Results

118M-param model, addition reasoning task, **300 held-out problems, greedy decoding**:

| Stage | Accuracy | Format rate |
|-------|----------|-------------|
| Base (pretrained only) | 0.0% | 0% |
| + SFT | 35.7% | 99% |
| **+ GRPO** | **91.0%** | 100% |

Other measured numbers:

| Metric | Value |
|--------|-------|
| Corpus | 841M tokens, 431k docs (from 500k streamed, filtered + MinHash-deduped) |
| Pretrain | 118M params, gated attention, val loss 10.5 → **4.40** (3,200 steps) |
| Throughput | ~145k tokens/sec, ~33% MFU on a single A100 80GB |

GRPO training curve (reward → ~1.0, KL stays bounded) is in [`artifacts/grpo_add.log`](artifacts/);
the pretrain curve is in [`artifacts/pretrain_base.log`](artifacts/).

### Attention ablation

Same config (118M, 800 steps, identical data) — swapping *only* the attention block:

| Backend | Val loss | Notes |
|---------|----------|-------|
| vanilla | 6.0037 | baseline |
| **gated** | **5.9477** | best — the gated-attention benefit, reproduced in-repo |
| NSA | n/a | ~85× slower at 2k context (sparse-attention overhead only pays off at long context) — see below |

### Serving

`serve/serve.py` loads a checkpoint and serves an OpenAI-compatible `/v1/chat/completions`
endpoint (and a `/generate` route) — measured **~60–214 tokens/sec** on a single A100.

## The honest framing

This is a **118M-param model trained for ~1 epoch on ~841M tokens** — it is a demonstration
of the *full method and infrastructure*, not a generally capable model. The point is a
**complete, reproducible, fully-owned pipeline that measurably improves the model at every
stage** (base → SFT → RL) on a domain where answers are **verifiable**, so the reward signal
is real rather than a learned reward model. Numbers are reported on held-out problems.

Two real findings from the work:
- **BPE blocks arithmetic.** With standard BPE, multi-digit numbers fuse into opaque tokens
  and the model can't even copy the operands — so numbers are rendered **digit-by-digit**,
  which lifts SFT accuracy from ~5% to 35.7% before RL.
- **GRPO needs a non-sparse base (the RL cold-start problem).** On a harder 3-digit
  addition/subtraction task, SFT only reached ~8% (carries/borrows exceed a 118M model's
  capacity), so almost no rollout earned a reward and GRPO **could not bootstrap** (stayed
  ~5%). GRPO works only when the SFT base already gets enough answers right to create reward
  variance — a concrete instance of why DeepSeek-R1 needed an SFT "cold-start" stage.

## What's built vs. planned

| Stage | Status |
|-------|--------|
| Data pipeline (filter, MinHash dedup, BPE, packing) | ✅ done |
| Pretraining (gated attention, MFU/throughput logging) | ✅ done |
| SFT (prompt-masked CoT fine-tuning) | ✅ done |
| GRPO from scratch (group-relative adv, PPO-clip, KL-to-ref) | ✅ done |
| Verifiable rewards (numeric + format) | ✅ done |
| Eval harness (accuracy + format on a verifiable task) | ✅ done |
| Attention ablation (vanilla / gated / NSA) | ✅ done — gated wins (5.95 < 6.00) |
| Serving (OpenAI-compatible inference API) | ✅ done (`serve/serve.py`) |
| NSA sparse-attention backend | ✅ runs, but impractical at 2k context (long-context only) |
| Code-execution reward (`eval/code_exec.py`) | ✅ implemented; ⏳ not yet used in a training run |
| Real benchmarks (GSM8K / MMLU / HumanEval) | ⏳ harness ready; needs a larger model |
| High-throughput serving on `mini-vllm` | ⏳ planned (the demo server above is the current path) |

## Architecture

- Decoder-only transformer, weight-tied embeddings, **selectable attention backend**
  (`vanilla | gated | nsa`). The **gated-attention** block (a query-dependent sigmoid gate on
  the attention output; NeurIPS 2025 "Gated Attention" repro) is **vendored in-repo** so this
  project stands alone. The `nsa` backend optionally imports my
  [nsa-mini](https://github.com/pavansaipendry) NSA implementation if present.
- Training loop from scratch: AdamW, warmup + cosine schedule, gradient accumulation to
  262k tokens/step, bf16 autocast, MFU + tokens/sec logging, val eval + checkpointing.
- **GRPO from scratch**: for each prompt, sample a group of completions, score with a
  verifiable reward, use the group-normalized reward as the advantage (no value network),
  update with a PPO-clipped objective + KL penalty to a frozen reference policy.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Data — verify the whole pipeline offline (no network/GPU):
python -m data.pipeline --sample --out data/build_sample
# ...or build the real corpus (streams from HuggingFace):
python -m data.pipeline --config configs/data.yaml --out data/build

# 2. Pretrain (GPU):
python -m pretrain.train --config configs/pretrain_100m.yaml --data-dir data/build \
    --out checkpoints/base --max-steps 3200

# 3. SFT on the verifiable reasoning task:
python -m posttrain.sft --ckpt checkpoints/base/ckpt.pt --data-dir data/build \
    --dataset arithmetic --out checkpoints/sft

# 4. GRPO RL from the SFT checkpoint:
python -m posttrain.grpo --ckpt checkpoints/sft/ckpt.pt --data-dir data/build \
    --dataset arithmetic --out checkpoints/grpo

# 5. Eval base vs SFT vs GRPO:
python -m eval.eval_reasoning --ckpt checkpoints/grpo/ckpt.pt --data-dir data/build \
    --dataset arithmetic --n 300 --greedy
```

Tests (offline, no GPU): `python tests/test_pipeline.py`

## Tests
6 offline tests cover the quality filters, MinHash/LSH dedup, Jaccard estimation, the
verifiable reward functions, and a full end-to-end run on the sample corpus.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the stage-by-stage plan and [`RESUME.md`](RESUME.md)
for a summary of results.
