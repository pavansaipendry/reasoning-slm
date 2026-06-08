# Roadmap

Budget assumption: **1× A100, a few hours total (~$25–40)**, cost-conscious.
Domain: **math/code reasoning** (verifiable answers → real RL rewards).

## Act I — Data + Pretrain
**Data pipeline** (`data/`) — *the long pole and the differentiator*
- [x] Document model + source streaming (`sources.py`) with an offline `--sample` mode
- [x] Gopher-style quality filters for prose + code (`filters.py`)
- [x] MinHash LSH near-dedup, plus an exact-hash fast path (`dedup.py`)
- [x] Train a 32k byte-level BPE tokenizer (`tokenizer.py`)
- [x] Tokenize + pack into memmap `.bin` shards with a manifest (`pack.py`)
- [x] One CLI to run it all + stats reporting (`pipeline.py`)
- [ ] Run the real mixture on RunPod, inspect stats, freeze a corpus

**Pretrain** (`model/`, `pretrain/`)
- [ ] ~100M decoder; attention backend switch: `vanilla | gated | nsa`
- [ ] Wire in `../gated-attention` and `../nsa-mini` blocks
- [ ] Training loop with MFU + tokens/sec logging; target a val-loss curve
- [ ] Mini scaling sweep (2–3 widths) for the README

## Act II — Post-train
- [ ] SFT on math/code CoT (GSM8K train, MetaMathQA, OpenMathInstruct)
- [ ] **GRPO from scratch** (`posttrain/grpo.py`) — group-relative advantages, KL to ref
- [ ] Verifiable rewards (`posttrain/rewards.py`): numeric check, code exec + unit test, format
- [ ] Parallel track: GRPO on Qwen2.5-0.5B for a demo that pops

## Act III — Eval + Serve
- [ ] `lm-eval-harness` configs (GSM8K, MMLU-subset) + custom code-exec eval
- [ ] base vs SFT vs RL table, with failure cases
- [ ] Serve on `../mini-vllm`, OpenAI-compatible endpoint, tokens/sec demo

## Compute napkin math
- Pretrain: 100M params × ~3B tokens × 6 FLOPs ≈ 1.8e18 FLOPs.
  A100 @ ~50% MFU (≈150 TFLOP/s effective) → ~3.3 h → ~$6.
- SFT: <1 h. GRPO: 2–4 h (rollouts dominate). Total ≈ $25–40.
