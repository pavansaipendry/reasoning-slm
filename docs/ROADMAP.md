# Roadmap

Domain: **math reasoning** on a *verifiable* task (answers are checkable → real RL rewards).
Hardware: single A100 80GB (RunPod).

## Act I — Data + Pretrain ✅ DONE
**Data pipeline** (`data/`)
- [x] Source streaming (`sources.py`) with an offline `--sample` mode
- [x] Gopher-style quality filters for prose + code (`filters.py`)
- [x] MinHash + LSH near-dedup, hand-rolled in NumPy (`dedup.py`)
- [x] 32k byte-level BPE tokenizer (`tokenizer.py`)
- [x] Tokenize + pack into memmap `.bin` shards with a manifest (`pack.py`)
- [x] One CLI + JSON stats report (`pipeline.py`)
- [x] Built real corpus: **841M tokens, 431k docs** (open-web-math + codeparrot-clean + fineweb-edu)

**Pretrain** (`model/`, `pretrain/`)
- [x] 118M decoder; attention backend switch (`vanilla | gated | nsa`)
- [x] Gated-attention block **vendored in-repo**
- [x] Training loop with MFU + tokens/sec logging
- [x] Trained base: val loss 10.5 → **4.40**, ~33% MFU

## Act II — Post-train ✅ DONE (math/arithmetic)
- [x] SFT (prompt-masked CoT) — `posttrain/sft.py`
- [x] **GRPO from scratch** (group-relative adv, PPO-clip, KL-to-ref) — `posttrain/grpo.py`
- [x] Verifiable rewards: numeric + format — `posttrain/rewards.py`
- [x] Result: SFT 35.7% → **GRPO 91.0%** accuracy (held-out, +55 pts)
- [ ] Code-execution reward (`eval/code_exec.py`) — implemented, not yet used in a run

## Act III — Eval + Serve
- [x] Reasoning eval harness (accuracy + format) — `eval/eval_reasoning.py`
- [x] base vs SFT vs GRPO table
- [ ] Serve on `../mini-vllm` / simple inference API + demo

## "Best results" — remaining work
**No GPU (polish):**
- [x] Accurate README + ROADMAP
- [ ] `eval/code_exec.py` sandboxed scorer
- [ ] `scripts/chat.py` interactive demo
- [ ] LICENSE, CI (run tests), results plot from logs

**GPU session (one start/stop, ~$10–15):**
- [ ] Harder reasoning task: mixed ops + larger numbers; attempt **Countdown** (TinyZero-style)
- [ ] Bigger/longer base for a more capable model
- [ ] Train an **NSA** run; **gated vs vanilla vs NSA** ablation (val loss)
- [ ] Serving demo (API + tokens/sec)
- [ ] Final eval table on the upgraded task

## Compute notes (measured)
- Pretrain 118M × 841M tokens (3,200 steps) ≈ 1.6 h, ~$2, ~33% MFU, ~145k tok/s.
- SFT: minutes. GRPO: ~25–35 min for 200 steps (rollouts dominate).
