# reasoning-slm — resume material

Living doc of concrete, defensible points for this project. Updated as milestones land.
Repo: `github.com/pavansaipendry/reasoning-slm`

---

## One-line project description
Built an end-to-end pipeline that takes raw web text to a math/code **reasoning language
model** — data curation, a from-scratch tokenizer, pretraining (with custom gated
attention), RL post-training, and serving — on infrastructure (attention kernels +
inference engine) I wrote myself.

---

## Resume bullets (concise, quantified, defensible TODAY)

- Engineered an **end-to-end LLM training pipeline** (data → tokenizer → pretrain) for a
  math/code reasoning model, curating a **189M-token corpus** from a multi-source
  HuggingFace stream (open-web-math, code, FineWeb-Edu) with streaming ingestion, quality
  filtering, dedup, and packing behind a single reproducible CLI.
- Implemented **MinHash + LSH near-duplicate detection from scratch in NumPy** (no external
  dedup library) and **Gopher-style quality heuristics** for prose and code; on a 120k-doc
  run, filtered out 6.7% low-quality docs and removed 2.7k near-duplicates, with a full
  rejection-reason histogram.
- Trained a **118M-parameter decoder transformer from scratch** with a custom
  **gated-attention** block (NeurIPS 2025 repro: +2.12%, attention-sink-free), achieving
  **~33% MFU and ~146k tokens/sec on a single A100 80GB** with clean convergence.
- Built the **GPU training loop from scratch** — AdamW + warmup/cosine schedule, gradient
  accumulation to 262k tokens/step, bf16 autocast, **MFU & throughput instrumentation**,
  validation eval, and checkpointing.
- Automated **GPU infrastructure programmatically** via the RunPod REST API (provision/
  destroy A100 pods, SSH key injection, repo sync, remote launch) and debugged a real
  **CUDA driver/container-image incompatibility**.

## Post-training bullets — now defensible with real numbers
- Pretrained a 118M gated-attention model on an **841M-token** corpus to **validation
  loss 4.40** (from 10.5) at ~33% MFU on a single A100.
- **Implemented GRPO (Group Relative Policy Optimization) from scratch** — group-relative
  advantages, PPO-clipped objective, KL-to-reference, no value network — and used it with
  **verifiable rewards** to lift a reasoning task's accuracy from **35.7% (SFT) to 91.0%**
  (held-out, +55 points) with a stable, bounded-KL learning curve.
- Diagnosed and fixed a **BPE-tokenization failure mode** (multi-digit numbers fuse into
  opaque tokens, blocking arithmetic) via digit-level number representation, raising SFT
  accuracy 5% → 35.7% before RL.
- (Planned) Serve the final model on a from-scratch inference engine (PagedAttention +
  Triton) with an OpenAI-compatible API.

---

## Detailed technical points (for interviews / long-form)

**Data engineering**
- Streaming multi-source ingestion (no full downloads); resilient to transient HF
  rate-limit errors via reopen-and-skip retry logic.
- Quality filters: length, mean-word-length, symbol/alpha ratios, duplicate-line fraction
  (prose) and line-length / alphanumeric checks (code), each emitting a rejection reason.
- Dedup: deterministic BLAKE2b shingle hashing, universal-hash MinHash permutations mod a
  Mersenne prime, LSH banding with band/row counts auto-tuned to a target Jaccard.
- 32k-vocab byte-level BPE tokenizer trained from scratch; memmap `.bin` packing + manifest.

**Modeling & training**
- Decoder-only transformer, weight-tied embeddings, pluggable attention (`vanilla | gated |
  nsa`); gated block vendored in-repo so it stands alone.
- Training: fused AdamW, decay/no-decay param groups, warmup+cosine LR, grad accumulation,
  bf16 autocast, MFU computed against A100 bf16 peak, periodic val eval + checkpoints.

**Infra / MLOps**
- Programmatic A100 provisioning (RunPod REST API), SSH automation, remote background jobs,
  persistent-volume checkpointing, CUDA/image compatibility debugging.

**Testing**
- 6 offline unit tests (filters, MinHash dedup, Jaccard estimation, reward functions, full
  sample pipeline) — run with no GPU/network.

---

## Measured results
| Metric | Value |
|--------|-------|
| Corpus | 841M tokens, 431k docs (from 500k streamed) |
| Quality-filter reject rate | 8.2% (40,959 / 500,000) |
| Near-dups removed (MinHash) | 28,014 |
| Model | 118.2M params, gated attention, 2048 ctx |
| Throughput / MFU | ~145k tokens/sec, ~33% MFU (single A100 80GB) |
| Pretrain convergence | val loss 10.53 → **4.40** (3,200 steps) |
| SFT accuracy (2-digit addition, held-out) | **35.7%** (format 99%) |
| **GRPO accuracy (same task)** | **91.0%** (format 100%) — +55 pts over SFT |

---

## ATS keywords
LLM pretraining · PyTorch · BPE tokenization · MinHash/LSH dedup · data curation ·
gradient accumulation · mixed precision (bf16) · MFU · attention mechanisms · gated
attention · RLHF/GRPO · verifiable rewards · CUDA · A100 · RunPod · MLOps · inference
serving · PagedAttention · Triton
