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

## Bullets that become defensible AFTER the runs in flight finish
- Pretrained the model on a **~1B-token corpus** to a base val-loss of `<X>` (run in progress).
- Post-trained with **SFT + GRPO (implemented from scratch) using verifiable rewards**
  (numeric-answer and code-execution checks), improving GSM8K/HumanEval-subset from
  `<base>` → `<final>` (planned).
- Served the model on a **from-scratch inference engine** (PagedAttention + custom Triton
  kernel) with an OpenAI-compatible API at `<N>` tokens/sec (planned).

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

## Measured results so far
| Metric | Value |
|--------|-------|
| Corpus (validation build) | 189.4M tokens, 109k docs (from 120k streamed) |
| Quality-filter reject rate | 6.7% (8,030 / 120,000) |
| Near-dups removed (MinHash) | 2,683 |
| Model | 118.2M params, gated attention, 2048 ctx |
| Throughput | ~146k tokens/sec, single A100 80GB |
| MFU | ~33% |
| Convergence (400-step val run) | val loss 10.53 → 7.36 |

---

## ATS keywords
LLM pretraining · PyTorch · BPE tokenization · MinHash/LSH dedup · data curation ·
gradient accumulation · mixed precision (bf16) · MFU · attention mechanisms · gated
attention · RLHF/GRPO · verifiable rewards · CUDA · A100 · RunPod · MLOps · inference
serving · PagedAttention · Triton
