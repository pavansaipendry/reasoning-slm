"""Reasoning eval: generate answers for a verifiable task and score them.

Reports accuracy (final answer correct) and format rate (emits <think>/<answer>).
Used to compare base vs SFT vs GRPO on the same prompts.

Note: this model uses learned absolute position embeddings, so left-padding a batch
would shift positions and corrupt generation. We therefore generate per prompt
(batch=1) — simple and correct; fast enough for a few hundred eval prompts.

Example:
    python -m eval.eval_reasoning --ckpt checkpoints/sft_arith/ckpt.pt \
        --data-dir data/build_1b --dataset arithmetic --n 200
"""

from __future__ import annotations

import argparse

import torch

from posttrain.common import get_device, load_model, load_tokenizer_from, special_ids
from posttrain import data as ptdata
from posttrain.rewards import extract_answer, numeric_reward, format_reward


def build_prompts(name, n, split):
    if name in ptdata.ARITHMETIC_TASKS:
        return ptdata.arithmetic_rl(n, seed=12345, task=name)   # different seed than training
    if name == "gsm8k":
        return ptdata.gsm8k_rl(n, split=split)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="arithmetic", choices=["arithmetic", "arithmetic_hard", "gsm8k"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--greedy", action="store_true", help="argmax decoding (else temp=0.7)")
    ap.add_argument("--show", type=int, default=3, help="print this many sample generations")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model, _ = load_model(args.ckpt, device)
    model.eval()
    tok = load_tokenizer_from(args.data_dir)
    eot_id, _ = special_ids(tok)
    prompts = build_prompts(args.dataset, args.n, args.split)

    temp = 1e-4 if args.greedy else 0.7
    top_k = 1 if args.greedy else 50

    n_correct = n_format = 0
    for i, rp in enumerate(prompts):
        ids = tok.encode(rp.prompt).ids
        x = torch.tensor([ids], device=device)
        out = model.generate(x, args.max_new_tokens, temperature=temp, top_k=top_k, eot_id=eot_id)
        comp = tok.decode(out[0].tolist()[len(ids):])
        c = numeric_reward(comp, rp.gold)
        n_correct += c
        n_format += 1 if format_reward(comp) >= 1.0 else 0
        if i < args.show:
            print(f"--- sample {i} ---\nPROMPT: {rp.prompt.strip()[-80:]}\n"
                  f"GOLD: {rp.gold} | PRED: {extract_answer(comp)} | correct={int(c)}\n"
                  f"GEN: {comp.strip()[:200]}\n", flush=True)

    n = len(prompts)
    print(f"\n=== {args.dataset} (n={n}) | ckpt={args.ckpt} ===")
    print(f"accuracy   : {n_correct/n*100:.1f}%  ({n_correct}/{n})")
    print(f"format rate: {n_format/n*100:.1f}%")


if __name__ == "__main__":
    main()
