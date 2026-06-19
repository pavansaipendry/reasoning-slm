"""Supervised fine-tuning on math/code chain-of-thought (Act II).

Causal-LM SFT with prompt masking: the loss is computed only over the target
(<think>/<answer>) tokens, not the prompt. Consumes a base checkpoint from
pretrain/train.py and the tokenizer that built the data.

Example:
    python -m posttrain.sft --ckpt checkpoints/base/ckpt.pt --data-dir data/build_1b \
        --dataset gsm8k --n 8000 --epochs 2 --out checkpoints/sft
"""

from __future__ import annotations

import argparse
import os
import random

import torch

from .common import get_device, load_model, load_tokenizer_from, special_ids
from . import data as ptdata


def encode_example(ex, tok, eot_id, block_size):
    """Return (input_ids, labels) with prompt positions masked to -1, or None if it
    doesn't fit / is degenerate."""
    p = tok.encode(ex.prompt).ids
    t = tok.encode(ex.target).ids + [eot_id]
    full = (p + t)[:block_size]
    if len(full) < 2 or len(p) >= len(full):
        return None
    x = full[:-1]
    y = full[1:]
    # mask everything that predicts a prompt token (positions 0 .. len(p)-2)
    for j in range(min(len(p) - 1, len(y))):
        y[j] = -1
    return x, y


def collate(batch, pad_id):
    maxlen = max(len(x) for x, _ in batch)
    X, Y = [], []
    for x, y in batch:
        X.append(x + [pad_id] * (maxlen - len(x)))
        Y.append(y + [-1] * (maxlen - len(y)))
    return torch.tensor(X), torch.tensor(Y)


def build_dataset(name, n):
    if name in ptdata.ARITHMETIC_TASKS:
        return ptdata.arithmetic_sft(n, task=name)
    if name == "gsm8k":
        return ptdata.gsm8k_sft(n)
    raise ValueError(f"unknown dataset {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="base (pretrained) checkpoint")
    ap.add_argument("--data-dir", required=True, help="dir with tokenizer.json")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "arithmetic", "arithmetic_hard"])
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    os.makedirs(args.out, exist_ok=True)
    model, mcfg_d = load_model(args.ckpt, device)
    tok = load_tokenizer_from(args.data_dir)
    eot_id, pad_id = special_ids(tok)
    block = mcfg_d["block_size"]

    raw = build_dataset(args.dataset, args.n)
    examples = [e for e in (encode_example(r, tok, eot_id, block) for r in raw) if e]
    print(f"SFT: {len(examples)} usable examples (of {len(raw)}) | dataset={args.dataset}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    rng = random.Random(0)
    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        for i in range(0, len(examples) - args.batch, args.batch):
            x, y = collate(examples[i : i + args.batch], pad_id)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    torch.save({"model": model.state_dict(), "config": mcfg_d, "step": step},
               os.path.join(args.out, "ckpt.pt"))
    print(f"done. SFT checkpoint -> {os.path.join(args.out, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
