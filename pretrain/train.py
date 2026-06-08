"""Pretraining entrypoint (Act I).

A compact but real training loop: memmap .bin data, AdamW + warmup/cosine schedule,
gradient accumulation to a target tokens-per-step, bf16 autocast on CUDA, periodic
val-loss eval, and MFU + tokens/sec logging. Consumes the dataset built by
`python -m data.pipeline` (data/build/{train,val}.bin + manifest.json).

Example:
    python -m pretrain.train --config configs/pretrain_100m.yaml --data-dir data/build \
        --out checkpoints/run1 --max-steps 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import yaml

from model.config import ModelConfig
from model.transformer import ReasoningLM


def get_batch(data: np.ndarray, block_size: int, batch: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def lr_at(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


def configure_optimizer(model, lr, weight_decay, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch, device, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch, device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_100m.yaml")
    ap.add_argument("--data-dir", default=None, help="override data dir (else from config)")
    ap.add_argument("--out", default="checkpoints/run1")
    ap.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    ap.add_argument("--attention", default=None, help="override: vanilla|gated|nsa")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    mcfg_d, ocfg, lcfg = cfg["model"], cfg["optim"], cfg["log"]
    data_dir = args.data_dir or cfg["data"]["dir"]
    os.makedirs(args.out, exist_ok=True)

    # vocab_size MUST match the tokenizer that produced the data -> read the manifest
    with open(os.path.join(data_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    mcfg_d["vocab_size"] = manifest["vocab_size"]
    if args.attention:
        mcfg_d["attention"] = args.attention
    mcfg = ModelConfig(**mcfg_d)

    dtype = np.uint16 if manifest["dtype"] == "uint16" else np.uint32
    train = np.memmap(os.path.join(data_dir, manifest["files"]["train"]), dtype=dtype, mode="r")
    val_path = os.path.join(data_dir, manifest["files"]["val"])
    val = np.memmap(val_path, dtype=dtype, mode="r") if os.path.getsize(val_path) else train

    device = args.device
    torch.manual_seed(1337)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = ReasoningLM(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = configure_optimizer(model, ocfg["lr"], ocfg["weight_decay"])

    block = mcfg.block_size
    micro = ocfg["micro_batch"]
    accum = max(1, ocfg["batch_tokens"] // (micro * block))
    max_steps = args.max_steps or ocfg["max_steps"]
    warmup = ocfg["warmup_steps"]
    tokens_per_step = micro * block * accum

    # A100 bf16 peak ~312 TFLOP/s; MFU = achieved / peak. flops/token ~ 6N (Kaplan).
    peak_flops = 312e12
    flops_per_step = 6 * n_params * tokens_per_step

    print(f"params={n_params/1e6:.1f}M  attention={mcfg.attention}  block={block}  "
          f"micro={micro} accum={accum} -> {tokens_per_step:,} tok/step  steps={max_steps}")
    print(f"train_tokens={len(train):,}  val_tokens={len(val):,}  device={device}")

    model.train()
    t0 = time.time()
    for step in range(max_steps):
        lr = lr_at(step, warmup, max_steps, ocfg["lr"], ocfg["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(accum):
            x, y = get_batch(train, block, micro, device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / accum).backward()
            loss_accum += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
        opt.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
        t0 = time.time()
        tok_s = tokens_per_step / dt
        mfu = (flops_per_step / dt) / peak_flops

        if step % 10 == 0 or step == max_steps - 1:
            print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"{dt*1000:6.0f}ms | {tok_s/1e3:6.1f}k tok/s | mfu {mfu*100:4.1f}%", flush=True)

        if lcfg.get("eval_interval") and (step % lcfg["eval_interval"] == 0 or step == max_steps - 1):
            vloss = estimate_loss(model, val, block, micro, device)
            print(f"  [eval] step {step} val_loss {vloss:.4f}", flush=True)
            ckpt = {"model": model.state_dict(), "config": mcfg_d, "step": step, "val_loss": vloss}
            torch.save(ckpt, os.path.join(args.out, "ckpt.pt"))

    print(f"done. final checkpoint -> {os.path.join(args.out, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
