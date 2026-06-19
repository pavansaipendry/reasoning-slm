"""Make plots from the training logs for the README / portfolio.

Parses artifacts/pretrain_base.log and artifacts/grpo_add.log and writes PNGs:
    artifacts/pretrain_loss.png   — pretrain train/val loss vs step
    artifacts/grpo_curve.png      — GRPO reward + accuracy vs step

    python -m scripts.plot_curves
"""

from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

_STEP = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)")
_VAL = re.compile(r"\[eval\]\s+step\s+(\d+)\s+val_loss\s+([\d.]+)")
_GRPO = re.compile(r"step\s+(\d+)\s+\|\s+reward\s+([\-\d.]+)\s+\|\s+acc\s+([\d.]+)")


def _read(path):
    with open(path) as fh:
        return fh.read()


def plot_pretrain():
    text = _read(os.path.join(ART, "pretrain_base.log"))
    steps = [(int(s), float(l)) for s, l in _STEP.findall(text)]
    vals = [(int(s), float(l)) for s, l in _VAL.findall(text)]
    if not steps:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*steps), label="train loss", alpha=0.7)
    if vals:
        plt.plot(*zip(*vals), "o-", label="val loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Pretrain (118M, gated attention)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ART, "pretrain_loss.png"), dpi=120)
    print("wrote artifacts/pretrain_loss.png")


def plot_grpo():
    text = _read(os.path.join(ART, "grpo_add.log"))
    rows = [(int(s), float(r), float(a)) for s, r, a in _GRPO.findall(text)]
    if not rows:
        return
    st, rw, ac = zip(*rows)
    plt.figure(figsize=(7, 4))
    plt.plot(st, ac, label="accuracy", color="tab:green")
    plt.plot(st, rw, label="reward", color="tab:blue", alpha=0.5)
    plt.xlabel("GRPO step"); plt.ylabel("value"); plt.title("GRPO RL: accuracy climbs 35% → 91%")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ART, "grpo_curve.png"), dpi=120)
    print("wrote artifacts/grpo_curve.png")


if __name__ == "__main__":
    plot_pretrain()
    plot_grpo()
