"""Post-training datasets.

Two kinds:
  - SFT examples: (prompt, chain-of-thought, answer) -> formatted target text.
  - GRPO prompts: (prompt, gold_answer) -> the policy generates, rewards check the answer.

Sources:
  - GSM8K (grade-school math, real) via HuggingFace.
  - A synthetic arithmetic task, generated locally — no network, gives GRPO a clean
    verifiable signal to warm up on (R1-zero / TinyZero style) and lets the loops be
    smoke-tested offline.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .common import build_prompt, build_target


@dataclass
class SFTExample:
    prompt: str       # text fed to the model
    target: str       # text the model should produce (<think>..</think><answer>..</answer>)


@dataclass
class RLPrompt:
    prompt: str       # text fed to the model
    gold: str         # gold answer string, for the verifiable reward


# ---------------- synthetic arithmetic (offline, verifiable) ----------------
def _spaced(x) -> str:
    """Render an integer digit-by-digit ('110' -> '1 1 0') so each digit is its own
    token. Without this, BPE fuses multi-digit numbers into opaque tokens and a small
    model can't even copy the operands, let alone add them."""
    return " ".join(str(x))


def arithmetic_examples(n: int, seed: int = 0, max_val: int = 99, ops=("+",)):
    """Yield n arithmetic problems as (question, cot, target_answer, gold).

    Numbers are digit-spaced in the text the model reads/writes; `gold` is the plain
    integer string used by the verifiable reward (the reward strips the spaces back).
    Defaults to addition only with 2-digit operands: tractable for a small model after
    SFT, but not solved — so GRPO has a reward signal to climb."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(2, max_val), rng.randint(2, max_val)
        op = rng.choice(list(ops))
        ans = a + b if op == "+" else (a - b if op == "-" else a * b)
        sa, sb, sans = _spaced(a), _spaced(b), _spaced(ans)
        q = f"What is {sa} {op} {sb}?"
        cot = f"We compute {sa} {op} {sb} = {sans}."
        out.append((q, cot, sans, str(ans)))
    return out


def arithmetic_sft(n: int, seed: int = 0):
    return [SFTExample(build_prompt(q), build_target(cot, target))
            for q, cot, target, _ in arithmetic_examples(n, seed)]


def arithmetic_rl(n: int, seed: int = 0):
    return [RLPrompt(build_prompt(q), gold) for q, _, _, gold in arithmetic_examples(n, seed)]


# ---------------- GSM8K (real) ----------------
_GSM_ANS = re.compile(r"####\s*(-?[\d,]+)")


def _gsm_split(answer_field: str):
    """GSM8K answers look like: '<reasoning>\\n#### 42'. Split into (cot, final)."""
    m = _GSM_ANS.search(answer_field)
    final = m.group(1).replace(",", "") if m else ""
    cot = _GSM_ANS.sub("", answer_field).strip()
    return cot, final


def gsm8k_sft(n: int | None = None, split: str = "train"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
    out = []
    for i, row in enumerate(ds):
        if n and i >= n:
            break
        cot, final = _gsm_split(row["answer"])
        out.append(SFTExample(build_prompt(row["question"]), build_target(cot, final)))
    return out


def gsm8k_rl(n: int | None = None, split: str = "train"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
    out = []
    for i, row in enumerate(ds):
        if n and i >= n:
            break
        _, final = _gsm_split(row["answer"])
        out.append(RLPrompt(build_prompt(row["question"]), final))
    return out
