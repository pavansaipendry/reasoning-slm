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
def arithmetic_examples(n: int, seed: int = 0, max_val: int = 50):
    """Yield n addition/multiplication problems with chain-of-thought + gold answer."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(2, max_val), rng.randint(2, max_val)
        op = rng.choice(["+", "*"])
        ans = a + b if op == "+" else a * b
        q = f"What is {a} {op} {b}?"
        cot = f"We compute {a} {op} {b} = {ans}."
        out.append((q, cot, str(ans)))
    return out


def arithmetic_sft(n: int, seed: int = 0):
    return [SFTExample(build_prompt(q), build_target(cot, ans))
            for q, cot, ans in arithmetic_examples(n, seed)]


def arithmetic_rl(n: int, seed: int = 0):
    return [RLPrompt(build_prompt(q), ans) for q, _, ans in arithmetic_examples(n, seed)]


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
