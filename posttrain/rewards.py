"""Verifiable rewards for GRPO (Act II).

These are *real* and unit-testable now even though the GRPO loop is still a stub —
the whole reason math/code was chosen is that correctness is checkable, so the
reward signal is honest rather than a learned reward model. Each reward maps a
model completion + gold answer to a float.
"""

from __future__ import annotations

import re

# The format we train the model to follow:  <think> ... </think> <answer> X </answer>
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Pull the content of the last <answer>...</answer> block, if present."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def _to_number(s: str):
    if s is None:
        return None
    # strip thousands separators AND inter-digit spaces ("1 1 0" -> "110" for the
    # digit-tokenized arithmetic representation)
    m = _NUM_RE.search(s.replace(",", "").replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def format_reward(completion: str) -> float:
    """Small shaping reward for emitting the think/answer scaffold correctly."""
    has_think = bool(_THINK_RE.search(completion))
    has_answer = bool(_ANSWER_RE.search(completion))
    return 0.5 * has_think + 0.5 * has_answer


def numeric_reward(completion: str, gold: str, tol: float = 1e-6) -> float:
    """1.0 if the model's final numeric answer matches gold within tol, else 0.0."""
    pred = _to_number(extract_answer(completion))
    gold_num = _to_number(gold)
    if pred is None or gold_num is None:
        return 0.0
    return 1.0 if abs(pred - gold_num) <= tol else 0.0


def total_reward(completion: str, gold: str, w_format: float = 0.1) -> float:
    """Correctness dominates; format is a small nudge. Used by GRPO."""
    return numeric_reward(completion, gold) + w_format * format_reward(completion)


# code-execution reward (execute + unit test) lives in eval/code_exec.py so the
# sandbox logic is shared between training rewards and evaluation.
