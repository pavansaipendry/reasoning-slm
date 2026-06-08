"""Shared helpers for SFT and GRPO: checkpoint/tokenizer loading, the prompt/answer
template, and token-level log-probabilities.

The reasoning format the model is trained to follow:
    <think> ...chain of thought... </think>
    <answer> final answer </answer>
This keeps the final answer machine-extractable so rewards are *verifiable*
(see posttrain/rewards.py).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.transformer import ReasoningLM


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----- prompt / target templates -----
_INSTR = ("Solve the problem. Put your reasoning between <think> and </think>, "
          "then the final answer between <answer> and </answer>.\n")


def build_prompt(question: str) -> str:
    return f"{_INSTR}Question: {question}\n"


def build_target(cot: str, answer: str) -> str:
    return f"<think>{cot}</think>\n<answer>{answer}</answer>"


# ----- loading -----
def load_model(ckpt_path: str, device: str | None = None):
    """Load a ReasoningLM from a checkpoint saved by pretrain/train.py or sft.py."""
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg_d = ckpt["config"]
    model = ReasoningLM(ModelConfig(**mcfg_d)).to(device)
    model.load_state_dict(ckpt["model"])
    return model, mcfg_d


def load_tokenizer_from(data_dir: str):
    """Load the tokenizer that produced a dataset (data_dir/tokenizer.json)."""
    from data.tokenizer import load_tokenizer
    path = os.path.join(data_dir, "tokenizer.json")
    return load_tokenizer(path)


def special_ids(tokenizer):
    eot = tokenizer.token_to_id("<|endoftext|>")
    pad = tokenizer.token_to_id("<|pad|>")
    return eot, (pad if pad is not None else eot)


# ----- token-level log-probs (used by GRPO) -----
def token_logprobs(model, input_ids):
    """Per-token log p(token_t | tokens_<t). Returns (B, T-1) aligned so that entry j
    is the log-prob of input_ids[:, j+1]."""
    logits, _ = model(input_ids)
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, tgt).squeeze(-1)
