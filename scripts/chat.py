"""Interactive demo: ask the trained model a question and watch it reason.

    python -m scripts.chat --ckpt checkpoints/grpo/ckpt.pt --data-dir data/build_1b
    python -m scripts.chat --ckpt checkpoints/grpo/ckpt.pt --data-dir data/build_1b \
        --question "What is 4 7 + 3 8?"     # numbers are digit-spaced (see README)

Single --question prints one answer; otherwise it's an interactive loop.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from posttrain.common import build_prompt, get_device, load_model, load_tokenizer_from, special_ids


def answer(model, tok, eot_id, device, question, max_new_tokens, temperature):
    prompt = build_prompt(question)
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device=device)
    top_k = 1 if temperature <= 0 else 50
    out = model.generate(x, max_new_tokens, temperature=max(temperature, 1e-4),
                         top_k=top_k, eot_id=eot_id)
    return tok.decode(out[0].tolist()[len(ids):]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--question", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model, _ = load_model(args.ckpt, device)
    model.eval()
    tok = load_tokenizer_from(args.data_dir)
    eot_id, _ = special_ids(tok)

    if args.question:
        print(answer(model, tok, eot_id, device, args.question, args.max_new_tokens, args.temperature))
        return

    print("reasoning-slm chat — type a question (Ctrl-C to quit).")
    try:
        while True:
            q = input("\n> ").strip()
            if not q:
                continue
            print(answer(model, tok, eot_id, device, q, args.max_new_tokens, args.temperature))
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
