"""Evaluation harness (Act III). SCAFFOLD.

Planned: wrap the model for lm-eval-harness (GSM8K, MMLU-subset) and run the custom
code-execution eval (eval/code_exec.py). Emit the base vs SFT vs RL comparison table
that goes in the README, including failure cases.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default="gsm8k,humaneval_subset")
    ap.parse_args()
    raise NotImplementedError("Eval stage lands in Act III.")


if __name__ == "__main__":
    main()
