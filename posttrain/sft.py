"""Supervised fine-tuning on math/code chain-of-thought (Act II). SCAFFOLD.

Planned: load the pretrained checkpoint, format (prompt, CoT, answer) examples into
the <think>/<answer> template, mask the prompt tokens in the loss, train ~1 epoch on
GSM8K-train / MetaMathQA / OpenMathInstruct + a small code-instruct set.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.parse_args()
    raise NotImplementedError("SFT stage lands after Act I pretrain.")


if __name__ == "__main__":
    main()
