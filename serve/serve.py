"""Serve the trained model on the from-scratch engine (Act III). SCAFFOLD.

Planned: load the GRPO checkpoint into ../mini-vllm (PagedAttention + continuous
batching + the custom Triton kernel), expose an OpenAI-compatible /v1/chat/completions
endpoint, and report tokens/sec for the live demo.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.parse_args()
    raise NotImplementedError("Serving stage lands in Act III (built on ../mini-vllm).")


if __name__ == "__main__":
    main()
