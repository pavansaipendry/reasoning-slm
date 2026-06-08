"""Pretraining entrypoint (Act I). SCAFFOLD.

Planned: memmap .bin data loader, AdamW + cosine schedule, grad accumulation to
~256k tokens/step, bf16 autocast, MFU + tokens/sec logging, periodic val + ckpt.
Implemented in the pretrain stage; the data it consumes (data/build/*.bin) is
produced by the already-working Act I pipeline (`python -m data.pipeline`).
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_100m.yaml")
    args = ap.parse_args()
    raise NotImplementedError(
        "Pretrain stage not implemented yet. Act I data pipeline is ready: "
        "run `python -m data.pipeline --config configs/data.yaml` first."
    )


if __name__ == "__main__":
    main()
