"""Act I orchestrator: sources -> filter -> dedup -> tokenizer -> packed .bin shards.

Run it two ways:

    # offline, no network/GPU — verifies every stage on the built-in sample corpus
    python -m data.pipeline --sample --out data/build_sample

    # the real mixture, streamed from HuggingFace
    python -m data.pipeline --config configs/data.yaml --out data/build

Pass 1 streams the raw sources, applies quality filters and MinHash dedup, and writes
the surviving documents to a compressed `clean.jsonl.zst`. The tokenizer is then trained
on a sample of that clean corpus, and finally the whole clean corpus is tokenized and
packed. A `report.json` captures the filter/dedup statistics — the most informative
artifact of the run.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterator

import zstandard as zstd

from .dedup import build_deduper
from .filters import FilterConfig, FilterStats, keep_document
from .sources import Document, sample_documents, stream_mixture


# Defaults used by --sample so it needs no config file.
_SAMPLE_CONFIG = {
    "filters": {},
    "dedup": {"enabled": True, "method": "minhash", "num_perm": 64,
              "jaccard_threshold": 0.7, "shingle_size": 3},
    "tokenizer": {"train": True, "vocab_size": 512, "sample_docs": 1000,
                  "special_tokens": ["<|endoftext|>", "<|pad|>"]},
    "pack": {"dtype": "uint16", "eot_token": "<|endoftext|>", "val_fraction": 0.0},
}


def _write_clean(docs, clean_path: str, fcfg: FilterConfig, dcfg: dict):
    """Pass 1: filter + dedup, streaming surviving docs to clean_path. Returns stats."""
    fstats = FilterStats()
    deduper = build_deduper(dcfg)
    n_in = 0
    cctx = zstd.ZstdCompressor(level=10)
    with open(clean_path, "wb") as raw, cctx.stream_writer(raw) as out:
        for doc in docs:
            n_in += 1
            if not fstats.record(keep_document(doc, fcfg)):
                continue
            if deduper is not None and deduper.is_duplicate(doc.text):
                continue
            line = json.dumps({"text": doc.text, "source": doc.source, "kind": doc.kind})
            out.write((line + "\n").encode("utf-8"))
    return {
        "documents_in": n_in,
        "filter": fstats.as_dict(),
        "dedup": deduper.as_dict() if deduper is not None else {"method": "none"},
    }


def _read_clean(clean_path: str) -> Iterator[Document]:
    dctx = zstd.ZstdDecompressor()
    with open(clean_path, "rb") as raw, dctx.stream_reader(raw) as fh:
        buf = b""
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line:
                    r = json.loads(line)
                    yield Document(r["text"], r["source"], r.get("kind", "prose"))


def run(config: dict, out_dir: str, sample: bool, limit: int | None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    clean_path = os.path.join(out_dir, "clean.jsonl.zst")
    fcfg = FilterConfig.from_dict(config.get("filters", {}))

    docs = iter(sample_documents()) if sample else stream_mixture(config, limit_per_source=limit)
    print("[1/3] filtering + dedup -> clean corpus ...")
    stats = _write_clean(docs, clean_path, fcfg, config.get("dedup", {}))
    print(f"      in={stats['documents_in']}  kept_by_filter={stats['filter']['kept']}  "
          f"dedup_removed={stats['dedup'].get('duplicates_removed', 0)}")

    # --- tokenizer ---
    tcfg = config.get("tokenizer", {})
    tok = None
    if tcfg.get("train", True):
        from .tokenizer import train_tokenizer
        n_sample = tcfg.get("sample_docs", 50000)
        print(f"[2/3] training {tcfg.get('vocab_size', 32000)}-vocab BPE on up to {n_sample} docs ...")

        def _texts():
            for i, d in enumerate(_read_clean(clean_path)):
                if i >= n_sample:
                    break
                yield d.text

        tok_out = tcfg.get("out") or os.path.join(out_dir, "tokenizer.json")
        tok = train_tokenizer(
            _texts(),
            vocab_size=tcfg.get("vocab_size", 32000),
            special_tokens=tcfg.get("special_tokens"),
            out_path=tok_out,
        )
        print(f"      saved tokenizer -> {tok_out}  (vocab={tok.get_vocab_size()})")
    else:
        from .tokenizer import load_tokenizer
        tok = load_tokenizer(tcfg["out"])

    # --- pack ---
    from .pack import pack_documents
    pcfg = config.get("pack", {})
    print("[3/3] tokenizing + packing -> train.bin / val.bin ...")
    manifest = pack_documents(
        _read_clean(clean_path),
        tok,
        out_dir,
        dtype=pcfg.get("dtype", "uint16"),
        eot_token=pcfg.get("eot_token", "<|endoftext|>"),
        val_fraction=pcfg.get("val_fraction", 0.0005),
    )
    print(f"      packed {manifest['train_tokens']:,} train + {manifest['val_tokens']:,} val tokens "
          f"from {manifest['n_docs']} docs")

    report = {"out_dir": out_dir, "sample": sample, "stats": stats, "manifest": manifest}
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def load_config(path: str | None, sample: bool) -> dict:
    if sample and not path:
        return _SAMPLE_CONFIG
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def main():
    ap = argparse.ArgumentParser(description="reasoning-slm Act I data pipeline")
    ap.add_argument("--config", default=None, help="path to a data config yaml")
    ap.add_argument("--sample", action="store_true", help="use the built-in offline sample corpus")
    ap.add_argument("--out", default="data/build", help="output directory")
    ap.add_argument("--limit", type=int, default=None, help="cap docs per source (debugging)")
    args = ap.parse_args()

    config = load_config(args.config, args.sample)
    report = run(config, args.out, args.sample, args.limit)
    print("\nReport:")
    print(json.dumps(report["stats"], indent=2))


if __name__ == "__main__":
    main()
