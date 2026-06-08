"""Corpus sources.

A `Document` is the unit that flows through the whole pipeline. Real sources stream
from HuggingFace (no full download); `sample_documents()` returns a tiny in-memory
corpus — including deliberate junk and near-duplicates — so the entire pipeline can be
run and tested offline with no network and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass
class Document:
    text: str
    source: str
    kind: str = "prose"          # "prose" | "code"
    meta: dict = field(default_factory=dict)


def stream_source(spec: dict, limit: int | None = None) -> Iterator[Document]:
    """Stream Documents from one source spec (see configs/data.yaml).

    Uses `datasets` in streaming mode so nothing is fully downloaded. Imported
    lazily so the offline `--sample` path needs neither the lib nor the network.
    """
    import time

    from datasets import load_dataset

    name = spec["name"]
    kind = spec.get("kind", "prose")
    text_key = spec.get("text_key", "text")
    max_docs = spec.get("max_docs")
    if limit is not None:
        max_docs = min(max_docs, limit) if max_docs else limit

    def _open(skip: int):
        ds = load_dataset(
            spec["hf_path"],
            spec.get("hf_config"),
            data_dir=spec.get("data_dir"),
            split=spec.get("split", "train"),
            streaming=True,
        )
        return ds.skip(skip) if skip else ds

    # Resilient streaming: on a transient error (e.g. HF rate-limit 429), reopen
    # the stream and skip past what we already yielded, so a long build survives.
    n, attempts = 0, 0
    while True:
        try:
            for row in _open(n):
                text = row.get(text_key)
                if not text:
                    continue
                yield Document(text=text, source=name, kind=kind, meta={"idx": n})
                n += 1
                if max_docs and n >= max_docs:
                    return
            return  # stream exhausted cleanly
        except Exception as e:  # noqa: BLE001 - want to survive any transient stream error
            attempts += 1
            if attempts > 6:
                print(f"[warn] source {name}: giving up after {n} docs ({type(e).__name__}: {str(e)[:80]})")
                return
            print(f"[warn] source {name}: error at doc {n}, retry {attempts} ({type(e).__name__})")
            time.sleep(5 * attempts)


def stream_mixture(config: dict, limit_per_source: int | None = None) -> Iterator[Document]:
    """Stream all sources in a data config, one after another."""
    for spec in config["sources"]:
        yield from stream_source(spec, limit=limit_per_source)


def sample_documents() -> list[Document]:
    """A small offline corpus that exercises every pipeline stage.

    Contains: clean prose, clean code, a near-duplicate of one prose doc (should be
    caught by MinHash), an exact duplicate (caught too), and several junk docs that
    each trip a *different* quality filter.
    """
    good_prose = (
        "The derivative of a function measures how its output changes as the input "
        "changes. For a polynomial, the power rule gives a quick way to differentiate: "
        "the derivative of x raised to the n is n times x raised to the n minus one. "
        "This is one of the first results students learn in a calculus course, and it "
        "generalizes neatly to sums and scalar multiples of such terms."
    )
    near_dup_prose = (
        "The derivative of a function measures how its output changes as the input "
        "varies. For a polynomial, the power rule gives a fast way to differentiate: "
        "the derivative of x to the n is n times x to the n minus one. This is among "
        "the first results students meet in a calculus course, and it generalizes "
        "neatly to sums and scalar multiples of such terms."
    )
    good_prose_2 = (
        "Bayes' theorem relates the conditional probability of an event to its reverse "
        "conditional. It states that the probability of A given B equals the probability "
        "of B given A times the probability of A, divided by the probability of B. In "
        "practice it lets us update a prior belief into a posterior once evidence arrives, "
        "which is the backbone of a great deal of statistical reasoning and inference."
    )
    good_code = (
        "def gcd(a, b):\n"
        "    \"\"\"Return the greatest common divisor of a and b.\"\"\"\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return abs(a)\n\n\n"
        "def lcm(a, b):\n"
        "    if a == 0 or b == 0:\n"
        "        return 0\n"
        "    return abs(a * b) // gcd(a, b)\n"
    )

    docs = [
        Document(good_prose, "sample-prose", "prose"),
        Document(near_dup_prose, "sample-prose", "prose"),     # near-dup of #0
        Document(good_prose, "sample-prose", "prose"),         # exact dup of #0
        Document(good_prose_2, "sample-prose", "prose"),
        Document(good_code, "sample-code", "code"),
        # --- junk, each trips a different filter ---
        Document("too short", "sample-junk", "prose"),                       # min_chars
        Document("#" * 600, "sample-junk", "prose"),                         # symbol ratio / alpha
        Document(("buy now! " * 80).strip(), "sample-junk", "prose"),        # dup-line / repetition-ish
        Document("a a a a a a a a a a a a a a a a a a a a a " * 20,          # mean word len too low
                 "sample-junk", "prose"),
    ]
    return docs


def iter_texts(docs: Iterable[Document]) -> Iterator[str]:
    for d in docs:
        yield d.text
