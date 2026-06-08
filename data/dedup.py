"""Near-duplicate detection via MinHash + LSH, hand-rolled in NumPy.

Why from scratch: it removes a dependency, it is fully testable offline, and the
mechanics (universal hashing -> MinHash signature -> LSH banding) are worth owning.

How it works:
  - Shingle each document into a set of word n-grams.
  - Hash each shingle to 32 bits (BLAKE2b, so it is deterministic across runs —
    unlike Python's salted hash()).
  - A MinHash signature is, for each of `num_perm` random hash permutations
    h_i(x) = (a_i * x + b_i) mod p, the minimum hashed shingle value. The fraction
    of equal signature entries between two docs is an unbiased estimate of their
    Jaccard similarity.
  - LSH banding: split the signature into `bands` bands of `rows` rows. Two docs
    that match on *any* whole band become candidates; candidates are then verified
    by estimated Jaccard >= threshold. Banding makes dedup near-linear instead of
    O(n^2) all-pairs.

The deduper is streaming: feed it documents in order; the first occurrence of a
cluster is kept and later near-duplicates are dropped.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Mersenne prime 2^61 - 1, used as the modulus for the hash permutations. With
# 32-bit shingle hashes and 32-bit (a, b) the product a*x+b stays below 2^64, so
# uint64 arithmetic does not overflow before the modulo.
_MERSENNE = (1 << 61) - 1
_MAX32 = (1 << 32) - 1


def shingles(text: str, k: int = 5) -> set[str]:
    """Set of word k-grams. Falls back to the whole text if shorter than k words."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _hash32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big")


def _optimal_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) with bands*rows == num_perm whose LSH S-curve crosses ~0.5
    nearest the target threshold, i.e. minimize |threshold - (1/bands)^(1/rows)|."""
    best = None
    for bands in range(1, num_perm + 1):
        if num_perm % bands:
            continue
        rows = num_perm // bands
        thr = (1.0 / bands) ** (1.0 / rows)
        err = abs(thr - threshold)
        if best is None or err < best[0]:
            best = (err, bands, rows)
    return best[1], best[2]


class MinHashDedup:
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        shingle_size: int = 5,
        seed: int = 1,
        bands: int | None = None,
    ) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        if bands is None:
            bands, rows = _optimal_bands(num_perm, threshold)
        else:
            rows = num_perm // bands
        self.bands, self.rows = bands, rows

        rng = np.random.default_rng(seed)
        # 1 <= a < p, 0 <= b < p; kept within 32 bits to avoid uint64 overflow.
        self.a = rng.integers(1, _MAX32, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MAX32, size=num_perm, dtype=np.uint64)

        self._buckets: dict[tuple, int] = {}      # band signature -> kept doc id
        self._sigs: dict[int, np.ndarray] = {}    # kept doc id -> signature
        self.n_seen = 0
        self.n_duplicates = 0

    def signature(self, text: str) -> np.ndarray:
        sh = shingles(text, self.shingle_size)
        if not sh:
            return np.full(self.num_perm, _MERSENNE, dtype=np.uint64)
        x = np.fromiter((_hash32(s) for s in sh), dtype=np.uint64, count=len(sh))
        # (a[:,None] * x[None,:] + b[:,None]) mod p   -> min over shingles per perm
        hashed = (np.outer(self.a, x) + self.b[:, None]) % _MERSENNE
        return hashed.min(axis=1)

    def _band_keys(self, sig: np.ndarray):
        for band in range(self.bands):
            chunk = sig[band * self.rows : (band + 1) * self.rows]
            yield (band, chunk.tobytes())

    def _estimated_jaccard(self, s1: np.ndarray, s2: np.ndarray) -> float:
        return float(np.mean(s1 == s2))

    def is_duplicate(self, text: str) -> bool:
        """Return True if `text` near-duplicates an already-kept doc; else register it."""
        self.n_seen += 1
        sig = self.signature(text)
        for key in self._band_keys(sig):
            other = self._buckets.get(key)
            if other is not None and self._estimated_jaccard(sig, self._sigs[other]) >= self.threshold:
                self.n_duplicates += 1
                return True
        # not a duplicate: register this doc as the cluster representative
        doc_id = len(self._sigs)
        self._sigs[doc_id] = sig
        for key in self._band_keys(sig):
            self._buckets.setdefault(key, doc_id)
        return False

    def as_dict(self) -> dict:
        return {
            "method": "minhash-lsh",
            "num_perm": self.num_perm,
            "bands": self.bands,
            "rows": self.rows,
            "threshold": self.threshold,
            "seen": self.n_seen,
            "duplicates_removed": self.n_duplicates,
            "kept": len(self._sigs),
        }


class ExactDedup:
    """Cheap exact-duplicate fast path (BLAKE2b of normalized text)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.n_seen = 0
        self.n_duplicates = 0

    def is_duplicate(self, text: str) -> bool:
        self.n_seen += 1
        h = hashlib.blake2b(" ".join(text.split()).encode("utf-8"), digest_size=16).hexdigest()
        if h in self._seen:
            self.n_duplicates += 1
            return True
        self._seen.add(h)
        return False

    def as_dict(self) -> dict:
        return {"method": "exact", "seen": self.n_seen, "duplicates_removed": self.n_duplicates}


def build_deduper(cfg: dict):
    """Factory from the `dedup:` block of the data config."""
    if not cfg or not cfg.get("enabled", True) or cfg.get("method") == "none":
        return None
    if cfg.get("method") == "exact":
        return ExactDedup()
    return MinHashDedup(
        num_perm=cfg.get("num_perm", 128),
        threshold=cfg.get("jaccard_threshold", 0.8),
        shingle_size=cfg.get("shingle_size", 5),
    )
