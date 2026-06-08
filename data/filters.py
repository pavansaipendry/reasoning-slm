"""Quality filtering — Gopher/RefinedWeb-style heuristics, pure Python (no deps).

Each document is checked against a set of cheap rules; the first rule it fails returns
a reason string so the pipeline can report *why* documents were dropped (the rejection
histogram is one of the more interesting artifacts of a data pipeline). Prose and code
use different rulesets because what counts as junk differs between them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sources import Document


@dataclass
class FilterConfig:
    min_chars: int = 200
    max_chars: int = 100_000
    min_mean_word_len: float = 3.0
    max_mean_word_len: float = 10.0
    max_symbol_to_word_ratio: float = 0.10
    min_alpha_fraction: float = 0.60
    max_dup_line_fraction: float = 0.30
    # code-specific
    code_max_mean_line_len: float = 200.0
    code_min_alnum_fraction: float = 0.40

    @classmethod
    def from_dict(cls, d: dict) -> "FilterConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


_SYMBOLS = ("#", "...", "…")


def _alpha_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def _alnum_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalnum() for c in text) / len(text)


def _dup_line_fraction(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return 0.0
    seen, dup = set(), 0
    for ln in lines:
        if ln in seen:
            dup += 1
        else:
            seen.add(ln)
    return dup / len(lines)


def check_prose(doc: Document, cfg: FilterConfig) -> str | None:
    """Return a rejection reason, or None to keep."""
    text = doc.text
    n = len(text)
    if n < cfg.min_chars:
        return "too_short"
    if n > cfg.max_chars:
        return "too_long"

    words = text.split()
    if not words:
        return "no_words"
    mean_word_len = sum(len(w) for w in words) / len(words)
    if mean_word_len < cfg.min_mean_word_len:
        return "mean_word_len_low"
    if mean_word_len > cfg.max_mean_word_len:
        return "mean_word_len_high"

    sym = sum(text.count(s) for s in _SYMBOLS)
    if sym / len(words) > cfg.max_symbol_to_word_ratio:
        return "symbol_ratio_high"

    if _alpha_fraction(text) < cfg.min_alpha_fraction:
        return "alpha_fraction_low"

    if _dup_line_fraction(text) > cfg.max_dup_line_fraction:
        return "dup_lines_high"

    return None


def check_code(doc: Document, cfg: FilterConfig) -> str | None:
    text = doc.text
    n = len(text)
    if n < cfg.min_chars:
        return "too_short"
    if n > cfg.max_chars:
        return "too_long"

    lines = text.splitlines() or [text]
    mean_line_len = sum(len(ln) for ln in lines) / len(lines)
    if mean_line_len > cfg.code_max_mean_line_len:
        return "code_line_len_high"      # likely minified / data blob

    if _alnum_fraction(text) < cfg.code_min_alnum_fraction:
        return "code_alnum_low"          # mostly punctuation/noise

    return None


def keep_document(doc: Document, cfg: FilterConfig) -> str | None:
    """Dispatch by document kind. Returns None to keep, else a reason string."""
    if doc.kind == "code":
        return check_code(doc, cfg)
    return check_prose(doc, cfg)


class FilterStats:
    """Counts kept docs and rejections-by-reason for the pipeline report."""

    def __init__(self) -> None:
        self.kept = 0
        self.reasons: dict[str, int] = {}

    def record(self, reason: str | None) -> bool:
        if reason is None:
            self.kept += 1
            return True
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return False

    @property
    def rejected(self) -> int:
        return sum(self.reasons.values())

    def as_dict(self) -> dict:
        return {"kept": self.kept, "rejected": self.rejected, "by_reason": dict(self.reasons)}
