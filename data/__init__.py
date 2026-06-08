"""Act I data pipeline: sources -> quality filters -> MinHash dedup -> tokenizer -> packed .bin shards."""

from .sources import Document

__all__ = ["Document"]
