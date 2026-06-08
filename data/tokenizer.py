"""Train and load a byte-level BPE tokenizer (the `tokenizers` library).

Byte-level BPE never produces unknown tokens (it falls back to raw bytes), which is
exactly what we want for a mixed math + code + prose corpus full of symbols.
"""

from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 32000,
    special_tokens: list[str] | None = None,
    out_path: str = "tokenizer.json",
) -> Tokenizer:
    special_tokens = special_tokens or ["<|endoftext|>", "<|pad|>"]

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(out_path)
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def token_id(tok: Tokenizer, token: str) -> int:
    tid = tok.token_to_id(token)
    if tid is None:
        raise KeyError(f"token {token!r} not in tokenizer vocab")
    return tid
