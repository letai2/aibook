"""جداسازی متن قبل از ساخت واژگان؛ دادهٔ ارزیابی در آموزش دخالت نمی‌کند."""

import hashlib
from pathlib import Path

from .tokenizer import CharacterTokenizer


def prepare_corpus(path, train_fraction=0.9, tokenizer=None):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    text = Path(path).read_text(encoding="utf-8")
    boundary = int(len(text) * train_fraction)
    train_text, valid_text = text[:boundary], text[boundary:]
    if not train_text or not valid_text:
        raise ValueError("both text splits must be nonempty")
    if tokenizer is None:
        tokenizer = CharacterTokenizer.from_text(train_text)
    train_ids, valid_ids = tokenizer.encode(train_text), tokenizer.encode(valid_text)
    metadata = {
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "train_fraction": train_fraction,
        "train_characters": len(train_text),
        "validation_characters": len(valid_text),
        "validation_unknown_rate": valid_ids.count(0) / len(valid_ids),
    }
    return train_ids, valid_ids, tokenizer, metadata
