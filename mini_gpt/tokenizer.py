from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CharacterTokenizer:
    """A transparent character-level tokenizer for educational experiments."""

    UNK = "<|unk|>"

    def __init__(self, tokens: Iterable[str]) -> None:
        items = list(tokens)
        if any(not isinstance(t, str) or len(t) != 1 for t in items if t != self.UNK):
            raise ValueError("character tokens must each contain one Unicode code point")
        unique = sorted(set(items))
        self.id_to_token = [self.UNK, *[t for t in unique if t != self.UNK]]
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        unknown_id = self.token_to_id[self.UNK]
        return [self.token_to_id.get(char, unknown_id) for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        pieces = []
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token id {token_id} is outside the vocabulary")
            token = self.id_to_token[int(token_id)]
            pieces.append("�" if token == self.UNK else token)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.id_to_token, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_tokens(tokens)

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "CharacterTokenizer":
        if (not isinstance(tokens, list) or not tokens or tokens[0] != cls.UNK
                or any(not isinstance(t, str) for t in tokens)
                or len(tokens) != len(set(tokens))
                or any(len(t) != 1 for t in tokens[1:])):
            raise ValueError("invalid or duplicate tokenizer vocabulary")
        tokenizer = cls([])
        tokenizer.id_to_token = list(tokens)
        tokenizer.token_to_id = {token: i for i, token in enumerate(tokens)}
        return tokenizer
