"""Explicit prompt budgeting for the educational model-to-system bridge.

The caller chooses priority order and supplies the *actual* tokenizer counter.
No model weights, files, conversation state, or caches are changed here.
"""
from dataclasses import dataclass
import json
from typing import Callable, Iterable


@dataclass(frozen=True)
class ContextItem:
    id: str
    text: str
    kind: str = "evidence"

    def __post_init__(self):
        if not all(isinstance(value, str) and value.strip()
                   for value in (self.id, self.text, self.kind)):
            raise ValueError("Context items require nonempty id, text and kind")


@dataclass(frozen=True)
class ContextPack:
    prompt: str
    token_count: int
    included_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]


def _render(question, items):
    # JSON escaping preserves boundaries; it is NOT a prompt-injection defense.
    return json.dumps({"question": question, "context": [
        {"id": item.id, "kind": item.kind, "text": item.text} for item in items
    ]}, ensure_ascii=False, separators=(",", ":"))


def build_context(question: str, items: Iterable[ContextItem], *, max_tokens: int,
                  reserve_tokens: int, count_tokens: Callable[[str], int]) -> ContextPack:
    """Greedily keep whole items in caller priority order within a token budget.

    Count each complete candidate serialization, not the sum of separately
    tokenized pieces. A rejected large item does not prevent a later small one
    from fitting. The question is mandatory; empty input is rejected.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty question is required")
    if (type(max_tokens) is not int or max_tokens <= 0 or
            type(reserve_tokens) is not int or not 0 <= reserve_tokens < max_tokens):
        raise ValueError("Require max_tokens > reserve_tokens >= 0, both integers")
    items = tuple(items)
    if any(not isinstance(item, ContextItem) for item in items):
        raise ValueError("Expected ContextItem records")
    if len({item.id for item in items}) != len(items):
        raise ValueError("Context item IDs must be unique")

    def count(prompt):
        value = count_tokens(prompt)
        if type(value) is not int or value < 0:
            raise ValueError("count_tokens must return a nonnegative integer")
        return value

    budget = max_tokens-reserve_tokens
    selected, omitted = [], []
    if count(_render(question, selected)) > budget:
        raise ValueError("Question and serialization exceed the input budget")
    for item in items:
        if count(_render(question, selected+[item])) <= budget:
            selected.append(item)
        else:
            omitted.append(item.id)
    prompt = _render(question, selected)
    return ContextPack(prompt, count(prompt), tuple(item.id for item in selected), tuple(omitted))
