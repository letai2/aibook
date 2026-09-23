"""Small inspectable retrieval components, not a semantic-search service.

The default index is exact cosine search over lexical count vectors. The
optional four-word teaching experiment trains its own embeddings explicitly;
it neither downloads weights nor changes the default index.
"""
from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable


@dataclass(frozen=True)
class Document:
    id: str
    text: str

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("A document needs a nonempty ID")
        if not isinstance(self.text, str):
            raise ValueError("Document text must be a string")


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    citations: tuple[str, ...]
    abstained: bool


COURSE_DOCUMENTS = (
    Document("practice", "Run the lesson exercise in Jupyter. Predict the result before running it."),
    Document("checkpoint", "Checkpoint stores model state and vocabulary. Open files only from a trusted source."),
    Document("evaluation", "Evaluate on unseen text. Lower training loss does not guarantee a correct answer."),
)


def chunk_document(document: Document, *, chunk_words: int = 24,
                   overlap_words: int = 4) -> list[Chunk]:
    """Whitespace-word chunks with exact Python-string offsets into the source.

    Words here are a splitting convenience, NOT the model's tokens. We keep
    original whitespace inside each span. Offsets count Unicode code points.
    """
    if not isinstance(document, Document):
        raise ValueError("Expected a Document")
    if (type(chunk_words) is not int or chunk_words <= 0 or
            type(overlap_words) is not int or not 0 <= overlap_words < chunk_words):
        raise ValueError("Require 0 <= overlap_words < chunk_words")
    words = list(re.finditer(r"\S+", document.text))
    chunks = []
    for first in range(0, len(words), chunk_words-overlap_words):
        last = min(first+chunk_words, len(words))
        start, end = words[first].start(), words[last-1].end()
        chunks.append(Chunk(f"{document.id}:{start}-{end}", document.id,
                            document.text[start:end], start, end))
        if last == len(words):
            break
    return chunks


def terms(text: str) -> list[str]:
    """Casefolded word-like units, with ZWNJ retained inside Persian words.

    No stemming, synonym expansion, or Persian/Arabic character normalization.
    This is a search convention, separate from CharacterTokenizer.
    """
    if not isinstance(text, str):
        raise ValueError("Search text must be a string")
    return re.findall(r"[^\W_]+(?:\u200c[^\W_]+)*", text.casefold())


def _validate_search(chunks, k):
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    chunks = tuple(chunks)
    if any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise ValueError("Expected Chunk records")
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique")
    return chunks


def keyword_search(query: str, chunks: Iterable[Chunk], *, k: int = 3) -> list[SearchHit]:
    """Rank by fraction of distinct query terms present; omit zero-overlap hits."""
    chunks = _validate_search(chunks, k)
    wanted = set(terms(query))
    if not wanted:
        return []
    hits = [SearchHit(chunk, len(wanted & set(terms(chunk.text)))/len(wanted))
            for chunk in chunks]
    return sorted((hit for hit in hits if hit.score > 0),
                  key=lambda hit: (-hit.score, hit.chunk.id))[:k]


def cosine_similarity(left, right) -> float:
    """Cosine with an explicit zero-vector convention: return zero similarity."""
    if len(left) != len(right):
        raise ValueError("Vector dimensions differ")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Vectors must be finite")
    left_scale = max((abs(value) for value in left), default=0)
    right_scale = max((abs(value) for value in right), default=0)
    if left_scale == 0 or right_scale == 0:
        return 0.0
    # Scaling each vector leaves cosine unchanged and avoids squaring enormous
    # or tiny finite values before normalization.
    left_unit_scale = [value/left_scale for value in left]
    right_unit_scale = [value/right_scale for value in right]
    left_norm = math.sqrt(sum(value*value for value in left_unit_scale))
    right_norm = math.sqrt(sum(value*value for value in right_unit_scale))
    score = sum((a/left_norm)*(b/right_norm)
                for a, b in zip(left_unit_scale, right_unit_scale))
    return max(-1.0, min(1.0, score))  # Bound floating-point round-off.


class VectorIndex:
    """A shared lexical vocabulary and exact scan, with no learned semantics."""
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = _validate_search(chunks, 1)
        self.vocabulary = tuple(sorted({term for chunk in self.chunks for term in terms(chunk.text)}))
        self.vectors = tuple(self.encode(chunk.text) for chunk in self.chunks)

    def encode(self, text: str) -> tuple[int, ...]:
        counts = Counter(terms(text))
        return tuple(counts[term] for term in self.vocabulary)

    def search(self, query: str, *, k: int = 3, min_score: float = 0.0) -> list[SearchHit]:
        _validate_search(self.chunks, k)
        if not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("min_score must be finite and between zero and one")
        query_vector = self.encode(query)
        hits = [SearchHit(chunk, cosine_similarity(query_vector, vector))
                for chunk, vector in zip(self.chunks, self.vectors)]
        return sorted((hit for hit in hits if hit.score > 0 and hit.score >= min_score),
                      key=lambda hit: (-hit.score, hit.chunk.id))[:k]


def answer_from_hits(hits: Iterable[SearchHit], *, min_score: float = 0.0) -> GroundedAnswer:
    """Quote selected evidence exactly; not a learned answer or truth verifier.

    Multiple passages remain separate, including contradictions. Existence of a
    citation is traceability, not proof that a passage is correct or relevant.
    """
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be finite and between zero and one")
    selected = [hit for hit in hits if hit.score > 0 and hit.score >= min_score]
    if not selected:
        return GroundedAnswer("No sufficient evidence was found in the supplied documents.", (), True)
    if len({hit.chunk.id for hit in selected}) != len(selected):
        raise ValueError("Duplicate evidence IDs")
    text = "\n".join(f"[{hit.chunk.id}] {hit.chunk.text}" for hit in selected)
    return GroundedAnswer(text, tuple(hit.chunk.id for hit in selected), False)


def train_tiny_embeddings(*, steps: int = 100):
    """An explicitly supervised, four-term toy representation-learning experiment.

    Positive pairs (save/keep and calculate/multiply) are *provided relevance
    labels*, not knowledge discovered from a corpus. There is no held-out claim.
    Returns (words, initial_vectors, trained_vectors, initial_loss, final_loss).
    PyTorch is imported only when the learner explicitly runs this experiment.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    import torch
    from torch import nn
    from torch.nn import functional as F

    words = ("save", "keep", "calculate", "multiply")
    initial = torch.tensor([[1., 0.2], [-0.2, 1.], [-1., 0.2], [0.2, -1.]])
    table = nn.Embedding.from_pretrained(initial.clone(), freeze=False)
    left = torch.tensor([0, 2, 0, 1])
    right = torch.tensor([1, 3, 2, 3])
    targets = torch.tensor([1., 1., -1., -1.])
    optimizer = torch.optim.SGD(table.parameters(), lr=0.2)

    def loss():
        scores = F.cosine_similarity(table(left), table(right), dim=-1)
        return ((scores-targets)**2).mean()

    first = loss().item()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss().backward()
        optimizer.step()
    return words, initial.tolist(), table.weight.detach().tolist(), first, loss().item()
