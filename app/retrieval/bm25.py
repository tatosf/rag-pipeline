import math
import re
from dataclasses import dataclass

from app.retrieval.store import Chunk, all_chunks

_TOKEN = re.compile(r"\w+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those", "it", "its",
    "as", "so", "than", "too", "very", "just", "about", "into", "up", "down", "out",
})

_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class BM25Index:
    chunk_ids: list[str]
    doc_tokens: list[list[str]]
    doc_lengths: list[int]
    avg_length: float
    doc_freq: dict[str, int]
    n_docs: int


def build_index(chunks: list[Chunk]) -> BM25Index:
    doc_tokens = [_tokenize(c.text) for c in chunks]
    doc_lengths = [len(toks) for toks in doc_tokens]
    n_docs = len(chunks)
    avg_length = sum(doc_lengths) / n_docs if n_docs else 0.0
    doc_freq: dict[str, int] = {}
    for toks in doc_tokens:
        for term in set(toks):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return BM25Index(
        chunk_ids=[c.id for c in chunks],
        doc_tokens=doc_tokens,
        doc_lengths=doc_lengths,
        avg_length=avg_length,
        doc_freq=doc_freq,
        n_docs=n_docs,
    )


def _idf(term: str, idx: BM25Index) -> float:
    df = idx.doc_freq.get(term, 0)
    return math.log((idx.n_docs - df + 0.5) / (df + 0.5) + 1.0)


def search(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    idx = build_index(all_chunks())
    if idx.n_docs == 0:
        return []
    q_terms = _tokenize(query)
    if not q_terms:
        return []

    scores = [0.0] * idx.n_docs
    for term in q_terms:
        if term not in idx.doc_freq:
            continue
        idf = _idf(term, idx)
        for i, toks in enumerate(idx.doc_tokens):
            tf = toks.count(term)
            if tf == 0:
                continue
            dl = idx.doc_lengths[i]
            norm = 1 - _B + _B * (dl / idx.avg_length) if idx.avg_length else 1.0
            scores[i] += idf * (tf * (_K1 + 1)) / (tf + _K1 * norm)

    ranked = sorted(
        ((idx.chunk_ids[i], s) for i, s in enumerate(scores) if s > 0),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]