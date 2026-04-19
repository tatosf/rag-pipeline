import math

import numpy as np


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists. Output is (id, fused_score), descending."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _orig) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def apply_confidence_boost(
    scored: list[tuple[str, float]],
    confidences: dict[str, int],
    boost: float = 0.1,
) -> list[tuple[str, float]]:
    """Multiplicatively nudge scores by log(1 + confidence)."""
    out = [
        (doc_id, score * (1.0 + boost * math.log(1.0 + confidences.get(doc_id, 1))))
        for doc_id, score in scored
    ]
    return sorted(out, key=lambda x: x[1], reverse=True)


def mmr(
    candidates: list[tuple[str, float]],
    embeddings: dict[str, np.ndarray],
    lambda_: float = 0.7,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Maximal Marginal Relevance: relevance vs. novelty against already-selected docs."""
    if not candidates:
        return []
    max_score = max(s for _, s in candidates) or 1.0
    pool = [(doc_id, score, score / max_score) for doc_id, score in candidates]
    selected: list[tuple[str, float]] = []

    while pool and len(selected) < k:
        best_idx = 0
        best_mmr = -float("inf")
        for i, (doc_id, _, rel_norm) in enumerate(pool):
            vec = embeddings.get(doc_id)
            if vec is None or not selected:
                max_sim = 0.0
            else:
                max_sim = max(
                    float(vec @ embeddings[s_id])
                    for s_id, _ in selected
                    if s_id in embeddings
                )
            mmr_score = lambda_ * rel_norm - (1 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
        doc_id, orig_score, _ = pool.pop(best_idx)
        selected.append((doc_id, orig_score))
    return selected