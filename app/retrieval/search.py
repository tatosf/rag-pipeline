import numpy as np
from pydantic import BaseModel

from app.config import settings
from app.llm import embed_one
from app.retrieval import bm25, cosine_similarity
from app.retrieval.fusion import apply_confidence_boost, mmr, reciprocal_rank_fusion
from app.retrieval.store import get_chunks_by_ids, load_embedding_matrix


class RetrievedChunk(BaseModel):
    id: str
    doc_name: str
    page: int
    text: str
    confidence: int
    score: float


def hybrid_search(query: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], float]:
    """BM25 + cosine similarity → RRF → confidence boost → MMR. Returns (chunks, max_fused_score)."""
    k = top_k or settings.top_k
    candidate_k = settings.candidate_k

    query_vec = embed_one(query)

    bm25_results = bm25.search(query, top_k=candidate_k)
    cosine_results = cosine_similarity.search(query_vec, top_k=candidate_k)

    fused = reciprocal_rank_fusion([bm25_results, cosine_results])
    if not fused:
        return [], 0.0

    ids = [doc_id for doc_id, _ in fused]
    chunk_lookup = get_chunks_by_ids(ids)
    confidences = {cid: chunk_lookup[cid].confidence for cid in ids if cid in chunk_lookup}

    boosted = apply_confidence_boost(fused, confidences, boost=settings.confidence_boost)
    max_score = max((s for _, s in boosted), default=0.0)

    all_ids, all_matrix = load_embedding_matrix()
    candidate_set = set(ids)
    id_to_emb: dict[str, np.ndarray] = {
        doc_id: all_matrix[i]
        for i, doc_id in enumerate(all_ids)
        if doc_id in candidate_set
    }

    top = mmr(boosted, id_to_emb, lambda_=0.7, k=k)

    results: list[RetrievedChunk] = []
    for doc_id, score in top:
        c = chunk_lookup.get(doc_id)
        if c is None:
            continue
        results.append(RetrievedChunk(
            id=c.id,
            doc_name=c.doc_name,
            page=c.page,
            text=c.text,
            confidence=c.confidence,
            score=score,
        ))
    return results, float(max_score)