import numpy as np

from app.retrieval.store import load_embedding_matrix


def search(query_vec: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
    """Cosine similarity search. query_vec must be L2-normalized."""
    ids, matrix = load_embedding_matrix()
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:top_k]
    return [(ids[int(i)], float(scores[int(i)])) for i in order]