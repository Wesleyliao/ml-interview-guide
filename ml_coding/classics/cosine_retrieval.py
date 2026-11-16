import numpy as np


def top_k_cosine(query, items, k):
    # query: (d,), items: (n, d). Returns top-k (indices, cosine_scores).
    query = np.asarray(query, dtype=float)
    items = np.asarray(items, dtype=float)
    k = min(max(k, 0), len(items))

    # 1. L2-normalize query and items (use denominator 1 so zero vectors stay 0).
    q_norm = np.linalg.norm(query)
    q = query / (q_norm if q_norm > 0 else 1.0)
    item_norms = np.linalg.norm(items, axis=1, keepdims=True)
    normed_items = items / np.where(item_norms > 0, item_norms, 1.0)

    # 2. Dot product of unit vectors equals cosine similarity: (n,).
    scores = normed_items @ q

    # 3. Sort descending (stable keeps smaller index first on exact ties).
    indices = np.argsort(-scores, kind="stable")[:k]
    return indices, scores[indices]
