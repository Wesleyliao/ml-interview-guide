import numpy as np


def masked_pool(X, mask, mode="mean"):
    # X: (B, T, D) token embeddings; mask: (B, T) with 1/True for real tokens, 0/False for pad.
    # Returns: (B, D), with zeros for an all-padding row.
    valid = np.asarray(mask, dtype=bool)[..., None]  # (B, T, 1) broadcasts over D
    count = valid.sum(axis=1)  # (B, 1)

    if mode == "mean":
        # 1. Zero out padded positions, sum over T, and divide by valid token count.
        total = np.where(valid, X, 0.0).sum(axis=1)
        return total / np.maximum(count, 1)

    if mode == "max":
        # 2. Fill pad with -inf before max so 0-padding never beats negative real values.
        pooled = np.where(valid, X, -np.inf).max(axis=1)
        return np.where(count > 0, pooled, 0.0)
