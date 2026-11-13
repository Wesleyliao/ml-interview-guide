import numpy as np


class SelfAttentionNumpy:
    # Single head, no mask, no bias. Weights are supplied so tests are repeatable.
    def __init__(self, Wq, Wk, Wv):
        # Wq, Wk: (embed_dim, key_dim); Wv: (embed_dim, val_dim)
        self.Wq, self.Wk, self.Wv = Wq, Wk, Wv

    def forward(self, X):
        # Input: (batch, seq_len, embed_dim)
        # Outline: project Q/K/V -> scaled scores -> softmax -> weighted values.
        # einsum labels: b=batch, t=query position, s=key position,
        # e=embedding feature, k=key feature, v=value feature.

        # 1. Project each token into query, key and value vectors.
        Q = np.einsum('bte,ek->btk', X, self.Wq)  # (B, T, K)
        K = np.einsum('bse,ek->bsk', X, self.Wk)  # (B, S, K); S=T
        V = np.einsum('bse,ev->bsv', X, self.Wv)  # (B, S, V)

        # 2. Dot each query with every key, summing over key features.
        scores = np.einsum('btk,bsk->bts', Q, K) / np.sqrt(Q.shape[-1])
        # scores: (B, T, S). A causal mask would go here, before softmax.

        # 3. Stable softmax over keys for each query (last axis).
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        # 4. Sum values weighted by attention, contracting key positions.
        return np.einsum('bts,bsv->btv', weights, V)  # (B, T, V)


class TransformerBlockNumpy:
    # Same pre-norm structure as PyTorch; no learned norm scale/bias or biases.
    def __init__(self, Wq, Wk, Wv, W1, W2):
        # Wq/Wk/Wv: (E, E); W1: (E, H); W2: (H, E).
        # Attention must return E features so the residual shapes match.
        self.attention = SelfAttentionNumpy(Wq, Wk, Wv)
        self.W1, self.W2 = W1, W2

    def layer_norm(self, X):
        # Normalize features independently for each token: (B, T, E).
        mean = X.mean(axis=-1, keepdims=True)
        variance = X.var(axis=-1, keepdims=True)
        return (X - mean) / np.sqrt(variance + 1e-5)

    def forward(self, X):
        # 1. Normalize -> self-attention -> residual: (B, T, E).
        X = X + self.attention.forward(self.layer_norm(X))

        # 2. Normalize -> expand -> ReLU -> project back -> residual.
        normalized = self.layer_norm(X)
        hidden = np.einsum('bte,eh->bth', normalized, self.W1)
        hidden = np.maximum(hidden, 0)  # ReLU: (B, T, H)
        output = np.einsum('bth,he->bte', hidden, self.W2)
        return X + output  # (B, T, E)
