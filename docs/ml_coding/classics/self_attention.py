import torch
import torch.nn as nn
import torch.nn.functional as F

# Your single-head, unmasked baseline. Add causal masking as a follow-up.
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, key_dim, val_dim):
        super().__init__()
        self.key_dim = key_dim
        self.Wk = nn.Linear(embed_dim, key_dim)
        self.Wq = nn.Linear(embed_dim, key_dim)
        self.Wv = nn.Linear(embed_dim, val_dim)

    def forward(self, X):
        # X: (batch, seq_len, embed_dim)
        # 1. Project tokens to Q/K: (B, T, key_dim), V: (B, T, val_dim).
        K = self.Wk(X)
        Q = self.Wq(X)
        V = self.Wv(X)
        # 2. Compare every query to every key and scale: (B, T, T).
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.key_dim ** 0.5
        # 3. Normalize over key positions for each query.
        attention = F.softmax(scores, dim=2)
        # 4. Weighted sum of values for every query.
        return torch.bmm(attention, V)  # (batch, seq_len, val_dim)


class TransformerBlock(nn.Module):
    # Minimal encoder-style block: one attention head, no masking or dropout.
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim, embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, X):
        # Input / output: (batch, seq_len, embed_dim).
        # 1. Normalize, mix information across tokens, then add the residual.
        X = X + self.attention(self.norm1(X))
        # 2. Normalize, transform each token independently, add the residual.
        X = X + self.ffn(self.norm2(X))
        return X
