"""Lab 1 (45 min): Classical ML From Scratch & Numerically Stable Losses in PyTorch.

Covers the three most common ML-coding warmups and screen questions:
1. `stable_log_softmax` & `masked_cross_entropy` (OpenAI / Anthropic loss & stability check)
2. `kmeans_pytorch` (Vectorized Lloyd's K-Means — one of the most frequent ML coding screens)
3. `GaussianNaiveBayes` (Log-space class posteriors without underflow)

Run `.venv/bin/python ml_coding/pytorch_transformers/ex1_classical_and_losses.py` to test your code.
Reference answers: `ml_coding/pytorch_transformers/solutions_pytorch.py`.
"""

from __future__ import annotations

import math
import torch


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log-softmax along `dim` WITHOUT calling `F.log_softmax` or `torch.logsumexp`.

    Must remain finite for extreme inputs like `logits = +/- 1000.0`.
    """
    raise NotImplementedError("Implement stable_log_softmax.")


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean cross-entropy over non-ignored tokens WITHOUT calling `F.cross_entropy`.

    Shapes:
        logits: (B, T, V) or (N, V)
        targets: (B, T) or (N,) int64 with class indices in [0, V) or `ignore_index`

    Traps to handle:
        - Clamp ignored targets before `gather` so `-100` never triggers an out-of-bounds index error.
        - If ALL tokens equal `ignore_index`, return `logits.sum() * 0.0` (differentiable 0.0, not NaN).
    """
    raise NotImplementedError("Implement masked_cross_entropy.")


def kmeans_pytorch(
    x: torch.Tensor,
    k: int,
    num_iters: int = 20,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized K-Means on `x` of shape (N, D) -> `(centroids (k, D), assignments (N,))`.

    Rules:
        1. Sample `k` initial centroids without replacement using `torch.Generator().manual_seed(seed)`.
        2. Compute `(N, k)` squared Euclidean distances without Python loops over `N`.
        3. If a cluster becomes empty, re-seed it to the point farthest from its current centroid.
        4. Stop early when `assignments` stop changing.
    """
    raise NotImplementedError("Implement kmeans_pytorch.")


class GaussianNaiveBayes:
    """Vectorized Gaussian Naive Bayes classifier computed strictly in log-probability space."""

    def fit(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> "GaussianNaiveBayes":
        """Estimate per-class log-priors `(C,)`, means `(C, D)`, and variances `(C, D) + eps`."""
        raise NotImplementedError("Implement GaussianNaiveBayes.fit.")

    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized class log-probabilities `(N, C)` using `stable_log_softmax`."""
        raise NotImplementedError("Implement GaussianNaiveBayes.predict_log_proba.")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels `(N,)` matching the original class IDs in `y`."""
        raise NotImplementedError("Implement GaussianNaiveBayes.predict.")


def _check() -> None:
    import torch.nn.functional as F

    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    assert torch.allclose(stable_log_softmax(logits, -1), F.log_softmax(logits, -1), atol=1e-5)
    assert torch.isfinite(stable_log_softmax(logits.detach() * 800.0, -1)).all()

    targets = torch.tensor([[1, -100, 3, 0], [-100, -100, -100, -100]])
    loss = masked_cross_entropy(logits, targets)
    assert torch.allclose(loss, F.cross_entropy(logits.view(-1, 7), targets.view(-1), ignore_index=-100), atol=1e-5)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    x_km = torch.cat([torch.randn(25, 3) - 4.0, torch.randn(25, 3) + 4.0], dim=0)
    c, a = kmeans_pytorch(x_km, k=2, seed=42)
    assert c.shape == (2, 3) and len(set(a[:25].tolist())) == 1 and a[0] != a[25]

    y_nb = torch.tensor([0] * 25 + [1] * 25)
    gnb = GaussianNaiveBayes().fit(x_km, y_nb)
    preds = gnb.predict(x_km)
    assert (preds == y_nb).float().mean().item() > 0.95
    assert torch.allclose(gnb.predict_log_proba(x_km).exp().sum(dim=-1), torch.ones(50), atol=1e-5)
    print("Lab 1 checks passed!")


if __name__ == "__main__":
    _check()
