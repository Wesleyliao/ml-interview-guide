"""Lab 4 (35–45 min): Debug & Repair a Broken PyTorch Sequence Training Pipeline.

Models the live "debug this broken ML pipeline" round reported at Meta (E6 MLE), OpenAI,
Anthropic, Figma, and Decagon.

`buggy_train_and_eval` below runs WITHOUT throwing an exception, yet contains **5 classic bugs**:
  Bug 1 (Data leakage): Standardizes features using mean/std computed across `train + val` combined.
  Bug 2 (Mask broadcast / polarity bug): Masks pad tokens by multiplying scores by `0` instead of
        setting padded key positions to `-inf` before softmax (so padded keys still steal probability mass!).
  Bug 3 (Gradient accumulation bug): Never calls `optimizer.zero_grad()`, accumulating stale gradients!
  Bug 4 (Mode toggle bug): Leaves `model` in `.train()` mode (with `Dropout(0.5)` active) and tracks
        gradients during validation.
  Bug 5 (Numerical stability bug): Computes `torch.log(torch.softmax(logits, dim=-1))` which yields
        `NaN` when logits are large, instead of `F.log_softmax` / `F.cross_entropy`.

Your task: Implement `fixed_train_and_eval` below fixing all 5 bugs while keeping the same signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyPooledClassifier(nn.Module):
    """Single-head attention-pooled binary sequence classifier."""

    def __init__(self, in_dim: int, hidden_dim: int = 24, dropout: float = 0.4) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, buggy_masking: bool = False) -> torch.Tensor:
        """x: (B, T, D), mask: (B, T) bool (`True` = valid token, `False` = pad). Assumes >=1 True per row."""
        h = torch.tanh(self.proj(x))  # (B, T, H)
        scores = (h * self.query).sum(dim=-1) / (h.shape[-1] ** 0.5)  # (B, T)
        if buggy_masking:
            # BUG 2: setting score=0 gives exp(0)=1 weight to PAD tokens!
            weights = torch.softmax(scores * mask.float(), dim=-1)
        else:
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
        pooled = self.drop((h * weights.unsqueeze(-1)).sum(dim=1))
        return self.head(pooled)


def buggy_train_and_eval(
    x_train: torch.Tensor,
    mask_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    mask_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Reference broken trainer — DO NOT EDIT (used by tests to demonstrate the 5 failure modes)."""
    torch.manual_seed(seed)
    # BUG 1: Fits scaler on train + val combined, AND includes zero-padded positions in mean/std!
    all_x = torch.cat([x_train, x_val], dim=0)
    mean = all_x.mean(dim=(0, 1), keepdim=True)
    std = all_x.std(dim=(0, 1), keepdim=True) + 1e-6
    x_tr = (x_train - mean) / std
    x_va = (x_val - mean) / std

    model = TinyPooledClassifier(x_train.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        # BUG 3: Missing opt.zero_grad()!
        logits = model(x_tr, mask_train, buggy_masking=True)
        # BUG 5: Unstable log(softmax(logits)) -> -inf / NaN on large logit margins
        log_probs = torch.log(torch.softmax(logits, dim=-1))
        loss = -log_probs[torch.arange(len(y_train)), y_train].mean()
        loss.backward()
        opt.step()

    # BUG 4: Never switches to model.eval() or torch.no_grad() -> stochastic dropout corrupts eval!
    val_logits = model(x_va, mask_val, buggy_masking=True)
    val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()
    return {"train_loss": float(loss.item()), "val_acc": val_acc}


def fixed_train_and_eval(
    x_train: torch.Tensor,
    mask_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    mask_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Repair all 5 bugs:
    1. Compute feature `mean` and `std` ONLY over valid (`mask_train == True`) tokens of `x_train`!
    2. Use proper `-inf` attention masking (`buggy_masking=False`).
    3. Call `model.train()` and `opt.zero_grad()` each step.
    4. Use numerically stable `F.cross_entropy(logits, y_train)`.
    5. Evaluate under `model.eval()` and `torch.no_grad()`, returning deterministic `val_acc`
       (and return `{"train_loss": ..., "val_acc": ..., "val_acc_repeat": ...}` or verify determinism).
    """
    raise NotImplementedError("Implement fixed_train_and_eval.")


def _check() -> None:
    torch.manual_seed(11)
    x_tr = torch.randn(64, 8, 6)
    m_tr = torch.zeros(64, 8, dtype=torch.bool)
    m_tr[:, :2] = True  # Only first 2 tokens are real; 6 tokens are pure noise padding!
    x_tr[~m_tr] = 50.0  # Huge pad sentinel values that wreck unmasked pooling & unmasked normalization!
    y_tr = (x_tr[:, 0, 0] + x_tr[:, 1, 0] > 0).long()

    x_va = torch.randn(64, 8, 6)
    m_va = torch.zeros(64, 8, dtype=torch.bool)
    m_va[:, :2] = True
    x_va[~m_va] = 50.0
    y_va = (x_va[:, 0, 0] + x_va[:, 1, 0] > 0).long()

    res1 = fixed_train_and_eval(x_tr, m_tr, y_tr, x_va, m_va, y_va, epochs=40, lr=0.05, seed=0)
    # Changing x_val outliers must NEVER alter how x_train is normalized (zero train/val leakage check!)
    x_va_corrupted = x_va.clone()
    x_va_corrupted[:, 0, :] += 10000.0
    res_leak_check = fixed_train_and_eval(x_tr, m_tr, y_tr, x_va_corrupted, m_va, y_va, epochs=40, lr=0.05, seed=0)
    assert abs(res1["train_loss"] - res_leak_check["train_loss"]) < 1e-6, "Validation data leaked into train loss!"
    assert res1["val_acc"] >= 0.90, f"Expected >= 0.90 val_acc, got {res1['val_acc']:.3f}"
    print("Lab 4 checks passed!")


if __name__ == "__main__":
    _check()
