"""Lab 3 (45 min): Post-Training Primitives — LoRA, DPO & GRPO in PyTorch.

Focused, realistic implementations (~15–25 lines per component) commonly tested in
Applied AI / Post-Training / Research Engineer interviews:
1. `LoRALinear`: Frozen base linear + rank-`r` adapter (`lora_B == 0` init + `merge_weights()`).
2. `dpo_loss`: Direct Preference Optimization loss + detached implicit reward margins.
3. `grpo_loss`: Group Relative Policy Optimization with group-variance collapse guard + k3 KL.
"""

from __future__ import annotations

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around a frozen `nn.Linear` layer."""

    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("Rank r must be > 0")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # TODO: Initialize `self.lora_A` of shape (r, in_features) with kaiming_uniform_
        #       and `self.lora_B` of shape (out_features, r) with zeros so initial output == base_layer(x).
        raise NotImplementedError("Initialize LoRA parameters.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute `base_layer(x) + scaling * (x @ lora_A.T @ lora_B.T)`."""
        raise NotImplementedError("Implement LoRALinear.forward.")

    def merge_weights(self) -> nn.Linear:
        """Return a new standalone `nn.Linear` with `W_merged = W_base + scaling * (B @ A)`."""
        raise NotImplementedError("Implement LoRALinear.merge_weights.")


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss and detached implicit rewards `(loss, chosen_rewards, rejected_rewards)`.

    Args:
        policy_chosen_logps, policy_rejected_logps: `(B,)` sequence log-probs under trainable policy
        ref_chosen_logps, ref_rejected_logps: `(B,)` sequence log-probs under frozen reference model
        beta: KL temperature parameter (`beta > 0`)

    Returns:
        loss: scalar `-F.logsigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l))).mean()`
        chosen_rewards: detached `(B,)` tensor `beta * (pi_w - ref_w)`
        rejected_rewards: detached `(B,)` tensor `beta * (pi_l - ref_l)`
    """
    raise NotImplementedError("Implement dpo_loss.")


def grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    response_mask: torch.Tensor,
    ref_logps: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute token-masked GRPO clipped surrogate objective with optional Schulman k3 KL penalty.

    Shapes:
        policy_logps, old_logps, ref_logps: `(B, T)` token log-probs (`B` is a multiple of `group_size`)
        rewards: `(B,)` scalar verifier reward per sampled completion
        response_mask: `(B, T)` float/bool mask (`1` for generated response tokens, `0` for prompt/pad)

    Steps:
        1. Reshape `rewards` to `(B // group_size, group_size)` and compute detached group-relative
           advantages: `adv = (r - mean) / (std_unbiased_false + eps)`.
           Note: Use ` unbiased=False` (population std) + `eps` so constant-reward groups give `adv == 0`
           instead of `NaN`!
        2. Detach `old_logps`, compute token importance ratio `exp(policy_logps - old_logps)`, and
           PPO-clipped surrogate `-min(ratio * adv, clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)`.
        3. If `ref_logps is not None and beta_kl > 0`, add `beta_kl * (exp(ref - pi) - (ref - pi) - 1)`
           with `ref_logps` detached.
        4. Average over valid tokens using `response_mask` (returning differentiable 0.0 if mask sum is 0).
    """
    raise NotImplementedError("Implement grpo_loss.")


def _check() -> None:
    torch.manual_seed(3)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6), "Initial LoRA delta must be exact zero!"
    loss_lora = lora(x).pow(2).mean()
    loss_lora.backward()
    assert base.weight.grad is None and lora.lora_B.grad is not None
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.2)
        merged = lora.merge_weights()
        assert torch.allclose(lora(x), merged(x), atol=1e-5)

    pi_w = torch.tensor([-2.0, -1.5], requires_grad=True)
    pi_l = torch.tensor([-3.0, -2.5], requires_grad=True)
    ref_w = torch.tensor([-2.2, -1.6])
    ref_l = torch.tensor([-2.9, -2.4])
    dl, rw, rl = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.2)
    dl.backward()
    assert not rw.requires_grad and (rw > rl).all() and pi_w.grad is not None

    pol = torch.randn(4, 5, requires_grad=True)
    old = pol.detach().clone()
    ref = pol.detach() - 0.1
    flat_rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])  # zero-variance trap!
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 4, dtype=torch.float32)
    gl = grpo_loss(pol, old, flat_rewards, group_size=4, response_mask=mask, ref_logps=ref)
    gl.backward()
    assert torch.isfinite(gl) and pol.grad is not None and torch.isfinite(pol.grad).all()
    print("Lab 3 checks passed!")


if __name__ == "__main__":
    _check()
