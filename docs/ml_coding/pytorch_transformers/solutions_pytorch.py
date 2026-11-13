"""Reference Implementations for PyTorch, Transformers & Post-Training Labs (Labs 1–4)."""

from __future__ import annotations

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ex2_attention_and_kvcache import KVCache
from .ex4_buggy_pytorch_trainer import TinyPooledClassifier


# ============================================================================
# Lab 1: Classical ML & Numerically Stable Losses
# ============================================================================


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_logits = logits.max(dim=dim, keepdim=True).values
    shifted = logits - max_logits
    log_denom = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    return shifted - log_denom


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid_mask = flat_targets != ignore_index
    valid_count = valid_mask.sum()
    if valid_count == 0:
        return flat_logits.sum() * 0.0

    log_probs = stable_log_softmax(flat_logits, dim=-1)
    safe_targets = flat_targets.masked_fill(~valid_mask, 0).long()
    picked = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    nll = -picked.masked_fill(~valid_mask, 0.0)
    return nll.sum() / valid_count


def kmeans_pytorch(
    x: torch.Tensor,
    k: int,
    num_iters: int = 20,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or k <= 0 or x.shape[0] < k or num_iters <= 0:
        raise ValueError("Expected x of shape (N, D) with N >= k > 0 and num_iters >= 1")

    gen = torch.Generator(device=x.device).manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=gen, device=x.device)
    centroids = x[perm[:k]].clone()
    prev_assignments: Optional[torch.Tensor] = None

    for _ in range(num_iters):
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x @ c^T, clamped for float precision
        x_sq = (x * x).sum(dim=-1, keepdim=True)  # (N, 1)
        c_sq = (centroids * centroids).sum(dim=-1).unsqueeze(0)  # (1, k)
        dists = (x_sq + c_sq - 2.0 * (x @ centroids.T)).clamp_min(0.0)  # (N, k)
        assignments = dists.argmin(dim=-1)

        if prev_assignments is not None and torch.equal(assignments, prev_assignments):
            break
        prev_assignments = assignments

        min_dists = dists.gather(1, assignments.unsqueeze(1)).squeeze(1)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(assignments, minlength=k)
        new_centroids.index_add_(0, assignments, x)

        non_empty = counts > 0
        new_centroids[non_empty] /= counts[non_empty].unsqueeze(-1).to(x.dtype)

        if (~non_empty).any():
            # Deterministic empty-cluster repair: pick farthest points one by one
            candidate_dists = min_dists.clone()
            for cluster_id in torch.where(~non_empty)[0].tolist():
                farthest_idx = int(candidate_dists.argmax().item())
                new_centroids[cluster_id] = x[farthest_idx]
                candidate_dists[farthest_idx] = -1.0

        centroids = new_centroids

    # Final assignment sync against returned centroids
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    c_sq = (centroids * centroids).sum(dim=-1).unsqueeze(0)
    final_dists = (x_sq + c_sq - 2.0 * (x @ centroids.T)).clamp_min(0.0)
    return centroids, final_dists.argmin(dim=-1)


class GaussianNaiveBayes:
    def __init__(self) -> None:
        self.classes_: Optional[torch.Tensor] = None
        self.log_priors_: Optional[torch.Tensor] = None
        self.means_: Optional[torch.Tensor] = None
        self.vars_: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> "GaussianNaiveBayes":
        classes = torch.unique(y, sorted=True)
        c_count = classes.numel()
        d = x.shape[1]
        means = torch.zeros((c_count, d), dtype=x.dtype, device=x.device)
        vars_ = torch.zeros((c_count, d), dtype=x.dtype, device=x.device)
        log_priors = torch.zeros(c_count, dtype=x.dtype, device=x.device)

        n_total = float(x.shape[0])
        for idx, cls in enumerate(classes):
            cls_x = x[y == cls]
            log_priors[idx] = math.log(cls_x.shape[0] / n_total)
            means[idx] = cls_x.mean(dim=0)
            vars_[idx] = cls_x.var(dim=0, unbiased=False) + eps

        self.classes_ = classes
        self.log_priors_ = log_priors
        self.means_ = means
        self.vars_ = vars_
        return self

    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.classes_ is None or self.means_ is None or self.vars_ is None or self.log_priors_ is None:
            raise RuntimeError("Call fit() before predict_log_proba().")
        # Broadcast (N, 1, D) against (1, C, D)
        diff = x.unsqueeze(1) - self.means_.unsqueeze(0)
        log_likelihood = -0.5 * (
            math.log(2.0 * math.pi) + torch.log(self.vars_.unsqueeze(0)) + (diff * diff) / self.vars_.unsqueeze(0)
        ).sum(dim=-1)
        joint_log_prob = self.log_priors_.unsqueeze(0) + log_likelihood  # (N, C)
        return stable_log_softmax(joint_log_prob, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        assert self.classes_ is not None
        best = self.predict_log_proba(x).argmax(dim=-1)
        return self.classes_[best]


# ============================================================================
# Lab 2: RoPE & Causal Grouped-Query Attention with KV Cache
# ============================================================================


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x: (B, H, T, D_h), positions: (T,) or (B, T)."""
    bsz, _, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / float(half))
    )  # (D_h/2,)

    if positions.ndim == 1:
        pos = positions.unsqueeze(0).expand(bsz, seq_len).to(torch.float32)
    elif positions.ndim == 2:
        pos = positions.to(torch.float32)
    else:
        raise ValueError("positions must have shape (T,) or (B, T)")

    angles = pos.unsqueeze(-1) * inv_freq.view(1, 1, half)  # (B, T, D_h/2)
    cos = torch.cos(angles).unsqueeze(1).to(x.dtype)  # (B, 1, T, D_h/2)
    sin = torch.sin(angles).unsqueeze(1).to(x.dtype)  # (B, 1, T, D_h/2)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class ReferenceCausalGQAWithCache(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        if d_model % num_q_heads != 0 or num_q_heads % num_kv_heads != 0:
            raise ValueError("Invalid head configuration")
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        self.group_size = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        bsz, t_q, _ = x.shape
        past_len = kv_cache.seq_len if kv_cache is not None else 0

        q = self.q_proj(x).view(bsz, t_q, self.num_q_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(bsz, t_q, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(bsz, t_q, self.num_kv_heads, self.head_dim).transpose(1, 2)

        pos = torch.arange(past_len, past_len + t_q, device=x.device)
        q = apply_rope(q, pos)
        k_new = apply_rope(k_new, pos)

        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k_new], dim=2)
            v_all = torch.cat([kv_cache.v, v_new], dim=2)
        else:
            k_all, v_all = k_new, v_new

        next_cache = KVCache(k=k_all, v=v_all) if use_cache else None
        t_k = k_all.shape[2]

        # Expand KV heads to match Q heads without copying memory where possible
        if self.group_size > 1:
            k_exp = k_all.repeat_interleave(self.group_size, dim=1)
            v_exp = v_all.repeat_interleave(self.group_size, dim=1)
        else:
            k_exp, v_exp = k_all, v_all

        scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H_q, T_q, T_k)

        q_idx = torch.arange(past_len, past_len + t_q, device=x.device).unsqueeze(1)  # (T_q, 1)
        k_idx = torch.arange(0, t_k, device=x.device).unsqueeze(0)  # (1, T_k)
        allowed = k_idx <= q_idx  # (T_q, T_k)
        allowed = allowed.view(1, 1, t_q, t_k)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (bsz, t_k):
                raise ValueError(f"Expected key_padding_mask of shape {(bsz, t_k)}, got {tuple(key_padding_mask.shape)}")
            allowed = allowed & key_padding_mask.view(bsz, 1, 1, t_k)

        # Prevent all-masked rows from producing softmax([-inf, ...]) -> NaN (and NaN backward!)
        has_any_valid = allowed.any(dim=-1, keepdim=True)  # (B, 1, T_q, 1)
        safe_scores = scores.masked_fill(~allowed, float("-inf"))
        safe_scores = torch.where(has_any_valid, safe_scores, torch.zeros_like(safe_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = weights * has_any_valid.to(weights.dtype)

        if key_padding_mask is not None:
            query_valid = key_padding_mask[:, past_len : past_len + t_q].view(bsz, 1, t_q, 1)
            weights = weights * query_valid.to(weights.dtype)

        attn_out = torch.matmul(weights, v_exp)  # (B, H_q, T_q, D_h)
        merged = attn_out.transpose(1, 2).contiguous().view(bsz, t_q, self.d_model)
        return self.o_proj(merged), next_cache


# ============================================================================
# Lab 3: LoRA, DPO & GRPO
# ============================================================================


class ReferenceLoRALinear(nn.Module):
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

        dev, dt = base_layer.weight.device, base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, r), device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        has_bias = self.base_layer.bias is not None
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=has_bias,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            merged.weight.copy_(self.base_layer.weight + delta_w)
            if has_bias and self.base_layer.bias is not None:
                merged.bias.copy_(self.base_layer.bias)
        return merged


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if beta <= 0:
        raise ValueError("beta must be > 0")
    ref_w = ref_chosen_logps.detach()
    ref_l = ref_rejected_logps.detach()
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_w - ref_l
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    chosen_rewards = (beta * (policy_chosen_logps - ref_w)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - ref_l)).detach()
    return loss, chosen_rewards, rejected_rewards


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
    bsz, _ = policy_logps.shape
    if group_size <= 0 or bsz % group_size != 0:
        raise ValueError("Batch size must be divisible by positive group_size")

    grouped = rewards.detach().float().view(bsz // group_size, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=False, keepdim=True)
    advantages = ((grouped - mean) / (std + eps)).view(bsz, 1).to(policy_logps.dtype)

    old_detached = old_logps.detach()
    ratio = torch.exp(policy_logps - old_detached)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token_loss = -torch.minimum(unclipped, clipped)

    if ref_logps is not None and beta_kl > 0.0:
        ref_detached = ref_logps.detach()
        log_diff = ref_detached - policy_logps
        kl_k3 = torch.exp(log_diff) - log_diff - 1.0
        per_token_loss = per_token_loss + beta_kl * kl_k3

    mask = response_mask.to(policy_logps.dtype)
    denom = mask.sum()
    if denom == 0:
        return policy_logps.sum() * 0.0
    return (per_token_loss * mask).sum() / denom


# ============================================================================
# Lab 4: Repaired PyTorch Sequence Training Pipeline
# ============================================================================


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
    torch.manual_seed(seed)

    # Fix 1: Fit feature normalizer strictly on valid tokens of x_train ONLY
    valid_train_tokens = x_train[mask_train]  # (N_valid, D)
    mean = valid_train_tokens.mean(dim=0).view(1, 1, -1)
    std = valid_train_tokens.std(dim=0, unbiased=False).view(1, 1, -1).clamp_min(1e-6)
    x_tr = (x_train - mean) / std
    x_va = (x_val - mean) / std

    model = TinyPooledClassifier(x_train.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        opt.zero_grad()  # Fix 3: Clear stale gradients each step
        logits = model(x_tr, mask_train, buggy_masking=False)  # Fix 2: -inf pad mask
        loss = F.cross_entropy(logits, y_train)  # Fix 5: Numerically stable log-sum-exp CE
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    # Fix 4: Deterministic evaluation with dropout disabled & autograd off
    model.eval()
    with torch.no_grad():
        val_logits = model(x_va, mask_val, buggy_masking=False)
        val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()

    return {"train_loss": last_loss, "val_acc": val_acc}
