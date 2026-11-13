"""Comprehensive pytest verification suite for PyTorch, Transformers & Post-Training Labs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from .ex4_buggy_pytorch_trainer import buggy_train_and_eval
from .solutions_pytorch import (
    GaussianNaiveBayes,
    ReferenceCausalGQAWithCache,
    ReferenceLoRALinear,
    apply_rope,
    dpo_loss,
    fixed_train_and_eval,
    grpo_loss,
    kmeans_pytorch,
    masked_cross_entropy,
    stable_log_softmax,
)


def test_stable_log_softmax_matches_torch_and_handles_extremes() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 19, requires_grad=True)
    out = stable_log_softmax(logits, dim=-1)
    expected = F.log_softmax(logits, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5)

    extreme = torch.tensor([[1000.0, 999.0, -1000.0], [-1000.0, -1000.0, -1000.0]], requires_grad=True)
    ext_out = stable_log_softmax(extreme, dim=-1)
    assert torch.isfinite(ext_out).all()
    ext_out.sum().backward()
    assert extreme.grad is not None and torch.isfinite(extreme.grad).all()


def test_masked_cross_entropy_ignores_pad_and_handles_all_ignored() -> None:
    torch.manual_seed(1)
    logits = torch.randn(3, 5, 13, requires_grad=True)
    targets = torch.tensor(
        [
            [2, -100, 11, 0, -100],
            [-100, -100, -100, -100, -100],
            [4, 5, 6, -100, 1],
        ]
    )
    loss = masked_cross_entropy(logits, targets, ignore_index=-100)
    ref = F.cross_entropy(logits.view(-1, 13), targets.view(-1), ignore_index=-100)
    assert torch.allclose(loss, ref, atol=1e-5)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[1] == 0.0), "Fully ignored row must receive zero gradient"

    logits2 = torch.randn(2, 3, 5, requires_grad=True)
    all_ignored = torch.full((2, 3), -100, dtype=torch.long)
    zero_loss = masked_cross_entropy(logits2, all_ignored, ignore_index=-100)
    assert float(zero_loss.item()) == 0.0
    zero_loss.backward()
    assert logits2.grad is not None and torch.all(logits2.grad == 0.0)


def test_kmeans_separates_clusters_and_repairs_empty() -> None:
    torch.manual_seed(2)
    c1 = torch.randn(30, 4) * 0.3 - 5.0
    c2 = torch.randn(30, 4) * 0.3 + 5.0
    x = torch.cat([c1, c2], dim=0)
    centroids, assign = kmeans_pytorch(x, k=2, num_iters=25, seed=9)
    assert centroids.shape == (2, 4)
    assert len(set(assign[:30].tolist())) == 1
    assert len(set(assign[30:].tolist())) == 1
    assert assign[0].item() != assign[30].item()

    # Duplicate coordinates force an initial empty cluster if duplicate points are sampled
    dup = torch.tensor([[0.0, 0.0]] * 10 + [[10.0, 10.0], [20.0, 20.0]])
    c_dup, a_dup = kmeans_pytorch(dup, k=3, num_iters=15, seed=0)
    assert c_dup.shape == (3, 2) and torch.isfinite(c_dup).all()
    assert len(set(a_dup.tolist())) == 3


def test_gaussian_naive_bayes_log_space_stability() -> None:
    torch.manual_seed(3)
    x0 = torch.randn(40, 8) - 2.0
    x1 = torch.randn(40, 8) + 2.0
    x = torch.cat([x0, x1], dim=0)
    y = torch.tensor([10] * 40 + [20] * 40)  # non-contiguous class labels
    clf = GaussianNaiveBayes().fit(x, y)
    log_p = clf.predict_log_proba(x)
    assert log_p.shape == (80, 2)
    assert torch.allclose(log_p.exp().sum(dim=-1), torch.ones(80), atol=1e-5)
    preds = clf.predict(x)
    assert set(preds.tolist()).issubset({10, 20})
    assert (preds == y).float().mean().item() > 0.97


def test_rope_norm_preservation_and_relative_dot_product() -> None:
    torch.manual_seed(4)
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 2, 1, 8)
    q_rot_3 = apply_rope(q, torch.tensor([3]))
    k_rot_1 = apply_rope(k, torch.tensor([1]))
    q_rot_10 = apply_rope(q, torch.tensor([10]))
    k_rot_8 = apply_rope(k, torch.tensor([8]))
    # Norm preserved
    assert torch.allclose(q.norm(dim=-1), q_rot_3.norm(dim=-1), atol=1e-5)
    # Relative distance (3 - 1 == 10 - 8 == 2) gives identical attention dot product!
    dot_a = (q_rot_3 * k_rot_1).sum(dim=-1)
    dot_b = (q_rot_10 * k_rot_8).sum(dim=-1)
    assert torch.allclose(dot_a, dot_b, atol=1e-5)


def test_causal_gqa_no_future_leakage_padding_safety_and_kv_cache_parity() -> None:
    torch.manual_seed(5)
    attn = ReferenceCausalGQAWithCache(d_model=32, num_q_heads=4, num_kv_heads=2)

    # 1. Zero future-token gradient leakage check
    x = torch.randn(2, 6, 32, requires_grad=True)
    out, _ = attn(x, use_cache=False)
    # Loss depends ONLY on positions 0, 1, 2 -> gradient wrt x[:, 3:, :] MUST be exact zero
    out[:, :3, :].sum().backward()
    assert x.grad is not None
    assert torch.all(x.grad[:, 3:, :] == 0.0), "Causal mask leaked future token gradients!"

    # 2. Left-padding (all-masked first token) produces zero NaNs in forward AND backward
    x_pad = torch.randn(2, 5, 32, requires_grad=True)
    pad_mask = torch.tensor(
        [[False, False, True, True, False], [True, True, True, False, False]],
        dtype=torch.bool,
    )
    out_pad, _ = attn(x_pad, key_padding_mask=pad_mask)
    assert torch.isfinite(out_pad).all()
    out_pad.pow(2).mean().backward()
    assert x_pad.grad is not None and torch.isfinite(x_pad.grad).all()

    # 3. KV-cache prefill + single-token decode parity
    x_seq = torch.randn(2, 7, 32)
    with torch.no_grad():
        full_out, _ = attn(x_seq, use_cache=False)
        pre_out, cache = attn(x_seq[:, :3, :], use_cache=True)
        chunks = [pre_out]
        for t in range(3, 7):
            step_out, cache = attn(x_seq[:, t : t + 1, :], kv_cache=cache, use_cache=True)
            chunks.append(step_out)
        cached_out = torch.cat(chunks, dim=1)
        assert cache is not None and cache.seq_len == 7
        assert torch.allclose(full_out, cached_out, atol=1e-5)


def test_lora_dpo_and_grpo_contracts() -> None:
    torch.manual_seed(6)
    base = nn.Linear(12, 6, bias=True)
    lora = ReferenceLoRALinear(base, r=3, alpha=6.0)
    x = torch.randn(8, 12)
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    lora(x).sum().backward()
    assert base.weight.grad is None and base.bias.grad is None
    assert lora.lora_B.grad is not None and lora.lora_B.grad.abs().sum() > 0

    with torch.no_grad():
        lora.lora_B.normal_(0.0, 0.3)
        merged = lora.merge_weights()
        assert torch.allclose(lora(x), merged(x), atol=1e-5)

    pi_w = torch.tensor([-1.2, -0.8], requires_grad=True)
    pi_l = torch.tensor([-2.4, -2.0], requires_grad=True)
    ref_w = torch.tensor([-1.5, -1.0])
    ref_l = torch.tensor([-2.1, -1.9])
    loss_dpo, rw, rl = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.25)
    loss_dpo.backward()
    assert (rw > rl).all() and not rw.requires_grad and pi_w.grad is not None

    # GRPO: test normal variance AND zero-variance group collapse
    pol = torch.randn(6, 4, requires_grad=True)
    old = pol.detach().clone()
    ref = pol.detach() - 0.05
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # second group of 3 has std=0!
    mask = torch.tensor([[1, 1, 1, 0]] * 6, dtype=torch.float32)
    loss_grpo = grpo_loss(pol, old, rewards, group_size=3, response_mask=mask, ref_logps=ref)
    loss_grpo.backward()
    assert torch.isfinite(loss_grpo) and pol.grad is not None and torch.isfinite(pol.grad).all()


def test_fixed_trainer_beats_buggy_trainer_and_prevents_leakage() -> None:
    torch.manual_seed(9)
    x_tr = torch.randn(80, 8, 5)
    m_tr = torch.zeros(80, 8, dtype=torch.bool)
    m_tr[:, :2] = True
    x_tr[~m_tr] = 100.0
    y_tr = (x_tr[:, 0, 0] + x_tr[:, 1, 0] > 0).long()

    x_va = torch.randn(80, 8, 5)
    m_va = torch.zeros(80, 8, dtype=torch.bool)
    m_va[:, :2] = True
    x_va[~m_va] = 100.0
    y_va = (x_va[:, 0, 0] + x_va[:, 1, 0] > 0).long()

    buggy = buggy_train_and_eval(x_tr, m_tr, y_tr, x_va, m_va, y_va, epochs=35, lr=0.05, seed=0)
    fixed = fixed_train_and_eval(x_tr, m_tr, y_tr, x_va, m_va, y_va, epochs=35, lr=0.05, seed=0)
    assert fixed["val_acc"] > buggy["val_acc"] + 0.15
    assert fixed["val_acc"] >= 0.90

    # Zero val->train leakage guarantee
    x_va_shifted = x_va + 5000.0
    fixed_shifted = fixed_train_and_eval(x_tr, m_tr, y_tr, x_va_shifted, m_va, y_va, epochs=35, lr=0.05, seed=0)
    assert abs(fixed["train_loss"] - fixed_shifted["train_loss"]) < 1e-6
