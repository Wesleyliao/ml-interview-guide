"""Lab 2 (50–60 min): Causal + Padding Grouped-Query Attention, RoPE & KV Cache.

Directly targets the classic OpenAI / Anthropic PyTorch Transformer assessment:
- Phase 1: Implement RoPE (`apply_rope`) with explicit token position indices `(B, T)` or `(T,)`.
- Phase 2: Implement `CausalGQAWithCache` supporting `num_q_heads >= num_kv_heads`, causal masking,
           boolean `key_padding_mask`, AND safe handling when a padded query has zero valid keys
           (no `NaN` in forward OR backward!).
- Phase 3: Support incremental autoregressive `KVCache` (`prefill` + token-by-token `decode`) with
           exact logit/hidden-state parity (`atol=1e-5`) against full-sequence causal forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn


@dataclass
class KVCache:
    """Stores accumulated key/value tensors of shape `(B, H_kv, T_cached, D_h)`."""

    k: torch.Tensor
    v: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.k.shape[2])


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply Rotary Positional Embeddings to even/odd pairs `(x[..., 0::2], x[..., 1::2])`.

    Shapes:
        x: `(B, H, T, D_h)` where `D_h` is even
        positions: `(T,)` or `(B, T)` integer position indices (crucial during cached decode!)
        returns: `(B, H, T, D_h)` in the same dtype as `x`
    """
    raise NotImplementedError("Implement apply_rope.")


class CausalGQAWithCache(nn.Module):
    """Grouped-Query Causal Self-Attention with RoPE, Key-Padding Mask, and Optional KV Cache."""

    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        if d_model % num_q_heads != 0:
            raise ValueError("d_model must be divisible by num_q_heads")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
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
        """Run causal GQA forward pass.

        Args:
            x: `(B, T_q, d_model)` input hidden states.
            key_padding_mask: Optional `(B, T_total)` bool tensor (`True` = valid token, `False` = pad),
                              where `T_total = (kv_cache.seq_len if kv_cache else 0) + T_q`.
            kv_cache: Optional previous `KVCache` for incremental decoding.
            use_cache: If True, return updated `KVCache`; otherwise return `None`.

        Key invariants:
            1. Query at step `i` (0 <= i < T_q) sits at absolute position `past_len + i` and may ONLY
               attend to valid keys at absolute positions `j <= past_len + i`.
            2. Apply `apply_rope` to `q` (at positions `past_len .. past_len + T_q - 1`) and to the NEW
               `k` slice BEFORE concatenating onto `kv_cache` (so cached keys are never rotated twice!).
            3. If a query row has ZERO unmasked keys (e.g., leading pad token), its attention output must
               be `0.0` and `loss.backward()` must NOT produce `NaN` gradients!
        """
        raise NotImplementedError("Implement CausalGQAWithCache.forward.")


def _check() -> None:
    torch.manual_seed(7)
    attn = CausalGQAWithCache(d_model=32, num_q_heads=4, num_kv_heads=2)
    x = torch.randn(2, 6, 32, requires_grad=True)
    pad_mask = torch.tensor(
        [[False, True, True, True, True, False], [True, True, True, True, True, True]],
        dtype=torch.bool,
    )
    out_full, _ = attn(x, key_padding_mask=pad_mask, use_cache=False)
    assert out_full.shape == (2, 6, 32) and torch.isfinite(out_full).all()
    out_full.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Prefill (first 3 tokens) + step-by-step decode (last 3 tokens) parity check on unpadded sequence
    x_clean = torch.randn(2, 6, 32)
    with torch.no_grad():
        expected, _ = attn(x_clean, use_cache=False)
        prefill_out, cache = attn(x_clean[:, :3, :], use_cache=True)
        decoded = [prefill_out]
        for step in range(3, 6):
            step_out, cache = attn(x_clean[:, step : step + 1, :], kv_cache=cache, use_cache=True)
            decoded.append(step_out)
        actual = torch.cat(decoded, dim=1)
        assert torch.allclose(expected, actual, atol=1e-5), f"Max diff: {(expected - actual).abs().max()}"
    print("Lab 2 checks passed!")


if __name__ == "__main__":
    _check()
