# Copyright (c) 2025 DeepSeek-V3 Authors
# SPDX-License-Identifier: MIT
"""
DeepSeek-V3 Model Architecture

This module defines the core transformer model architecture for DeepSeek-V3,
including Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE) layers.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """Configuration arguments for DeepSeek-V3 model."""
    vocab_size: int = 102400
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    n_kv_heads: int = 128
    # Multi-head Latent Attention (MLA) parameters
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # Mixture of Experts parameters
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    score_func: str = "softmax"
    route_scale: float = 1.0
    # RoPE parameters
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    # Misc
    max_seq_len: int = 4096
    dtype: str = "bf16"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Precompute rotary positional embedding frequencies.

    Args:
        dim: Head dimension for rotary embeddings.
        end: Maximum sequence length.
        theta: Base frequency for RoPE.
        dtype: Output tensor dtype.

    Returns:
        Complex tensor of shape (end, dim // 2).
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(dtype=dtype)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim).
        xk: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim).
        freqs_cis: Precomputed frequency tensor.

    Returns:
        Rotated query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]  # (seq_len, 1, head_dim // 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network used in dense layers."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
