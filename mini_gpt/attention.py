"""توجه: جست‌وجوی عددی با پرسش، کلید و مقدار."""

import math

import torch
from torch import nn

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """ورودی/خروجی (B,T,C)، وزن قابل مشاهده (B,H,T,T)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.channels = config.embedding_dim
        self.context_length = config.context_length
        self.qkv = nn.Linear(self.channels, 3 * self.channels)
        self.output = nn.Linear(self.channels, self.channels)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        allowed = torch.ones(config.context_length, config.context_length, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", allowed[None, None, :, :])

    def forward(self, x, return_weights=False, *, causal=True, trace=None):
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"expected (B,T,{self.channels}); got {tuple(x.shape)}")
        B, T, C = x.shape
        if not 1 <= T <= self.context_length:
            raise ValueError("T must be between 1 and context_length")
        H, D = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, T, H, D).transpose(1, 2)
        k = k.reshape(B, T, H, D).transpose(1, 2)
        v = v.reshape(B, T, H, D).transpose(1, 2)
        raw_scores = q @ k.transpose(-2, -1)
        scores = raw_scores / math.sqrt(D)
        scaled_scores = scores
        if causal:
            scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # وزن‌های قابل تفسیر پیش از حذف تصادفی نگهداری می‌شوند.
        attended = self.attention_dropout(weights) @ v
        if trace is not None:
            # همان عددهای مسیر اصلی، نه یک پیاده‌سازی دوم برای نمایش.
            trace.update(q=q, k=k, v=v, raw_scores=raw_scores, scaled_scores=scaled_scores,
                         mask=self.causal_mask[:, :, :T, :T] if causal else torch.ones_like(scores, dtype=torch.bool),
                         masked_scores=scores, weights=weights, weighted_values=attended)
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        result = self.output_dropout(self.output(attended))
        return (result, weights) if return_weights else result
