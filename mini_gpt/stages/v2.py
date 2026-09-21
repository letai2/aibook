"""نسخهٔ دو فقط برای مشاهده است: توجه بدون پوشش برای آموزش علّی معتبر نیست."""

import math
import torch
from torch import nn


class SingleHead(nn.Module):
    def __init__(self, vocab_size, channels=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.head = nn.Linear(channels, vocab_size)
        self.causal = False

    def forward(self, ids):
        x = self.embedding(ids)
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
        if self.causal:
            T = ids.shape[1]
            allowed = torch.ones(T, T, device=ids.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~allowed, float("-inf"))
        self.last_weights = torch.softmax(scores, dim=-1)
        return self.head(self.last_weights @ v)
