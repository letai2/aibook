"""نسخهٔ چهار: توجه چندسر همراه با نمایش موقعیت."""

import torch
from torch import nn
from mini_gpt.attention import CausalSelfAttention


class MultiHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positions = nn.Embedding(config.context_length, config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.positions(positions)
        return self.head(self.attention(x))
