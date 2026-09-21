"""نسخهٔ شش: چند بلوک و نرمال‌سازی نهایی؛ همان مدل نهایی کتاب."""

from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT


def build(vocab_size):
    return MiniGPT(ModelConfig(vocab_size, context_length=8, embedding_dim=16,
                              num_heads=2, num_layers=2, dropout=0.0))
