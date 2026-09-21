"""A small, readable GPT-style language model used by the Persian textbook."""

from .config import ModelConfig
from .tokenizer import CharacterTokenizer

__all__ = ["CharacterTokenizer", "MiniGPT", "ModelConfig"]


def __getattr__(name):
    # مراحل صفر و tokenizer بدون نصب PyTorch نیز قابل استفاده‌اند.
    if name == "MiniGPT":
        from .model import MiniGPT
        return MiniGPT
    raise AttributeError(name)
