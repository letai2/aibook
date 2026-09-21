from __future__ import annotations

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    """Turn token IDs into shifted input/target windows.

    For [A, B, C, D, E] and context_length=4:
    input  = [A, B, C, D]
    target = [B, C, D, E]
    """

    def __init__(self, token_ids: list[int], context_length: int) -> None:
        if type(context_length) is not int or context_length < 1:
            raise ValueError("context_length must be a positive integer")
        if any(type(token) is not int or token < 0 for token in token_ids):
            raise ValueError("token_ids must contain nonnegative integers; do not silently round floats")
        if len(token_ids) <= context_length:
            raise ValueError("text must contain more tokens than context_length")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("window index is outside the dataset")
        chunk = self.tokens[index : index + self.context_length + 1]
        return chunk[:-1], chunk[1:]


def split_tokens(token_ids: list[int], train_fraction: float = 0.9) -> tuple[list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split = int(len(token_ids) * train_fraction)
    return token_ids[:split], token_ids[split:]
