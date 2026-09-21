"""نسخهٔ یک: هر نویسه یک بردار دارد؛ هنوز ارتباطی با گذشته ندارد."""

from torch import nn


class TokenOnly(nn.Module):
    def __init__(self, vocab_size, channels=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        self.head = nn.Linear(channels, vocab_size)

    def forward(self, ids):
        return self.head(self.embedding(ids))
