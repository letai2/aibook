"""دو تبدیل داخل هر بلوک: ارتباط بین موقعیت‌ها و پردازش هر موقعیت."""

from torch import nn

from .attention import CausalSelfAttention
from .config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        C = config.embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(C, 4 * C),
            nn.GELU(),
            nn.Linear(4 * C, C),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x, *, causal=True, residual=True, trace=None):
        if trace is not None:
            trace.update(input=x, attention={})
        update = self.attention(self.norm_1(x), causal=causal,
                                trace=trace["attention"] if trace is not None else None)
        x = x + update if residual else update
        update = self.feed_forward(self.norm_2(x))
        output = x + update if residual else update
        if trace is not None:
            trace.update(feed_forward=update, output=output)
        return output
