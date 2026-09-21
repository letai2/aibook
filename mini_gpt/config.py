from dataclasses import dataclass


@dataclass
class ModelConfig:
    """All dimensions needed to build MiniGPT.

    The deliberately small defaults run on CPU.  For multi-head attention,
    embedding_dim must be divisible by num_heads because D = C / H.
    """

    vocab_size: int
    context_length: int = 64
    embedding_dim: int = 96
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "embedding_dim", "num_heads", "num_layers"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.context_length < 1:
            raise ValueError("context_length must be positive")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
