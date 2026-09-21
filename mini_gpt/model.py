"""مدل زبانی کوچک: شناسه → نمایش → بلوک‌ها → امتیاز نشانهٔ بعدی."""

import torch
from torch import nn
from torch.nn import functional as F

from .attention import CausalSelfAttention
from .config import ModelConfig
from .sampling import choose_token, filter_logits
from .transformer import FeedForward, TransformerBlock

# سازگاری با نام استفاده‌شده در ویرایش نخست
_filter_logits = filter_logits


class MiniGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        C = config.embedding_dim
        self.token_embedding = nn.Embedding(config.vocab_size, C)
        self.position_embedding = nn.Embedding(config.context_length, C)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(C)
        self.language_model_head = nn.Linear(C, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _validate_ids(self, ids):
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("token_ids must be torch.long with shape (B,T)")
        if ids.shape[0] < 1 or not 1 <= ids.shape[1] <= self.config.context_length:
            raise ValueError("B must be positive and 1 <= T <= context_length")
        if ids.device != self.token_embedding.weight.device:
            raise ValueError("model and token_ids must be on the same device")
        if ids.min().item() < 0 or ids.max().item() >= self.config.vocab_size:
            raise ValueError("token ID is outside the vocabulary")

    def forward(self, token_ids, targets=None, *, causal=True, use_positions=True, residual=True, trace=None):
        self._validate_ids(token_ids)
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids)
        token_vectors = x
        position_vectors = self.position_embedding(positions) if use_positions else torch.zeros_like(x[0])
        if use_positions:
            x = x + position_vectors
        if trace is not None:
            trace.update(token_embedding=token_vectors, position_embedding=position_vectors,
                         combined_embedding=x, layers=[])
        x = self.dropout(x)
        for block in self.blocks:
            block_trace = {} if trace is not None else None
            x = block(x, causal=causal, residual=residual, trace=block_trace)
            if trace is not None:
                trace["layers"].append(block_trace)
        logits = self.language_model_head(self.final_norm(x))
        if trace is not None:
            trace["logits"] = logits
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must be torch.long with shape (B,T), matching input")
            if targets.device != token_ids.device:
                raise ValueError("targets and inputs must be on the same device")
            if targets.min().item() < 0 or targets.max().item() >= self.config.vocab_size:
                raise ValueError("target ID is outside the vocabulary")
            loss = F.cross_entropy(logits.reshape(B * T, self.config.vocab_size), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, greedy=False):
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if token_ids.ndim != 2 or token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("prompt must have shape (B,T), with B,T > 0")
        # اعتبارسنجی کل متن اولیه، حتی قسمت بریده‌شده
        if token_ids.dtype != torch.long or token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("prompt must contain valid long token IDs")
        if token_ids.device != self.token_embedding.weight.device:
            raise ValueError("model and prompt must be on the same device")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = token_ids[:, -self.config.context_length:]
                logits, _ = self(context)
                next_id = choose_token(logits[:, -1, :], temperature, top_k, top_p, greedy)
                token_ids = torch.cat((token_ids, next_id), dim=1)
        finally:
            self.train(was_training)
        return token_ids
