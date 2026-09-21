import tempfile
from pathlib import Path

import torch

from mini_gpt.config import ModelConfig
from mini_gpt.dataset import NextTokenDataset
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharacterTokenizer


def run() -> None:
    text = "سلام دنیا! این یک متن کوچک برای آزمایش مدل است. " * 4
    tokenizer = CharacterTokenizer.from_text(text)
    assert tokenizer.decode(tokenizer.encode("سلام")) == "سلام"

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        tokenizer.save(path)
        assert CharacterTokenizer.load(path).decode(tokenizer.encode("دنیا")) == "دنیا"

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    dataset = NextTokenDataset(tokenizer.encode(text), config.context_length)
    inputs, targets = dataset[0]
    model = MiniGPT(config)
    logits, loss = model(inputs.unsqueeze(0), targets.unsqueeze(0))
    assert logits.shape == (1, config.context_length, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None

    generated = model.generate(inputs[:2].unsqueeze(0), max_new_tokens=3, top_k=1)
    assert generated.shape == (1, 5)
    print("smoke test passed")


if __name__ == "__main__":
    run()
