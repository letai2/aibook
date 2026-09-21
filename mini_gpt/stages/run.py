"""اجرای کوچک مرحله‌های یک تا شش: مشاهدهٔ شکل و حساسیت به آینده."""

import argparse
import torch

from mini_gpt.config import ModelConfig
from .v1 import TokenOnly
from .v2 import SingleHead
from .v3 import CausalSingleHead
from .v4 import MultiHeadModel
from .v5 import OneBlockModel
from .v6 import build


def build_stage(stage):
    config = ModelConfig(12, context_length=8, embedding_dim=16, num_heads=2,
                         num_layers=1, dropout=0.0)
    factories = {
        1: lambda: TokenOnly(12),
        2: lambda: SingleHead(12),
        3: lambda: CausalSingleHead(12),
        4: lambda: MultiHeadModel(config),
        5: lambda: OneBlockModel(config),
        6: lambda: build(12),
    }
    return factories[stage]()


def logits_of(model, ids):
    result = model(ids)
    return result[0] if isinstance(result, tuple) else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    model = build_stage(args.stage).eval()
    original = torch.tensor([[1, 2, 3, 4, 5]])
    changed = original.clone()
    changed[:, 3:] = torch.tensor([9, 10])
    with torch.no_grad():
        a, b = logits_of(model, original), logits_of(model, changed)
    print("logits shape:", tuple(a.shape))
    print("earlier-position change:", (a[:, :3] - b[:, :3]).abs().max().item())
    print("parameters:", sum(p.numel() for p in model.parameters()))
    if args.stage == 2:
        print("v2 sees future tokens; it is a counterexample, not a valid causal language model.")


if __name__ == "__main__":
    main()
