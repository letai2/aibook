"""فایل ذخیرهٔ مستقل شامل وزن‌ها، واژگان و وضعیت ادامهٔ آموزش."""

from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig
from .model import MiniGPT
from .tokenizer import CharacterTokenizer


def save_checkpoint(path, model, tokenizer, optimizer, step, metadata, batch_generator, best_loss):
    payload = {
        "format_version": 2,
        "model": model.state_dict(),
        "config": asdict(model.config),
        "tokens": tokenizer.id_to_token,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metadata": metadata,
        "best_loss": best_loss,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng_batches": batch_generator.get_state(),
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # بارگذاری فایل فقط از منبع مورد اعتماد؛ اشیای Python دلخواه لازم نداریم.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 2:
        raise ValueError("expected edition-2 checkpoint; retrain or explicitly migrate an old checkpoint")
    config = ModelConfig(**payload["config"])
    tokenizer = CharacterTokenizer.from_tokens(payload["tokens"])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("checkpoint vocabulary size does not match configuration")
    model = MiniGPT(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, tokenizer, payload
