"""انتخاب دستگاه و ثبت بذر تصادفی برای آزمایش‌های قابل مقایسه."""

import random

import torch


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
