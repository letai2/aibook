"""انتخاب شناسه از امتیازها؛ همهٔ روش‌ها در همین فایل تعریف می‌شوند."""

import math

import torch


def filter_logits(logits, top_k=None, top_p=None):
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[-1] == 0:
        raise ValueError("logits must have shape (B,V), with B,V > 0")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("top_p must be in (0,1]")
    filtered = logits.clone()
    if top_k is not None:
        k = min(top_k, filtered.shape[-1])
        values, indices = torch.topk(filtered, k, dim=-1)
        # حتی در تساوی امتیازها دقیقاً k نامزد باقی می‌ماند.
        filtered = torch.full_like(filtered, float("-inf")).scatter(1, indices, values)
    if top_p is not None and top_p < 1:
        ordered, indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        remove = cumulative >= top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        original_order = torch.zeros_like(remove).scatter(1, indices, remove)
        filtered = filtered.masked_fill(original_order, float("-inf"))
    return filtered


def sampling_distribution(logits, temperature=1.0, top_k=None, top_p=None, greedy=False):
    """توزیعِ روش انتخاب؛ برای انتخاب قطعی یک خانه احتمال یک دارد."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("expected finite logits with shape (B,V)")
    scaled = logits / temperature
    if not torch.isfinite(scaled).all():
        raise ValueError("temperature is too small for these logits and dtype")
    filtered = filter_logits(scaled, top_k, top_p)
    if greedy or top_k == 1:
        # argmax در تساوی، نخستین شناسه را برمی‌گزیند.
        return torch.zeros_like(logits).scatter(1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return torch.softmax(filtered, dim=-1)


def choose_token(logits, temperature=1.0, top_k=None, top_p=None, greedy=False):
    probabilities = sampling_distribution(logits, temperature, top_k, top_p, greedy)
    if greedy or top_k == 1:
        return probabilities.argmax(dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1)
