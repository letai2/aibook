"""CPU measurements and a bounded local inference call, not a public server."""

from dataclasses import dataclass
import math
from statistics import median
from time import perf_counter

import torch


def summarize_timings(seconds, generated_tokens):
    """Every repeat performs the same workload; use median rather than best time."""
    seconds = list(seconds)
    if (not seconds or any(not math.isfinite(s) or s <= 0 for s in seconds)
            or type(generated_tokens) is not int or generated_tokens < 1):
        raise ValueError("positive finite timings and token count required")
    middle = median(seconds)
    return {"repeats": len(seconds), "median_seconds": middle,
            "min_seconds": min(seconds), "max_seconds": max(seconds),
            "generated_tokens_per_run": generated_tokens,
            "tokens_per_second": generated_tokens / middle}


def benchmark_generation(model, prompt, *, new_tokens=8, repeats=3, warmup=1):
    """Measure the real uncached generate() on CPU, with deterministic decoding.

    Times exclude model loading, tokenizer, HTTP, queuing and retrieval. The
    separate one-token call approximates model-side first-token time only; it
    is not end-to-end time-to-first-token of a serving system.
    """
    if next(model.parameters()).device.type != "cpu" or prompt.device.type != "cpu":
        raise ValueError("this teaching benchmark supports CPU only")
    if any(type(v) is not int or v < 1 for v in (new_tokens, repeats)):
        raise ValueError("new_tokens and repeats must be positive integers")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    for _ in range(warmup):
        model.generate(prompt, new_tokens, greedy=True)
    timings, first_timings = [], []
    for _ in range(repeats):
        start = perf_counter()
        output = model.generate(prompt, new_tokens, greedy=True)
        timings.append(perf_counter() - start)
        start = perf_counter()
        model.generate(prompt, 1, greedy=True)
        first_timings.append(perf_counter() - start)
    return {**summarize_timings(timings, new_tokens * prompt.shape[0]),
            "batch_size": prompt.shape[0], "prompt_tokens_per_sequence": prompt.shape[1],
            "new_tokens_per_sequence": new_tokens, "threads": torch.get_num_threads(),
            "first_token_model_seconds": median(first_timings),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "kv_cache": False, "output_ids": output.tolist()}


def quantize_symmetric(values, bits=8):
    """Toy symmetric per-tensor quantization, NOT a faster inference kernel.

    For bits < 8 the returned tensor still physically uses int8 storage. Counting
    bits as if packed is a separate theoretical estimate, not measured memory.
    """
    if type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError("bits must be an integer in 2..8")
    if not values.is_floating_point() or not values.numel() or not torch.isfinite(values).all():
        raise ValueError("a nonempty finite floating tensor is required")
    maximum = values.detach().abs().max().item()
    bound = 2 ** (bits - 1) - 1
    scale = maximum / bound if maximum else 1.0
    quantized = (values.detach() / scale).round().clamp(-bound, bound).to(torch.int8)
    reconstructed = quantized.to(values.dtype) * scale
    return quantized, scale, reconstructed


@dataclass(frozen=True)
class RequestLimits:
    max_prompt_characters: int = 512
    max_new_tokens: int = 32

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_prompt_characters, self.max_new_tokens)):
            raise ValueError("request limits must be positive integers")


class InferenceSession:
    """Reuse one loaded model and reject invalid work before generation.

    Strictly local and synchronous: no listener, threads, sessions or automatic
    memory writes. This bounds tokens, not elapsed wall time. A real timeout,
    concurrency isolation and authentication are separate serving work.
    """
    def __init__(self, model, tokenizer, limits=None):
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("model and tokenizer vocabulary sizes differ")
        self.model, self.tokenizer = model, tokenizer
        self.limits = limits or RequestLimits()

    def request(self, prompt, *, new_tokens=1):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        if len(prompt) > self.limits.max_prompt_characters:
            raise ValueError("prompt exceeds the character budget")
        if type(new_tokens) is not int or not 1 <= new_tokens <= self.limits.max_new_tokens:
            raise ValueError("requested output exceeds the token budget")
        ids = self.tokenizer.encode(prompt)
        if 0 in ids:
            raise ValueError("unknown characters: use the fixed training vocabulary")
        if len(ids) + new_tokens > self.model.config.context_length:
            raise ValueError("prompt plus output reserve exceeds context; refusing silent truncation")
        device = next(self.model.parameters()).device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        output = self.model.generate(inputs, new_tokens, greedy=True)
        return {"text": self.tokenizer.decode(output[0, len(ids):].tolist()),
                "prompt_tokens": len(ids), "generated_tokens": new_tokens,
                "greedy": True, "weights_updated": False}
