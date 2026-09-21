"""تولید متن از checkpoint مستقل و ثبت‌شده."""

import argparse

import torch

from .checkpoint import load_checkpoint
from .runtime import resolve_device, seed_everything


def main(args):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    seed_everything(args.seed)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt must contain at least one character")
    if 0 in prompt_ids:
        print("هشدار: بعضی نویسه‌های ورودی در واژگان آموزش وجود ندارند.")
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model.generate(inputs, args.tokens, args.temperature, args.top_k, args.top_p, args.greedy)
    print(tokenizer.decode(result[0].tolist()))


def build_parser():
    parser = argparse.ArgumentParser(description="Generate text from MiniGPT")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--prompt", default="مدل ")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=2)
    return parser


if __name__ == "__main__":
    from .console import configure_console
    configure_console()
    main(build_parser().parse_args())
