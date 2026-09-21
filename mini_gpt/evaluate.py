"""ارزیابی همهٔ هدف‌ها با وزن برابر؛ بدون تغییر پارامترها."""

import argparse
import math
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .dataset import NextTokenDataset
from .runtime import resolve_device


def evaluation_ids(path, tokenizer, training_metadata, *, independent_test=False):
    """Use the checkpoint vocabulary; never fit a vocabulary on held-out text."""
    from .data import prepare_corpus
    if independent_test:
        text = Path(path).read_text(encoding="utf-8")
        if not text:
            raise ValueError("test text must be nonempty")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest == training_metadata["corpus_sha256"]:
            raise ValueError("test text is the training corpus; provide separately held-out documents")
        ids = tokenizer.encode(text)
        unknown_id = tokenizer.token_to_id[tokenizer.UNK]
        return ids, {"split": "test", "unknown_rate": ids.count(unknown_id)/len(ids),
                     "corpus_sha256": digest}
    _, ids, _, metadata = prepare_corpus(path, training_metadata["train_fraction"], tokenizer)
    if metadata["corpus_sha256"] != training_metadata["corpus_sha256"]:
        raise ValueError("corpus differs from training; use --test-text for independently held-out text")
    return ids, {"split": "validation", "unknown_rate": metadata["validation_unknown_rate"],
                 "corpus_sha256": metadata["corpus_sha256"]}


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        for index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            _, loss = model(inputs.to(device), targets.to(device))
            count = targets.numel()
            total_loss += loss.item() * count
            total_tokens += count
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("evaluation loader contains no target tokens")
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate the original validation tail or a separate test document")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument("--text", help="Original training corpus; default: data/sample.txt")
    corpus.add_argument("--test-text", help="Independent UTF-8 test document, evaluated in full with the saved vocabulary")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("threads and batch_size must be positive")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device)
    valid_ids, metadata = evaluation_ids(args.test_text or args.text or "data/sample.txt", tokenizer,
                                         payload["metadata"], independent_test=args.test_text is not None)
    loader = DataLoader(NextTokenDataset(valid_ids, model.config.context_length), batch_size=args.batch_size)
    loss = evaluate(model, loader, device)
    perplexity = math.exp(loss) if loss < 700 else float("inf")
    print(f"{metadata['split']}_loss={loss:.6f} perplexity={perplexity:.4f} unknown_rate={metadata['unknown_rate']:.4f}")
    if args.test_text:
        print("Test uses all overlapping windows. Different file content does not prove independence; check document overlap.")


if __name__ == "__main__":
    from .console import configure_console
    configure_console()
    main()
