"""آموزش گام‌محور: پنجرهٔ تصادفی، ارزیابی، ذخیره و ادامهٔ قابل تکرار."""

import argparse
import csv
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .data import prepare_corpus
from .dataset import NextTokenDataset
from .evaluate import evaluate
from .model import MiniGPT
from .runtime import resolve_device, seed_everything
from .schedule import ScheduleConfig


def random_batch(dataset, batch_size, generator):
    indices = torch.randint(len(dataset), (batch_size,), generator=generator).tolist()
    pairs = [dataset[i] for i in indices]
    return torch.stack([x for x, _ in pairs]), torch.stack([y for _, y in pairs])


def train(args):
    if args.steps < 1 or args.eval_every < 1 or args.threads < 1:
        raise ValueError("steps, eval_every and threads must be positive")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("output directory is not empty; choose a new --output or use --resume")
    if args.resume and Path(args.resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("resume must use last.pt in the same --output directory; copy the full run to relocate it")

    seed_everything(args.seed)
    batch_generator = torch.Generator().manual_seed(args.seed + 1)
    start_step, best_loss = 0, float("inf")
    if args.resume:
        model, tokenizer, saved = load_checkpoint(args.resume, device)
        config = model.config
        meta = saved["metadata"]
        for name in ("context_length", "embedding_dim", "num_heads", "num_layers", "dropout",
                     "batch_size", "learning_rate", "train_fraction", "schedule",
                     "warmup_steps", "decay_steps", "min_lr_ratio"):
            if getattr(args, name) is not None:
                raise ValueError(f"--resume restores {name}; omit its override")
        batch_size, learning_rate = meta["batch_size"], meta["learning_rate"]
        schedule = ScheduleConfig(**meta.get("schedule", {}))
        train_ids, valid_ids, _, current = prepare_corpus(args.text, meta["train_fraction"], tokenizer)
        if current["corpus_sha256"] != meta["corpus_sha256"]:
            raise ValueError("resume corpus differs from the saved corpus")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        optimizer.load_state_dict(saved["optimizer"])
        # ساخت مدل عدد تصادفی مصرف کرده است؛ وضعیت ذخیره‌شده را پس از آن بازیابی می‌کنیم.
        torch.set_rng_state(saved["rng_cpu"])
        if device.type == "cuda" and saved["rng_cuda"]:
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        batch_generator.set_state(saved["rng_batches"])
        start_step, best_loss = saved["step"], saved["best_loss"]
        if args.steps <= start_step:
            raise ValueError("--steps is the total target step count and must exceed saved step")
        metrics = output / "metrics.csv"
        if not metrics.is_file() or not (output / "best.pt").is_file():
            raise ValueError("resume requires metrics.csv and best.pt alongside last.pt")
        with metrics.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or int(rows[-1]["step"]) != start_step:
            raise ValueError("metrics and last.pt steps differ; restore a consistent run before resuming")
    else:
        fraction = args.train_fraction if args.train_fraction is not None else 0.9
        train_ids, valid_ids, tokenizer, meta = prepare_corpus(args.text, fraction)
        defaults = ModelConfig(tokenizer.vocab_size)
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            **{name: getattr(args, name) if getattr(args, name) is not None else getattr(defaults, name)
               for name in ("context_length", "embedding_dim", "num_heads", "num_layers", "dropout")},
        )
        batch_size = args.batch_size if args.batch_size is not None else 16
        learning_rate = args.learning_rate if args.learning_rate is not None else 3e-4
        schedule = ScheduleConfig(
            mode=args.schedule if args.schedule is not None else "constant",
            warmup_steps=args.warmup_steps if args.warmup_steps is not None else 0,
            decay_steps=args.decay_steps if args.decay_steps is not None else 0,
            min_lr_ratio=args.min_lr_ratio if args.min_lr_ratio is not None else 0.1,
        )
        if batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive")
        model = MiniGPT(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        meta.update(batch_size=batch_size, learning_rate=learning_rate, seed=args.seed,
                    torch_version=str(torch.__version__), threads=args.threads, device=str(device),
                    schedule=asdict(schedule))

    meta.setdefault("schedule", asdict(schedule))
    train_data = NextTokenDataset(train_ids, config.context_length)
    valid_data = NextTokenDataset(valid_ids, config.context_length)
    # DataLoader حتی بدون shuffle از مولد بذر می‌گیرد؛ مولد جدا، مسیر آموزش را ثابت نگه می‌دارد.
    validation = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(0))
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters())} "
          f"vocab={tokenizer.vocab_size} validation_unknown={meta['validation_unknown_rate']:.3%}")
    metrics_path = output / "metrics.csv"
    write_header = not metrics_path.exists()
    columns = ["step", "train_batch_loss", "validation_loss", "gradient_norm", "learning_rate"]
    if not write_header:
        with metrics_path.open(newline="", encoding="utf-8") as existing:
            columns = next(csv.reader(existing))
        if columns not in (["step", "train_batch_loss", "validation_loss", "gradient_norm"],
                           ["step", "train_batch_loss", "validation_loss", "gradient_norm", "learning_rate"]):
            raise ValueError("unrecognized metrics columns")
    with metrics_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for step in range(start_step + 1, args.steps + 1):
            current_rate = schedule.learning_rate(learning_rate, step)
            for group in optimizer.param_groups:
                group["lr"] = current_rate
            inputs, targets = random_batch(train_data, batch_size, batch_generator)
            model.train()
            _, loss = model(inputs.to(device), targets.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                validation_loss = evaluate(model, validation, device)
                is_best = validation_loss < best_loss
                best_loss = min(best_loss, validation_loss)
                writer.writerow(dict(step=step, train_batch_loss=loss.item(), validation_loss=validation_loss,
                                     gradient_norm=float(norm), learning_rate=current_rate))
                stream.flush()
                print(f"step={step:5d} train_batch={loss.item():.4f} val={validation_loss:.4f} grad={norm:.3f} lr={current_rate:.6g}")
                save_checkpoint(output / "last.pt", model, tokenizer, optimizer, step, meta, batch_generator, best_loss)
                if is_best:
                    save_checkpoint(output / "best.pt", model, tokenizer, optimizer, step, meta, batch_generator, best_loss)
    return output / "last.pt"


def build_parser():
    parser = argparse.ArgumentParser(description="Train MiniGPT, or resume to a total number of steps")
    parser.add_argument("--text", default="data/sample.txt")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    for name in ("batch-size", "context-length", "embedding-dim", "num-heads", "num-layers"):
        parser.add_argument("--" + name, type=int, default=None)
    for name in ("dropout", "learning-rate", "train-fraction"):
        parser.add_argument("--" + name, type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--decay-steps", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    return parser


if __name__ == "__main__":
    from .console import configure_console
    configure_console()
    train(build_parser().parse_args())
