"""بازکردن جعبهٔ مدل واقعی: خروجی JSON محلی از همان مسیر forward."""

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .runtime import resolve_device, seed_everything
from .sampling import choose_token, sampling_distribution


def values(tensor):
    return tensor.detach().cpu().tolist()


def read_metrics(path):
    if path is None:
        return None
    if Path(path).stat().st_size > 2_000_000:
        raise ValueError("metrics file must be at most 2 MB")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "train_batch_loss", "validation_loss", "gradient_norm"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("metrics must be a Mini-GPT metrics.csv file")
        rows = []
        for row in reader:
            numeric = {key: float(value) for key, value in row.items()}
            if any(not math.isfinite(value) for value in numeric.values()):
                raise ValueError("metrics must contain finite numbers")
            if numeric["step"] < 1 or numeric["step"] != int(numeric["step"]):
                raise ValueError("metrics steps must be positive integers")
            if rows and numeric["step"] <= rows[-1]["step"]:
                raise ValueError("metrics steps must be strictly increasing")
            numeric["step"] = int(numeric["step"])
            rows.append(numeric)
            if len(rows) > 10_000:
                raise ValueError("metrics must contain at most 10000 rows")
    return {"columns": reader.fieldnames, "rows": rows}


@torch.no_grad()
def inspect_model(model, tokenizer, prompt, *, layer=0, head=0, max_tokens=16,
                  generate_tokens=4, temperature=1.0, top_k=None, top_p=None, greedy=False):
    if type(max_tokens) is not int or not 1 <= max_tokens <= 64:
        raise ValueError("max_tokens must be between 1 and 64")
    if type(generate_tokens) is not int or not 0 <= generate_tokens <= 4:
        raise ValueError("generate_tokens must be between 0 and 4")
    if type(layer) is not int or not 0 <= layer < model.config.num_layers:
        raise ValueError("layer is outside the model")
    if type(head) is not int or not 0 <= head < model.config.num_heads:
        raise ValueError("head is outside the model")
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes differ")
    full_ids = tokenizer.encode(prompt)
    if not full_ids:
        raise ValueError("prompt must not be empty")
    limit = min(max_tokens, model.config.context_length)
    context_ids = full_ids[-limit:]
    device = model.token_embedding.weight.device
    inputs = torch.tensor([context_ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        trace = {}
        logits, _ = model(inputs, trace=trace)
        probabilities = logits.softmax(-1)
        # حتی اگر تولید صفر باشد، گزینه‌های نمونه‌گیری نامعتبر زود رد می‌شوند.
        sampling_distribution(logits[:, -1], temperature, top_k, top_p, greedy)
        chosen = trace["layers"][layer]["attention"]
        selected_mask = values(chosen["mask"][0, 0])
        masked_scores = [[float(value) if allowed else None for value, allowed in zip(row, mask_row)]
                         for row, mask_row in zip(values(chosen["masked_scores"][0, head]), selected_mask)]
        attention = {key: values(chosen[key][0, head]) for key in
                     ("q", "k", "v", "raw_scores", "scaled_scores", "weights", "weighted_values")}
        attention.update(layer=layer, head=head, head_dim=model.config.embedding_dim // model.config.num_heads,
                         mask=selected_mask, masked_scores=masked_scores)
        latest_probabilities = probabilities[0, -1]
        ordered = latest_probabilities.argsort(descending=True)[:min(10, tokenizer.vocab_size)].tolist()
        generated_ids = list(full_ids)
        steps = []
        for index in range(generate_tokens):
            active_ids = generated_ids[-limit:]
            active = torch.tensor([active_ids], dtype=torch.long, device=device)
            next_logits = model(active)[0][:, -1]
            distribution = sampling_distribution(next_logits, temperature, top_k, top_p, greedy)
            token_id = choose_token(next_logits, temperature, top_k, top_p, greedy).item()
            steps.append(dict(index=index + 1, context_token_ids=active_ids, token_id=token_id,
                              token=tokenizer.id_to_token[token_id], logits=values(next_logits[0]),
                              probabilities=values(next_logits.softmax(-1)[0]),
                              sampling_probabilities=values(distribution[0])))
            generated_ids.append(token_id)
        return {
            "schema": "mini-gpt-inspection-v1",
            "input": dict(prompt=prompt, full_token_ids=full_ids, context_token_ids=context_ids,
                          context_tokens=[tokenizer.id_to_token[i] for i in context_ids],
                          unknown_positions=[i for i, token_id in enumerate(full_ids) if token_id == 0],
                          truncated=len(full_ids) > limit, context_limit=limit),
            "shapes": dict(token_ids=list(inputs.shape), embedding=list(trace["token_embedding"].shape),
                           qkv=list(chosen["q"].shape), scores=list(chosen["weights"].shape),
                           logits=list(logits.shape)),
            "embeddings": dict(token=values(trace["token_embedding"][0]),
                               position=values(trace["position_embedding"]),
                               combined=values(trace["combined_embedding"][0])),
            "layers": [dict(index=i, input_shape=list(item["input"].shape), output_shape=list(item["output"].shape),
                            attention_shape=list(item["attention"]["weights"].shape),
                            feed_forward_shape=list(item["feed_forward"].shape))
                       for i, item in enumerate(trace["layers"])],
            "attention": attention,
            "output": dict(vocabulary=list(tokenizer.id_to_token), logits=values(logits[0]),
                           probabilities=values(probabilities[0]), greedy_id=logits[0, -1].argmax().item(),
                           top_candidates=[dict(id=i, token=tokenizer.id_to_token[i],
                                                probability=latest_probabilities[i].item()) for i in ordered]),
            "generation": dict(settings=dict(temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy),
                               steps=steps, text=tokenizer.decode(generated_ids), token_ids=generated_ids),
            "notes": ["عددها از مسیر واقعی مدل در حالت ارزیابی‌اند؛ حذف تصادفی خاموش است.",
                      "وزن توجه فقط ضریب ترکیب محتواست؛ به‌تنهایی توضیح علّی تصمیم مدل نیست.",
                      "خانهٔ ممنوع امتیاز با null و پوشش بولی ثبت شده؛ در محاسبه منفی بی‌نهایت بوده است.",
                      "تولید این ابزار نیز از طول زمینهٔ محدودشده استفاده می‌کند؛ شمارهٔ موقعیت هر پنجره از صفر است."],
        }
    finally:
        model.train(was_training)


def export_checkpoint(args):
    destination = Path(args.output)
    protected = [Path(args.checkpoint)] + ([Path(args.metrics)] if args.metrics else [])
    if any(destination.resolve() == source.resolve() for source in protected):
        raise ValueError("output must not replace the checkpoint or metrics input")
    if destination.exists() and not args.force:
        raise ValueError("output exists; choose another path or explicitly use --force")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    model, tokenizer, saved = load_checkpoint(args.checkpoint, resolve_device(args.device))
    seed_everything(args.seed)
    document = inspect_model(model, tokenizer, args.prompt, layer=args.layer, head=args.head,
                             max_tokens=args.max_tokens, generate_tokens=args.generate_tokens,
                             temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, greedy=args.greedy)
    checkpoint_path = Path(args.checkpoint)
    with checkpoint_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    document["source"] = dict(kind="checkpoint", checkpoint_name=checkpoint_path.name,
                              checkpoint_sha256=digest, step=saved["step"], format_version=saved["format_version"],
                              torch_version=str(torch.__version__), config=asdict(model.config),
                              training_metadata=saved["metadata"])
    document["generation"]["settings"]["seed"] = args.seed
    document["metrics"] = read_metrics(args.metrics)
    encoded = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # حالت x بدون force، جلوی بازنویسی ناخواسته در فاصلهٔ بررسی تا ذخیره را هم می‌گیرد.
    with destination.open("w" if args.force else "x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(f"inspection={destination} context={len(document['input']['context_token_ids'])} layer={args.layer} head={args.head}")
    return document


def build_parser():
    parser = argparse.ArgumentParser(description="Export actual Mini-GPT computations as local, strict JSON")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="مدل ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--generate-tokens", type=int, default=4)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    from .console import configure_console
    configure_console()
    export_checkpoint(build_parser().parse_args())
