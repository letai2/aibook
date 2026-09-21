"""۲۳ ایستگاه اجرایی روی قطعه‌های مشترک Mini-GPT، نه ۲۳ مدل بی‌ارتباط."""

import argparse
import json
import tempfile
from pathlib import Path

from .tokenizer import CharacterTokenizer
from .stages.v0 import transition_counts


TITLES = [
    "شمارش ادامهٔ نویسه", "نشانه‌بند", "واژگان و شناسه", "بردار نمایش", "اطلاعات موقعیت",
    "توجه تک‌سر بدون پوشش؛ نمونهٔ نادرست برای آموزش علّی", "پوشش آینده", "توجه چندسر",
    "شبکهٔ پیش‌خور", "مسیر جمع", "نرمال‌سازی ویژگی‌ها", "یک بلوک کامل", "چند بلوک",
    "سر مدل زبان", "یک حلقهٔ آموزش کوچک", "ارزیابی بدون تغییر وزن", "ذخیره و بارگذاری",
    "تولید پیاپی", "نمونه‌گیری", "دمای نمونه‌گیری", "نگه‌داشتن k گزینه", "جرم احتمال p",
    "Mini-GPT آموزشی یکپارچه",
]


def run_milestone(number, *, checkpoint=None):
    if type(number) is not int or not 0 <= number < len(TITLES):
        raise ValueError("milestone must be an integer from 0 to 22")
    if checkpoint is not None and number < 16:
        raise ValueError("--checkpoint is available from milestone 16")
    text = "سلام مدل سلام مدل سلام مدل سلام مدل "
    tokenizer = CharacterTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    report = {"milestone": number, "title": TITLES[number], "fixture_text": text,
              "note": "این ایستگاه با ورودی کوچک و ثابت کار می‌کند؛ جای آموزش پیکرهٔ واقعی را نمی‌گیرد."}
    if number == 0:
        counts = transition_counts(token_ids)
        report["probability_contract"] = "Unsmoothed observed count / row total; v0 generation instead uses add-one smoothing over its full vocabulary."
        report["vocabulary_includes_unknown"] = True
        report["transitions"] = [dict(current=tokenizer.id_to_token[current],
                                      following=tokenizer.id_to_token[following], count=count,
                                      probability=count / sum(row.values()))
                                 for current, row in counts.items() for following, count in row.items()]
        return report
    if number == 1:
        report.update(tokens=list(text), encoded=token_ids, decoded=tokenizer.decode(token_ids),
                      unknown_example={"input": "X", "ids": tokenizer.encode("X"),
                                       "decoded": tokenizer.decode(tokenizer.encode("X"))})
        return report
    if number == 2:
        report.update(vocabulary=[dict(id=index, token=token) for index, token in enumerate(tokenizer.id_to_token)],
                      token_to_id=tokenizer.token_to_id, ids=token_ids, decoded=tokenizer.decode(token_ids))
        return report

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from .attention import CausalSelfAttention
    from .checkpoint import load_checkpoint, save_checkpoint
    from .config import ModelConfig
    from .dataset import NextTokenDataset
    from .evaluate import evaluate
    from .model import MiniGPT
    from .sampling import choose_token, sampling_distribution
    from .transformer import FeedForward, TransformerBlock

    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = ModelConfig(tokenizer.vocab_size, 8, 16, 2, 2, 0.0)
    model = MiniGPT(config)
    ids = torch.tensor([token_ids[:8]], dtype=torch.long)
    targets = torch.tensor([token_ids[1:9]], dtype=torch.long)
    report.update(input_ids=ids.tolist(), shifted_targets=targets.tolist(), vocabulary=tokenizer.id_to_token)
    x = model.token_embedding(ids)
    position = model.position_embedding(torch.arange(ids.shape[1]))
    if number == 3:
        result = x
    elif number == 4:
        result = x + position
        report["position_vectors"] = position.detach().tolist()
    elif number in (5, 6, 7):
        attention_config = ModelConfig(tokenizer.vocab_size, 8, 16, 2 if number == 7 else 1, 1, 0.)
        attention = CausalSelfAttention(attention_config)
        result, weights = attention(x + position, return_weights=True, causal=number != 5)
        report["weights"] = weights.detach().tolist()
        report["future_weight_sum"] = weights.triu(1).sum().item()
    elif number == 8:
        result = FeedForward(config)(x)
    elif number == 9:
        update = FeedForward(config)(x)
        result = x + update
        report["update_norm"] = update.norm().item()
        report["input_norm"] = x.norm().item()
    elif number == 10:
        result = nn.LayerNorm(config.embedding_dim)(x)
        report["row_means"] = result.mean(-1).detach().tolist()
        report["row_variances"] = result.var(-1, unbiased=False).detach().tolist()
    elif number == 11:
        result = TransformerBlock(config)(x + position)
    elif number == 12:
        result = x + position
        for block in model.blocks:
            result = block(result)
        report["blocks"] = len(model.blocks)
    else:
        result, loss = model(ids, targets)
        report["initial_loss"] = loss.item()
        if number >= 14:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            for _ in range(20):
                _, loss = model(ids, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            result, loss = model(ids, targets)
            report["final_training_loss"] = loss.item()
        if number >= 15:
            # دادهٔ تمرینی جدا اما مصنوعی/هم‌خانواده است؛ معیار تعمیم عمومی نیست.
            validation_ids = tokenizer.encode("مدل سلام مدل سلام مدل ")
            loader = DataLoader(NextTokenDataset(validation_ids, config.context_length), batch_size=3)
            report["validation_loss"] = evaluate(model, loader, torch.device("cpu"))
        if number >= 16:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(checkpoint) if checkpoint is not None else Path(temporary) / "milestone.pt"
                if path.exists():
                    raise ValueError("checkpoint exists; choose a new path")
                path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(path, model, tokenizer, optimizer, 20,
                                {"purpose": "milestone fixture; not a resumable corpus training run"},
                                torch.Generator().manual_seed(43), report["validation_loss"])
                restored, restored_tokenizer, payload = load_checkpoint(path)
                report["checkpoint_roundtrip_equal"] = all(
                    torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
                report["checkpoint"] = str(path) if checkpoint is not None else "temporary file tested and removed"
                model = restored
                tokenizer = restored_tokenizer
        if number >= 17:
            model.eval()
            torch.manual_seed(42)
            prompt = ids[:, :3]
            generated = model.generate(prompt, 4, greedy=True)
            report["generated_ids"] = generated[0].tolist()
            report["generated_text"] = tokenizer.decode(generated[0].tolist())
        if number >= 18:
            logits = model(ids)[0][:, -1].detach()
            options = {"temperature": 0.5 if number == 19 else 1.0,
                       "top_k": 3 if number == 20 else None,
                       "top_p": 0.7 if number >= 21 else None}
            distribution = sampling_distribution(logits, **options)
            report.update(sampling_options=options, logits=logits[0].tolist(),
                          sampling_probabilities=distribution[0].tolist(),
                          sampled_id=choose_token(logits, **options).item())
        if number == 22:
            report["next_real_run"] = "python -m mini_gpt.train --text data/sample.txt --output runs/my-first --steps 100"
            report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    report["input_shape"] = list(ids.shape)
    report["output_shape"] = list(result.shape)
    report["output"] = result.detach().tolist()
    return report


def main():
    parser = argparse.ArgumentParser(description="Run one concrete Mini-GPT milestone, 0 through 22")
    parser.add_argument("--stage", type=int, choices=range(23), required=True)
    parser.add_argument("--output", help="optional new JSON output path")
    parser.add_argument("--checkpoint", help="optional new checkpoint path, milestones 16 and later")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise ValueError("output exists; choose a new path")
    if args.output and args.checkpoint and Path(args.output).resolve() == Path(args.checkpoint).resolve():
        raise ValueError("JSON output and checkpoint need different paths")
    report = run_milestone(args.stage, checkpoint=args.checkpoint)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded)


if __name__ == "__main__":
    from .console import configure_console
    configure_console()
    main()
