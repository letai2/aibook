"""Small, inspectable response-only SFT using the real MiniGPT.

The bundled experiment learns four one-character answers about this course.
It is a mechanism demonstration, not an instruction-following assistant.
No downloads, checkpoint prerequisites or filesystem writes are required.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .dataset import NextTokenDataset
from .model import MiniGPT
from .tokenizer import CharacterTokenizer
from .train import random_batch


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    answer: str


TRAIN_EXAMPLES = (
    InstructionExample("Does training update weights?", "y"),
    InstructionExample("Does generation update weights?", "n"),
    InstructionExample("Does training use targets?", "y"),
    InstructionExample("Are generated answers always correct?", "n"),
)
HELDOUT_EXAMPLES = (
    InstructionExample("Do weights change during training?", "y"),
    InstructionExample("Do weights change during generation?", "n"),
)
# Deliberately no formatted instruction/answer records in this pretraining text.
PRETRAIN_TEXT = (
    "Models learn from data. Training changes weights. "
    "Generating text does not update weights. The next token is a training target. "
    "Correct answers are not guaranteed. Do we have enough data? We need tests. "
    "y means yes and n means no. "
) * 8


def format_prompt(instruction):
    """A fixed visible format; the toy answer is y (yes) or n (no)."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty text")
    return "Question: " + instruction + "\nAnswer: "


def encode_example(tokenizer, example, context_length):
    """Return (x, y, response_mask), each with shape (1,T), without padding.

    The mask is aligned to shifted TARGETS. Unknown characters and oversized
    records are rejected instead of silently erasing the training objective.
    """
    if not isinstance(example.answer, str) or not example.answer:
        raise ValueError("answer must be nonempty text")
    if type(context_length) is not int or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    prefix = format_prompt(example.instruction)
    ids = tokenizer.encode(prefix + example.answer)
    if 0 in ids:
        raise ValueError("example contains characters outside the fixed vocabulary")
    if len(ids) - 1 > context_length:
        raise ValueError("instruction and answer exceed the training context")
    sequence = torch.tensor([ids], dtype=torch.long)
    # Target j is original token j+1; the first answer token starts at len(prefix).
    mask = torch.arange(len(ids) - 1)[None, :] >= len(prefix) - 1
    return sequence[:, :-1], sequence[:, 1:], mask


def response_loss(logits, targets, mask):
    """Token-weighted mean CE over selected response targets only."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or mask.shape != targets.shape:
        raise ValueError("expected logits (B,T,V) and targets/mask (B,T)")
    if targets.dtype != torch.long or mask.dtype != torch.bool:
        raise ValueError("targets must be long and mask must be bool")
    if logits.device != targets.device or logits.device != mask.device:
        raise ValueError("logits, targets and mask must share a device")
    if not mask.any().item():
        raise ValueError("response mask selects no targets")
    if targets.min().item() < 0 or targets.max().item() >= logits.shape[-1]:
        raise ValueError("all target IDs must belong to the vocabulary")
    return F.cross_entropy(logits[mask], targets[mask])


def train_response_step(model, optimizer, encoded):
    """One SFT update. The original MiniGPT.forward loss remains unchanged."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x, targets, mask = encoded
    loss = response_loss(model(x)[0], targets, mask)
    if not torch.isfinite(loss).item():
        raise ValueError("nonfinite response loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_instructions(model, tokenizer, examples, *, answer_tokens=1):
    """Measure response CE and greedy exact match for a FIXED answer budget.

    The budget is a task contract, never obtained from each reference answer.
    This toy model has no EOS. Returned rows are real predictions, including
    failures. Response CE is token-weighted; exact match is example-weighted.
    """
    examples = tuple(examples)
    if not examples or type(answer_tokens) is not int or answer_tokens < 1:
        raise ValueError("provide examples and a positive fixed answer budget")
    mode = model.training
    device = next(model.parameters()).device
    total_loss, count, rows = 0.0, 0, []
    model.eval()
    try:
        for example in examples:
            encoded = tuple(t.to(device) for t in encode_example(tokenizer, example, model.config.context_length))
            x, y, mask = encoded
            tokens = int(mask.sum().item())
            total_loss += response_loss(model(x)[0], y, mask).item() * tokens
            count += tokens
            prompt = torch.tensor([tokenizer.encode(format_prompt(example.instruction))], device=device)
            generated = model.generate(prompt, answer_tokens, greedy=True)
            prediction = tokenizer.decode(generated[0, prompt.shape[1]:].tolist())
            rows.append({"instruction": example.instruction, "expected": example.answer,
                         "prediction": prediction, "correct": prediction == example.answer})
    finally:
        model.train(mode)
    return {"response_loss": total_loss / count,
            "exact_match": sum(row["correct"] for row in rows) / len(rows),
            "response_tokens": count, "examples": len(rows), "rows": rows}


@dataclass
class TinyInstructionExperiment:
    base: MiniGPT
    tuned: MiniGPT
    tokenizer: CharacterTokenizer
    reports: dict
    history: list
    metadata: dict


def run_tiny_experiment(*, pretrain_steps=40, sft_steps=160, seed=41,
                        context_length=64, extra_pretraining_text=""):
    """CPU-only, deterministic-in-one-environment, independent two-stage run.

    This controls CPU RNG locally and restores it afterwards. Set the caller's
    torch.set_num_threads(1) for the short teaching workload. The held-out set
    contains unseen prompt WORDING, not new facts or a broad generalization test.
    """
    for value in (pretrain_steps, sft_steps, seed):
        if type(value) is not int or value < 0:
            raise ValueError("step counts and seed must be nonnegative integers")
    if pretrain_steps < 1:
        raise ValueError("at least one pretraining step is required before Fine-Tuning")
    if type(context_length) is not int or context_length < 32:
        raise ValueError("this experiment needs a context_length of at least 32")
    if not isinstance(extra_pretraining_text, str):
        raise ValueError("extra_pretraining_text must be a fixed training string")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        # Only pretraining text and TRAIN records define the vocabulary.
        pretraining_text = PRETRAIN_TEXT + extra_pretraining_text
        vocabulary_text = pretraining_text + "".join(format_prompt(e.instruction) + e.answer for e in TRAIN_EXAMPLES)
        tokenizer = CharacterTokenizer.from_text(vocabulary_text)
        config = ModelConfig(tokenizer.vocab_size, context_length, 24, 2, 1, 0.0)
        encoded = [encode_example(tokenizer, example, config.context_length) for example in TRAIN_EXAMPLES]
        base = MiniGPT(config).cpu()
        dataset = NextTokenDataset(tokenizer.encode(pretraining_text), 32)
        batches = torch.Generator().manual_seed(seed + 1)
        optimizer = torch.optim.AdamW(base.parameters(), lr=0.003, weight_decay=0.0)
        for _ in range(pretrain_steps):
            x, y = random_batch(dataset, 4, batches)
            optimizer.zero_grad(set_to_none=True)
            base(x, y)[1].backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        tuned = deepcopy(base)
        reports = {"base_train": evaluate_instructions(base, tokenizer, TRAIN_EXAMPLES),
                   "base_heldout": evaluate_instructions(base, tokenizer, HELDOUT_EXAMPLES)}
        optimizer = torch.optim.AdamW(tuned.parameters(), lr=0.003, weight_decay=0.0)
        history = []
        for step in range(sft_steps):
            loss = train_response_step(tuned, optimizer, encoded[step % len(encoded)])
            history.append(loss)
        reports.update(tuned_train=evaluate_instructions(tuned, tokenizer, TRAIN_EXAMPLES),
                       tuned_heldout=evaluate_instructions(tuned, tokenizer, HELDOUT_EXAMPLES))
        metadata = {"pretrain_steps": pretrain_steps, "sft_steps": sft_steps,
                    "context_length": context_length,
                    "largest_training_sequence": max(
                        [32] + [x.shape[1] for x, _, _ in encoded[:min(sft_steps, len(encoded))]]),
                    "extra_pretraining_characters": len(extra_pretraining_text),
                    "vocabulary_source": "pretraining corpus + TRAIN instruction records only",
                    "training_objective": "next-token pretraining, then one-character response-only SFT"}
        return TinyInstructionExperiment(base, tuned, tokenizer, reports, history, metadata)


def run_system_experiment(*, pretrain_steps=2, sft_steps=4, context_length=2048, seed=41):
    """Opt-in adapter experiment with a known protocol/serialization vocabulary.

    Capacity is allocated BEFORE training, never resized on an existing model.
    The fixed ASCII/protocol/course corpus is included in pretraining data, not
    learned from an evaluation prompt. Short training still visits only early
    position indices; the large context is NOT evidence of long-context skill.
    """
    import string
    from .assistant import PROTOCOL
    from .retrieval import COURSE_DOCUMENTS
    training_context = (string.printable + "\n" + PROTOCOL + "\n"
                        + "\n".join(document.text for document in COURSE_DOCUMENTS)
                        + "\nbrief explanation\n")
    experiment = run_tiny_experiment(pretrain_steps=pretrain_steps, sft_steps=sft_steps,
                                    seed=seed, context_length=context_length,
                                    extra_pretraining_text=training_context)
    experiment.metadata["purpose"] = "actual generation through controller; NOT JSON instruction training"
    experiment.metadata["extra_corpus"] = "fixed ASCII charset, controller protocol and authored course documents"
    return experiment


def save_instruction_bundle(path, model, tokenizer):
    """Write a new inference-only bundle; never overwrite an existing file.

    This is NOT train.py's resumable checkpoint: no optimizer/RNG/data history.
    A caller should choose a private trusted directory; this demo is not an
    atomic production artifact publisher.
    """
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabulary sizes differ")
    payload = {"kind": "mini-gpt-instruction-inference-v1", "config": asdict(model.config),
               "tokens": list(tokenizer.id_to_token),
               "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def load_instruction_bundle(path):
    """Load only a locally trusted inference bundle, explicitly on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "mini-gpt-instruction-inference-v1":
        raise ValueError("expected an instruction inference bundle, not a resume checkpoint")
    tokenizer = CharacterTokenizer.from_tokens(payload["tokens"])
    config = ModelConfig(**payload["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabulary sizes differ")
    model = MiniGPT(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), tokenizer


if __name__ == "__main__":
    import json
    from .console import configure_console
    configure_console()
    torch.set_num_threads(1)
    experiment = run_tiny_experiment()
    print(json.dumps(experiment.reports, ensure_ascii=False, indent=2))
