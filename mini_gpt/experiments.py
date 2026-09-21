"""آزمایش‌های قابل تکرار برای دیدن علت، نه فقط خواندن جواب."""

import argparse
import math
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .model import MiniGPT
from .sampling import choose_token


def tiny_model():
    torch.manual_seed(7)
    return MiniGPT(ModelConfig(12, context_length=8, embedding_dim=16,
                              num_heads=2, num_layers=2, dropout=0.0))


def attention_numbers():
    q = torch.tensor([[1., 0.], [0., 1.]])
    k = torch.tensor([[1., 1.], [0., 1.]])
    v = torch.tensor([[10., 0.], [0., 20.]])
    scores = q @ k.T
    scaled = scores / math.sqrt(2)
    masked = scaled.masked_fill(~torch.ones(2, 2, dtype=torch.bool).tril(), float("-inf"))
    for name, value in [("Q", q), ("K", k), ("K.T", k.T), ("QK.T", scores),
                        ("scaled", scaled), ("masked", masked),
                        ("weights", masked.softmax(-1)), ("output", masked.softmax(-1) @ v)]:
        print(name, tuple(value.shape), value.tolist())


def future():
    model = tiny_model().eval()
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 9, 10]])
    with torch.no_grad():
        for causal in (True, False):
            x, _ = model(a, causal=causal)
            y, _ = model(b, causal=causal)
            print("causal", causal, "past change", (x[:, :3] - y[:, :3]).abs().max().item())


def positions():
    model = tiny_model().eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[3, 2, 1, 4]])
    with torch.no_grad():
        for use_positions in (False, True):
            x, _ = model(a, use_positions=use_positions)
            y, _ = model(b, use_positions=use_positions)
            print("positions", use_positions, "last-token difference", (x[:, -1] - y[:, -1]).abs().max().item())
    print("Causal masking itself carries order constraints; removing positions need not erase all order information.")


def overfit():
    model = tiny_model()
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    y = torch.tensor([[2, 3, 4, 5, 6, 7]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = model(x, y)[1].item()
    for _ in range(80):
        loss = model(x, y)[1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = model(x, y)[1].item()
    print(f"initial={initial:.4f}, final={final:.4f}")
    return initial, final


def network():
    # مسئلهٔ XOR: پاسخ یک است اگر دو ورودی متفاوت باشند.
    torch.manual_seed(42)
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([0, 1, 1, 0])
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(400):
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("targets", y.tolist(), "predictions", model(x).argmax(-1).tolist(), "loss", loss.item())


def gradients():
    for rate in (0.01, 0.1, 1.1):
        w = -3.0
        for _ in range(20):
            w -= rate * 2 * (w - 2)
        print("learning rate", rate, "w after 20 steps", w)


def residual():
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, 5]])
    for enabled in (True, False):
        model = tiny_model()
        loss = model(x, y, residual=enabled)[1]
        loss.backward()
        print("residual", enabled, "embedding gradient", model.token_embedding.weight.grad.norm().item())
    print("One random run is diagnostic, not proof of better training.")


def shapes():
    model = tiny_model().eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        trace = {}
        logits, _ = model(ids, trace=trace)
    first = trace["layers"][0]
    print("IDs", tuple(ids.shape), "embedding+position", tuple(trace["combined_embedding"].shape),
          "actual first-block attention weights", tuple(first["attention"]["weights"].shape),
          "block output", tuple(first["output"].shape), "logits", tuple(logits.shape))
    try:
        model(ids.float())
    except ValueError as error:
        print("intentional bug:", error)
    print("fixed:", tuple(model(ids.long())[0].shape))


def sampling():
    logits = torch.tensor([[2., 1., 0.]])
    for temperature in (0.5, 1., 2.):
        print("temperature", temperature, "probabilities", (logits / temperature).softmax(-1).tolist())
    print("greedy", choose_token(logits, greedy=True).item())


def normalization():
    original = tiny_model()
    removed = copy.deepcopy(original)
    for block in removed.blocks:
        block.norm_1 = nn.Identity()
        block.norm_2 = nn.Identity()
    removed.final_norm = nn.Identity()
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    for label, model in (("LayerNorm", original), ("Identity", removed)):
        logits, loss = model(ids, targets)
        loss.backward()
        print(label, "loss", loss.item(), "logit std", logits.detach().std().item(),
              "embedding gradient norm", model.token_embedding.weight.grad.norm().item())
    print("Same weights and batch; only normalization changed. One untrained batch is not a training-quality verdict.")


def main():
    choices = {"attention": attention_numbers, "future": future, "positions": positions,
               "overfit": overfit, "network": network, "gradients": gradients,
               "residual": residual, "shapes": shapes, "sampling": sampling,
               "normalization": normalization}
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=choices)
    args = parser.parse_args()
    torch.set_num_threads(1)
    choices[args.name]()


if __name__ == "__main__":
    from .console import configure_console
    configure_console()
    main()
