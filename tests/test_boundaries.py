import csv
import tempfile
import unittest
from pathlib import Path

import torch

from mini_gpt.sampling import choose_token, filter_logits
from mini_gpt.stages.v6 import build
from mini_gpt.train import build_parser, train
from mini_gpt.config import ModelConfig
from mini_gpt.tokenizer import CharacterTokenizer
from mini_gpt.dataset import NextTokenDataset


class BoundaryTests(unittest.TestCase):
    def test_integer_dimensions_and_character_types(self):
        for value in (8.0, True, "8"):
            with self.assertRaises(ValueError):
                ModelConfig(12, embedding_dim=value)
        with self.assertRaises(ValueError):
            CharacterTokenizer(["a", 2])
        for tokens in ([1.5, 2, 3], [1, -1, 2], [True, 2, 3]):
            with self.assertRaises(ValueError):
                NextTokenDataset(tokens, 1)

    def test_top_p_exact_boundary(self):
        # Softmax([0,0]) is exactly [0.5,0.5]: reaching p is enough.
        kept = torch.isfinite(filter_logits(torch.zeros(1, 2), top_p=0.5))
        self.assertEqual(kept.sum().item(), 1)

    def test_empty_batches_and_temperature_overflow(self):
        with self.assertRaises(ValueError):
            choose_token(torch.empty(0, 3))
        with self.assertRaises(ValueError):
            choose_token(torch.empty(1, 0))
        with self.assertRaises(ValueError):
            choose_token(torch.tensor([[1e30, 0.]]), temperature=1e-30)
        with self.assertRaises(ValueError):
            build(12).generate(torch.empty(0, 2, dtype=torch.long), 2)

    def test_resume_rejects_wrong_directory_or_stale_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "text.txt"
            corpus.write_text("abcde " * 20, encoding="utf-8")
            parser = build_parser()
            args = parser.parse_args(["--text", str(corpus), "--output", str(root / "run"),
                    "--steps", "1", "--context-length", "2", "--embedding-dim", "8",
                    "--num-heads", "2", "--num-layers", "1", "--threads", "1"])
            checkpoint = train(args)
            common = ["--text", str(corpus), "--resume", str(checkpoint), "--steps", "2"]
            with self.assertRaisesRegex(ValueError, "same --output"):
                train(parser.parse_args(common + ["--output", str(root / "other")]))
            with (root / "run" / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow([2, 1.0, 1.0, 1.0])
            with self.assertRaisesRegex(ValueError, "steps differ"):
                train(parser.parse_args(common + ["--output", str(root / "run")]))


if __name__ == "__main__":
    unittest.main()
