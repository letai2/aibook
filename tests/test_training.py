import tempfile
import unittest
from pathlib import Path

import torch

from mini_gpt.checkpoint import load_checkpoint
from mini_gpt.train import build_parser, train


class TrainingTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_exact_cpu_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root/"text.txt"
            corpus.write_text("سلام دنیا و یک متن کوچک. " * 8, encoding="utf-8")
            common = ["--text", str(corpus), "--context-length","4", "--embedding-dim","8",
                      "--num-heads","2", "--num-layers","1", "--dropout","0.1",
                      "--batch-size","2", "--eval-every","2", "--threads","1"]
            parser = build_parser()
            full = train(parser.parse_args(common+["--steps","6","--output",str(root/"full")]))
            first = train(parser.parse_args(common+["--steps","4","--output",str(root/"split")]))
            resumed = train(parser.parse_args(["--text",str(corpus),"--resume",str(first),
                        "--steps","6","--eval-every","2","--threads","1","--output",str(root/"split")]))
            a, tokenizer_a, pa = load_checkpoint(full)
            b, tokenizer_b, pb = load_checkpoint(resumed)
            self.assertEqual(pa["step"], 6)
            self.assertEqual(pb["step"], 6)
            self.assertEqual(tokenizer_a.id_to_token, tokenizer_b.id_to_token)
            for key in a.state_dict():
                self.assertTrue(torch.equal(a.state_dict()[key],b.state_dict()[key]), key)
            ids = torch.tensor([tokenizer_a.encode("سلام")])
            a.eval()
            b.eval()
            self.assertTrue(torch.equal(a(ids)[0], b(ids)[0]))

    def test_refuse_overwrite_and_changed_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root/"text.txt"
            corpus.write_text("abcde " * 20, encoding="utf-8")
            parser = build_parser()
            args = parser.parse_args(["--text",str(corpus),"--output",str(root/"run"),
                        "--steps","1","--context-length","2","--embedding-dim","8",
                        "--num-heads","2","--num-layers","1","--threads","1"])
            saved = train(args)
            with self.assertRaises(ValueError):
                train(args)
            corpus.write_text("different " * 20, encoding="utf-8")
            with self.assertRaises(ValueError):
                train(parser.parse_args(["--text",str(corpus),"--resume",str(saved),
                            "--steps","2","--output",str(root/"run")]))


if __name__ == "__main__":
    unittest.main()
