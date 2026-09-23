import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from mini_gpt.checkpoint import load_checkpoint, save_checkpoint
from mini_gpt.config import ModelConfig
from mini_gpt.inspect import build_parser as inspection_parser, export_checkpoint, inspect_model, read_metrics
from mini_gpt.milestones import run_milestone
from mini_gpt.model import MiniGPT
from mini_gpt.sampling import sampling_distribution
from mini_gpt.schedule import ScheduleConfig
from mini_gpt.tokenizer import CharacterTokenizer
from mini_gpt.train import build_parser, train


class InspectionScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.tokenizer = CharacterTokenizer.from_text("hello model ")
        self.model = MiniGPT(ModelConfig(self.tokenizer.vocab_size, 8, 16, 2, 2, .2))

    def test_shared_trace_matches_forward_and_attention_math(self):
        self.model.eval()
        ids = torch.tensor([self.tokenizer.encode("hello")])
        with torch.no_grad():
            ordinary = self.model(ids)[0]
            trace = {}
            traced = self.model(ids, trace=trace)[0]
        self.assertTrue(torch.equal(ordinary, traced))
        for block in trace["layers"]:
            attention = block["attention"]
            expected_scores = attention["q"] @ attention["k"].transpose(-2, -1)
            self.assertTrue(torch.equal(attention["raw_scores"], expected_scores))
            self.assertTrue(torch.equal(attention["scaled_scores"], expected_scores / math.sqrt(8)))
            self.assertTrue(torch.equal(attention["weights"], attention["masked_scores"].softmax(-1)))
            self.assertTrue(torch.equal(attention["weighted_values"], attention["weights"] @ attention["v"]))

    def test_inspector_real_logits_strict_json_shapes_and_mode(self):
        self.model.train()
        report = inspect_model(self.model, self.tokenizer, "hello model hello", layer=1, head=1,
                               max_tokens=4, generate_tokens=4, greedy=True)
        self.assertTrue(self.model.training)
        self.assertTrue(report["input"]["truncated"])
        self.assertEqual(report["shapes"]["qkv"], [1, 2, 4, 8])
        self.assertEqual(report["attention"]["masked_scores"][0][1], None)
        self.assertEqual(report["attention"]["weights"][0][1], 0)
        self.assertEqual(len(report["generation"]["steps"]), 4)
        json.dumps(report, allow_nan=False)
        self.model.eval()
        ids = torch.tensor([report["input"]["context_token_ids"]])
        self.assertEqual(report["output"]["logits"], self.model(ids)[0][0].tolist())
        for step in report["generation"]["steps"]:
            logits = self.model(torch.tensor([step["context_token_ids"]]))[0][0, -1]
            self.assertEqual(step["logits"], logits.tolist())
            self.assertEqual(step["token_id"], logits.argmax().item())
            self.assertAlmostEqual(sum(step["sampling_probabilities"]), 1.)

    def test_inspection_bounds_and_error_restore(self):
        for options in ({"layer": 2}, {"head": -1}, {"max_tokens": 65}, {"generate_tokens": 5},
                        {"generate_tokens": 0, "temperature": 0}):
            self.model.train()
            with self.assertRaises(ValueError):
                inspect_model(self.model, self.tokenizer, "hello", **options)
            self.assertTrue(self.model.training)

    def test_inspection_export_matches_checkpoint_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            optimizer = torch.optim.AdamW(self.model.parameters())
            save_checkpoint(checkpoint, self.model, self.tokenizer, optimizer, 2, {"test": True},
                            torch.Generator().manual_seed(1), 2.0)
            args = inspection_parser().parse_args(["--checkpoint", str(checkpoint), "--prompt", "hello",
                                                   "--output", str(root / "trace.json"), "--greedy"])
            with contextlib.redirect_stdout(io.StringIO()):
                document = export_checkpoint(args)
            self.assertEqual(document["schema"], "mini-gpt-inspection-v1")
            self.assertEqual(document["source"]["step"], 2)
            self.assertEqual(len(document["source"]["checkpoint_sha256"]), 64)
            self.assertEqual(json.loads((root / "trace.json").read_text(encoding="utf-8")), document)
            with self.assertRaises(ValueError):
                export_checkpoint(args)

    def test_metrics_reader_rejects_nonfinite_or_reversed_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            prefix = "step,train_batch_loss,validation_loss,gradient_norm,learning_rate\n"
            path.write_text(prefix + "1,2,3,4,0.01\n2,1,2,3,0.005\n", encoding="utf-8")
            self.assertEqual(read_metrics(path)["rows"][1]["learning_rate"], .005)
            for text in ("1,nan,2,3,0.01\n", "2,1,2,3,0.01\n1,1,2,3,0.01\n"):
                path.write_text(prefix + text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_metrics(path)

    def test_schedule_endpoints_and_invalid_configuration(self):
        schedule = ScheduleConfig("cosine", 2, 6, .1)
        self.assertAlmostEqual(schedule.learning_rate(.01, 1), .005)
        self.assertAlmostEqual(schedule.learning_rate(.01, 2), .01)
        self.assertAlmostEqual(schedule.learning_rate(.01, 4), .0055)
        self.assertAlmostEqual(schedule.learning_rate(.01, 6), .001)
        self.assertAlmostEqual(schedule.learning_rate(.01, 999), .001)
        self.assertEqual(ScheduleConfig().learning_rate(.01, 100), .01)
        for values in (("cosine", 0, 0, .1), ("cosine", 2, 2, .1), ("constant", 0, 9, .1),
                       ("cosine", 1, 5, -1), ("cosine", True, 5, .1)):
            with self.assertRaises(ValueError):
                ScheduleConfig(*values)

    def test_scheduled_resume_exact_and_horizon_unchanged(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("hello model hello world " * 10, encoding="utf-8")
            parser = build_parser()
            common = ["--text", str(corpus), "--context-length", "4", "--embedding-dim", "8",
                      "--num-heads", "2", "--num-layers", "1", "--batch-size", "2", "--threads", "1",
                      "--dropout", ".2", "--schedule", "cosine", "--warmup-steps", "2", "--decay-steps", "6",
                      "--min-lr-ratio", ".1", "--eval-every", "2"]
            full = train(parser.parse_args(common + ["--steps", "8", "--output", str(root / "full")]))
            first = train(parser.parse_args(common + ["--steps", "3", "--output", str(root / "split")]))
            resume = ["--text", str(corpus), "--resume", str(first), "--output", str(root / "split"),
                      "--steps", "8", "--threads", "1", "--eval-every", "2"]
            resumed = train(parser.parse_args(resume))
            a, _, pa = load_checkpoint(full)
            b, _, pb = load_checkpoint(resumed)
            for key, value in a.state_dict().items():
                self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
            self.assertEqual(pb["metadata"]["schedule"]["decay_steps"], 6)
            self.assertEqual(pa["optimizer"]["param_groups"][0]["lr"], pb["optimizer"]["param_groups"][0]["lr"])
            with (root / "split" / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(rows[-1]["learning_rate"]), .00003)
            with self.assertRaisesRegex(ValueError, "omit its override"):
                train(parser.parse_args(resume + ["--schedule", "constant"]))

    def test_sampling_distribution_contract(self):
        logits = torch.tensor([[2., 1., 0.]])
        self.assertEqual(sampling_distribution(logits, greedy=True).tolist(), [[1., 0., 0.]])
        self.assertEqual((sampling_distribution(logits, top_k=2) > 0).sum().item(), 2)
        self.assertEqual(sampling_distribution(torch.zeros(1, 4), top_k=1).tolist(), [[1., 0., 0., 0.]])

    def test_all_23_milestones_are_executable_and_causal_boundary(self):
        reports = [run_milestone(stage) for stage in range(23)]
        for number, report in enumerate(reports):
            self.assertEqual(report["milestone"], number)
            json.dumps(report, allow_nan=False)
        self.assertGreater(reports[5]["future_weight_sum"], 0.)
        self.assertEqual(reports[6]["future_weight_sum"], 0.)
        self.assertLess(reports[14]["final_training_loss"], reports[14]["initial_loss"])
        self.assertTrue(reports[16]["checkpoint_roundtrip_equal"])
        self.assertEqual(reports[22]["output_shape"][:2], [1, 8])


if __name__ == "__main__":
    unittest.main()
