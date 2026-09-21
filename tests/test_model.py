import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_gpt.attention import CausalSelfAttention
from mini_gpt.config import ModelConfig
from mini_gpt.data import prepare_corpus
from mini_gpt.dataset import NextTokenDataset
from mini_gpt.evaluate import evaluate
from mini_gpt.experiments import overfit
from mini_gpt.model import MiniGPT
from mini_gpt.sampling import choose_token, filter_logits
from mini_gpt.stages.run import build_stage, logits_of
from mini_gpt.tokenizer import CharacterTokenizer


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.config = ModelConfig(12, 8, 16, 2, 2, 0.0)

    def test_config_rejects_invalid_shapes(self):
        for changes in ({"num_heads": 0}, {"embedding_dim": 15, "num_heads": 2},
                        {"num_layers": 0}, {"dropout": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ModelConfig(12, **changes)

    def test_tokenizer_roundtrip_and_persistence(self):
        tokenizer = CharacterTokenizer.from_text("سلام دنیا")
        self.assertEqual(tokenizer.decode(tokenizer.encode("سلام")), "سلام")
        self.assertEqual(tokenizer.encode("ژ"), [0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"tokens.json"
            tokenizer.save(path)
            self.assertEqual(CharacterTokenizer.load(path).id_to_token, tokenizer.id_to_token)
        for tokens in ([], ["a"], ["<|unk|>", "a", "a"], ["<|unk|>", "ab"]):
            with self.assertRaises(ValueError):
                CharacterTokenizer.from_tokens(tokens)

    def test_train_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"corpus.txt"
            path.write_text("aaaaabbbbbZ", encoding="utf-8")
            train, valid, tokenizer, meta = prepare_corpus(path, 10/11)
            self.assertNotIn("Z", tokenizer.token_to_id)
            self.assertEqual(valid, [0])
            self.assertEqual(meta["validation_unknown_rate"], 1.0)

    def test_dataset_shift_boundary(self):
        dataset = NextTokenDataset([1,2,3,4,5], 3)
        self.assertEqual(len(dataset), 2)
        x,y = dataset[1]
        self.assertEqual(x.tolist(), [2,3,4])
        self.assertEqual(y.tolist(), [3,4,5])
        with self.assertRaises(IndexError):
            dataset[2]

    def test_attention_weights_are_probabilities_before_dropout(self):
        config = ModelConfig(12, 8, 16, 2, 1, 0.5)
        attention = CausalSelfAttention(config).train()
        output, weights = attention(torch.randn(2,5,16), return_weights=True)
        self.assertEqual(output.shape, (2,5,16))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2,2,5)))
        self.assertEqual(weights.triu(1).abs().max().item(), 0.)

    def test_causal_prefix_invariance(self):
        model = MiniGPT(self.config).eval()
        a = torch.tensor([[1,2,3,4,5]])
        b = torch.tensor([[1,2,3,9,10]])
        with torch.no_grad():
            first = model(a)[0][:,:3]
            second = model(b)[0][:,:3]
            self.assertTrue(torch.allclose(first, second, atol=1e-7))
            leaked_a = model(a, causal=False)[0][:,:3]
            leaked_b = model(b, causal=False)[0][:,:3]
            self.assertGreater((leaked_a-leaked_b).abs().max().item(), 1e-6)

    def test_non_square_head_shapes(self):
        attention = CausalSelfAttention(ModelConfig(12, 9, 20, 4, 1, 0))
        output, weights = attention(torch.randn(3,7,20), return_weights=True)
        self.assertEqual(output.shape, (3,7,20))
        self.assertEqual(weights.shape, (3,4,7,7))

    def test_all_parameters_receive_gradients(self):
        model = MiniGPT(self.config)
        x = torch.tensor([[1,2,3,4]])
        logits, loss = model(x, torch.tensor([[2,3,4,5]]))
        self.assertEqual(logits.shape, (1,4,12))
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_invalid_inputs(self):
        model = MiniGPT(self.config)
        for ids in (torch.ones(2,3), torch.ones(2,9,dtype=torch.long),
                    torch.tensor([1,2]), torch.tensor([[99]])):
            with self.assertRaises(ValueError):
                model(ids)
        with self.assertRaises(ValueError):
            model(torch.ones(1,3,dtype=torch.long), torch.ones(1,2,dtype=torch.long))

    def test_topk_ties_keep_exact_k(self):
        filtered = filter_logits(torch.zeros(2,7), top_k=3)
        self.assertTrue(torch.equal(torch.isfinite(filtered).sum(-1), torch.tensor([3,3])))
        self.assertEqual(choose_token(torch.zeros(1,7), top_k=1).item(), 0)

    def test_topp_keeps_threshold_crossing_token(self):
        probabilities = torch.tensor([[0.6,0.25,0.1,0.05]])
        filtered = filter_logits(probabilities.log(), top_p=0.8)
        self.assertEqual(torch.isfinite(filtered).tolist(), [[True,True,False,False]])
        filtered = filter_logits(probabilities.log(), top_p=0.01)
        self.assertEqual(torch.isfinite(filtered).sum().item(), 1)

    def test_sampling_controls(self):
        for kwargs in ({"temperature":0}, {"temperature":float("nan")},
                       {"top_p":0}, {"top_k":0}):
            with self.assertRaises(ValueError):
                choose_token(torch.zeros(1,4), **kwargs)

    def test_generation_long_prompt_and_mode_restore(self):
        model = MiniGPT(self.config).train()
        original = torch.tensor([[1,2,3,4,5,6,7,8,9]])
        result = model.generate(original, 4, greedy=True)
        self.assertEqual(result.shape, (1,13))
        self.assertTrue(torch.equal(result[:,:9], original))
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            model.generate(original, 1, temperature=-1)
        self.assertTrue(model.training)

    def test_weighted_evaluation_and_mode(self):
        model = MiniGPT(self.config).train()
        dataset = NextTokenDataset([1,2,3,4,5,6,7,8], 3)
        value = evaluate(model, DataLoader(dataset,batch_size=2), torch.device("cpu"))
        self.assertTrue(model.training)
        model.eval()
        with torch.no_grad():
            expected = sum(model(x[None], y[None])[1].item() for x,y in dataset)/len(dataset)
        self.assertAlmostEqual(value, expected, places=6)

    def test_all_stages(self):
        ids = torch.tensor([[1,2,3,4]])
        for stage in range(1,7):
            self.assertEqual(logits_of(build_stage(stage),ids).shape, (1,4,12))

    def test_overfit_one_batch(self):
        initial, final = overfit()
        self.assertLess(final, initial*0.1)

    def test_manual_cross_entropy(self):
        logits = torch.tensor([[2.,1.,0.]], dtype=torch.float64)
        expected = math.log(math.exp(2)+math.exp(1)+1)-2
        actual = torch.nn.functional.cross_entropy(logits,torch.tensor([0])).item()
        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
