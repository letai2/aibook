"""Independent evaluation, Unicode CLI and teaching-contract regressions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader
from mini_gpt.data import prepare_corpus
from mini_gpt.dataset import NextTokenDataset
from mini_gpt.evaluate import evaluate, evaluation_ids
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT


class EducationalAuditTests(unittest.TestCase):
    def test_independent_test_reuses_vocabulary_and_preserves_model(self):
        with tempfile.TemporaryDirectory(prefix='mini gpt آزمایش ') as directory:
            path = Path(directory)
            corpus = path/'original.txt'
            corpus.write_text('abcabcabcabcabcabcabcabc', encoding='utf-8')
            _, _, tokenizer, metadata = prepare_corpus(corpus, 0.75)
            test = path/'متن test.txt'
            test.write_text('abcZabcZabcZ', encoding='utf-8')
            original_tokens = list(tokenizer.id_to_token)
            original_mapping = dict(tokenizer.token_to_id)
            ids, report = evaluation_ids(test, tokenizer, metadata, independent_test=True)
            self.assertEqual(report['split'], 'test')
            self.assertEqual(report['unknown_rate'], 0.25)
            self.assertEqual(ids, tokenizer.encode('abcZabcZabcZ'))
            self.assertEqual(tokenizer.id_to_token, original_tokens)
            self.assertEqual(tokenizer.token_to_id, original_mapping)
            model = MiniGPT(ModelConfig(tokenizer.vocab_size, 3, 8, 2, 1, 0.1))
            before = {key: value.clone() for key, value in model.state_dict().items()}
            loss = evaluate(model, DataLoader(NextTokenDataset(ids, 3), batch_size=2), 'cpu')
            self.assertTrue(0 < loss < 100)
            self.assertTrue(model.training)
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_validation_identity_and_test_separation_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'corpus.txt'
            path.write_text('abcabcabcabc', encoding='utf-8')
            _, valid, tokenizer, metadata = prepare_corpus(path, 0.5)
            self.assertEqual(evaluation_ids(path, tokenizer, metadata)[0], valid)
            with self.assertRaisesRegex(ValueError, 'training corpus'):
                evaluation_ids(path, tokenizer, metadata, independent_test=True)
            path.write_text('abcZabcZabcZ', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'corpus differs'):
                evaluation_ids(path, tokenizer, metadata)
            path.write_text('', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                evaluation_ids(path, tokenizer, metadata, independent_test=True)
            path.write_text('ab', encoding='utf-8')
            ids, _ = evaluation_ids(path, tokenizer, metadata, independent_test=True)
            with self.assertRaises(ValueError):
                NextTokenDataset(ids, 3)

    def test_evaluation_modes_are_mutually_exclusive(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'mini_gpt.evaluate',
                                 '--text', 'one.txt', '--test-text', 'two.txt'],
                                capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'not allowed with argument', result.stderr)

    def test_stage_zero_redirected_unicode_and_probability_label(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'mini_gpt.milestones', '--stage', '0'],
                                capture_output=True, timeout=30,
                                env={**os.environ, 'PYTHONIOENCODING': 'cp1252'})
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.decode('utf-8'))
        self.assertIn('شمارش', report['title'])
        self.assertIn('Unsmoothed', report['probability_contract'])

    def test_sft_mask_is_aligned_with_shifted_targets(self):
        full = ['q0', 'q1', 'sep', 'a0', 'a1']
        inputs, targets = full[:-1], full[1:]
        mask = [target in {'a0', 'a1'} for target in targets]
        self.assertEqual([int(x) for x in mask], [0, 0, 1, 1])
        self.assertEqual([(x, y) for x, y, keep in zip(inputs, targets, mask) if keep],
                         [('sep', 'a0'), ('a0', 'a1')])

    def test_response_loss_mask_keeps_prompt_gradient_paths(self):
        # Illustration outside MiniGPT.forward; this is not a hidden SFT API.
        from torch.nn import functional as F
        torch.manual_seed(31)
        model = MiniGPT(ModelConfig(8,4,8,2,1,0.))
        x = torch.tensor([[1,2,3,4]])
        targets = torch.tensor([[2,3,4,5]])
        selected = torch.tensor([[False,False,True,True]])
        logits, _ = model(x)

        def response_loss(target_ids, keep):
            if not keep.any():
                raise ValueError('At least one target must be selected')
            losses = F.cross_entropy(logits.flatten(0,1),target_ids.flatten(),reduction='none')
            return losses[keep.flatten()].mean()

        loss = response_loss(targets,selected)
        changed = targets.clone()
        changed[0,:2] = torch.tensor([6,7])
        torch.testing.assert_close(loss,response_loss(changed,selected))
        with self.assertRaises(ValueError):
            response_loss(targets,torch.zeros_like(selected))
        explicit_losses = torch.tensor([0.8,0.4,0.6,1.0])
        self.assertAlmostEqual(explicit_losses[selected.flatten()].mean().item(),0.8,places=6)
        loss.backward()
        self.assertGreater(model.token_embedding.weight.grad[1].abs().sum().item(),0)
        with self.assertRaises(ValueError):
            model(x,torch.full_like(targets,-100))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
