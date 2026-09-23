"""Small CPU tests for post-training, system evaluation and local execution."""

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharacterTokenizer
from mini_gpt.instruction import (InstructionExample, TRAIN_EXAMPLES, HELDOUT_EXAMPLES,
    format_prompt, encode_example, response_loss, train_response_step,
    evaluate_instructions, run_tiny_experiment, run_system_experiment,
    save_instruction_bundle, load_instruction_bundle)
from mini_gpt.evaluation_system import (EvalCase, score_result, summarize_scores,
    regression_ids, course_fixture_suite, evaluate_cases)
from mini_gpt.performance import (summarize_timings, benchmark_generation,
    quantize_symmetric, RequestLimits, InferenceSession)


class SystemTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.experiment = run_tiny_experiment()

    def test_shifted_response_mask_includes_first_answer(self):
        example = InstructionExample("ab?", "ba")
        tokenizer = CharacterTokenizer.from_text(format_prompt(example.instruction) + example.answer)
        x, y, mask = encode_example(tokenizer, example, 40)
        prefix_length = len(format_prompt(example.instruction))
        self.assertEqual(mask.sum().item(), 2)
        self.assertTrue(mask[0, prefix_length - 1])
        self.assertFalse(mask[0, prefix_length - 2])
        self.assertEqual(tokenizer.decode(y[mask].tolist()), "ba")
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        for bad in (InstructionExample("ab?", ""), InstructionExample("X", "b")):
            with self.assertRaises(ValueError):
                encode_example(tokenizer, bad, 40)
        with self.assertRaises(ValueError):
            encode_example(tokenizer, example, 2)

    def test_response_loss_ignores_prompt_labels_not_prompt_gradient(self):
        logits = torch.tensor([[[5., 1., 0.], [0., 2., 1.], [1., 0., 3.]]], requires_grad=True)
        targets = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[False, True, True]])
        expected = (F.cross_entropy(logits[:, 1], targets[:, 1])
                    + F.cross_entropy(logits[:, 2], targets[:, 2])) / 2
        torch.testing.assert_close(response_loss(logits, targets, mask), expected)
        changed = targets.clone()
        changed[0, 0] = 2
        torch.testing.assert_close(response_loss(logits, changed, mask), expected)
        response_loss(logits, targets, mask).backward()
        self.assertEqual(logits.grad[:, 0].abs().sum().item(), 0)
        model = deepcopy(self.experiment.base)
        x, y, selected = encode_example(self.experiment.tokenizer, TRAIN_EXAMPLES[0], 64)
        model.zero_grad(set_to_none=True)
        response_loss(model(x)[0], y, selected).backward()
        self.assertGreater(model.token_embedding.weight.grad[x[0, 0]].norm().item(), 0)

    def test_response_loss_contract_failures(self):
        logits = torch.randn(1, 3, 4)
        targets = torch.tensor([[0, 1, 2]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        for y, selected in ((targets.float(), mask), (targets, mask.long()),
                            (targets, torch.zeros_like(mask)), (targets[:, :2], mask),
                            (torch.tensor([[0, -100, 2]]), mask)):
            with self.assertRaises(ValueError):
                response_loss(logits, y, selected)

    def test_sft_really_updates_a_copy_and_preserves_base(self):
        experiment = self.experiment
        self.assertIsNot(experiment.base, experiment.tuned)
        self.assertIsNot(experiment.base.token_embedding.weight, experiment.tuned.token_embedding.weight)
        self.assertFalse(torch.equal(experiment.base.token_embedding.weight, experiment.tuned.token_embedding.weight))
        self.assertLess(experiment.reports["tuned_train"]["response_loss"],
                        experiment.reports["base_train"]["response_loss"] / 4)
        self.assertEqual(len(experiment.history), 160)
        # Held-out quality is a measurement, never a mandatory improvement claim.
        self.assertEqual(experiment.reports["tuned_heldout"]["examples"], len(HELDOUT_EXAMPLES))
        self.assertTrue(all(math.isfinite(v) for v in experiment.history))

    def test_eval_is_token_weighted_restores_mode_and_does_not_update(self):
        model = deepcopy(self.experiment.tuned).train()
        tokenizer = self.experiment.tokenizer
        examples = (TRAIN_EXAMPLES[0], InstructionExample(TRAIN_EXAMPLES[1].instruction, "ny"))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        report = evaluate_instructions(model, tokenizer, examples, answer_tokens=1)
        self.assertTrue(model.training)
        self.assertEqual(report["response_tokens"], 3)
        expected = 0
        model.eval()
        with torch.no_grad():
            for example in examples:
                x, y, mask = encode_example(tokenizer, example, 64)
                expected += response_loss(model(x)[0], y, mask).item() * len(example.answer)
        self.assertAlmostEqual(report["response_loss"], expected / 3, places=6)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(len(row["prediction"]) == 1 for row in report["rows"]))

    def test_independent_run_restores_cpu_rng_and_validates_phase(self):
        state = torch.get_rng_state().clone()
        first = run_tiny_experiment(pretrain_steps=1, sft_steps=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        second = run_tiny_experiment(pretrain_steps=1, sft_steps=0)
        self.assertTrue(torch.equal(first.base.token_embedding.weight, second.base.token_embedding.weight))
        self.assertTrue(torch.equal(first.base.token_embedding.weight, first.tuned.token_embedding.weight))
        for options in ({"pretrain_steps": 0}, {"sft_steps": -1}, {"seed": True}):
            with self.assertRaises(ValueError):
                run_tiny_experiment(**options)

    def test_bundle_roundtrip_unicode_path_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="aibook-sft-") as directory:
            path = Path(directory) / "مدل آزمایشی.pt"
            experiment = self.experiment
            save_instruction_bundle(path, experiment.tuned, experiment.tokenizer)
            loaded, tokenizer = load_instruction_bundle(path)
            self.assertEqual(tokenizer.id_to_token, experiment.tokenizer.id_to_token)
            self.assertFalse(loaded.training)
            for name, tensor in experiment.tuned.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]))
            prompt = format_prompt(TRAIN_EXAMPLES[0].instruction)
            self.assertEqual(InferenceSession(loaded, tokenizer).request(prompt),
                             InferenceSession(experiment.tuned, experiment.tokenizer).request(prompt))
            with self.assertRaises(FileExistsError):
                save_instruction_bundle(path, experiment.tuned, experiment.tokenizer)
            payload = torch.load(path, weights_only=True)
            payload["kind"] = "other"
            wrong = Path(directory) / "wrong.pt"
            torch.save(payload, wrong)
            with self.assertRaises(ValueError):
                load_instruction_bundle(wrong)

    def test_training_sequence_metadata_counts_only_visited_examples(self):
        # SFT cycles from the first example; unvisited longer examples do not count.
        for steps in (0, 1, 2, len(TRAIN_EXAMPLES), len(TRAIN_EXAMPLES) + 1):
            experiment = run_tiny_experiment(pretrain_steps=1, sft_steps=steps)
            visited = TRAIN_EXAMPLES[:min(steps, len(TRAIN_EXAMPLES))]
            lengths = [len(format_prompt(e.instruction) + e.answer) - 1 for e in visited]
            self.assertEqual(experiment.metadata['largest_training_sequence'], max([32] + lengths))
        first_length = len(format_prompt(TRAIN_EXAMPLES[0].instruction) + TRAIN_EXAMPLES[0].answer) - 1
        last_length = len(format_prompt(TRAIN_EXAMPLES[-1].instruction) + TRAIN_EXAMPLES[-1].answer) - 1
        self.assertGreater(last_length, max(32, first_length))

    def test_opt_in_controller_path_calls_real_trained_model_without_fallback(self):
        from mini_gpt.assistant import MiniGPTBackend, ScriptedFixture, run_assistant
        from mini_gpt.retrieval import COURSE_DOCUMENTS, chunk_document
        experiment = run_system_experiment()
        self.assertEqual(self.experiment.tuned.config.context_length, 64)
        self.assertEqual(experiment.tuned.config.context_length, 2048)
        self.assertLess(experiment.metadata['largest_training_sequence'], 64)
        self.assertGreater(experiment.metadata['extra_pretraining_characters'], 0)
        self.assertEqual(experiment.metadata['sft_steps'], 4)
        self.assertFalse(torch.equal(experiment.base.language_model_head.weight,
                                    experiment.tuned.language_model_head.weight))
        before = {name: value.clone() for name, value in experiment.tuned.state_dict().items()}
        backend = MiniGPTBackend(experiment.tuned, experiment.tokenizer, max_new_tokens=8)
        chunks = [c for d in COURSE_DOCUMENTS for c in chunk_document(d, chunk_words=24, overlap_words=4)]
        with patch.object(experiment.tuned, 'generate', wraps=experiment.tuned.generate) as generate:
            with patch.object(ScriptedFixture, 'complete', side_effect=AssertionError('No fallback allowed')):
                result = run_assistant('Checkpoint', backend, chunks=chunks)
            self.assertEqual(generate.call_count, 1)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(result.status, 'invalid_action')
        self.assertTrue(any(event['state'] == 'propose' for event in result.events))
        self.assertTrue(all(torch.equal(before[n], value) for n, value in experiment.tuned.state_dict().items()))
        self.assertFalse(result.verified)

    def test_eval_fixtures_are_real_pipeline_not_model_accuracy(self):
        cases, run_case = course_fixture_suite()
        report = evaluate_cases(cases, run_case)
        self.assertEqual(report["cases"], 3)
        self.assertEqual(report["pass_rate"], 1)
        self.assertEqual(report["retrieval_cases"], 1)
        self.assertTrue(all("ScriptedFixture" in row["backend"] for row in report["rows"]))
        result = run_case(cases[1])
        self.assertEqual(result.tool_results[0]["value"], 5)
        self.assertIn("execute", [event["state"] for event in result.events])

    def test_correct_answer_cannot_hide_bad_citation_tool_or_verification(self):
        cases, runner = course_fixture_suite()
        document_case = cases[0]
        result = runner(document_case)
        for altered in (replace(result, citations=()), replace(result, verified=False),
                        replace(result, context_ids=()), replace(result, status="step_limit")):
            score = score_result(document_case, altered)
            self.assertTrue(score["answer_exact"])
            self.assertFalse(score["passed"])
        arithmetic = runner(cases[1])
        self.assertFalse(score_result(cases[1], replace(arithmetic, tool_results=()))["passed"])

    def test_recall_denominator_and_named_regressions(self):
        cases, runner = course_fixture_suite()
        result = runner(cases[0])
        changed_case = replace(cases[0], relevant_ids=(cases[0].relevant_ids[0], "missing"))
        self.assertEqual(score_result(changed_case, result)["retrieval_recall"], 0.5)
        packed_out = score_result(cases[0], replace(result, context_ids=()))
        self.assertEqual(packed_out["retrieval_recall"], 1.0)
        self.assertEqual(packed_out["context_recall"], 0.0)
        report = evaluate_cases(cases, runner)
        after = deepcopy(report["rows"])
        after[1]["passed"] = False
        self.assertEqual(regression_ids(report["rows"], after), ["arithmetic"])
        self.assertEqual(summarize_scores([report["rows"][1]])["mean_retrieval_recall"], None)
        with self.assertRaises(ValueError):
            regression_ids(report["rows"], after[:1])
        with self.assertRaises(ValueError):
            summarize_scores([])
        with self.assertRaises(ValueError):
            summarize_scores([after[0], after[0]])

    def test_timing_summary_uses_work_and_median_not_best(self):
        result = summarize_timings([0.2, 0.1, 0.3], 12)
        self.assertEqual(result["median_seconds"], 0.2)
        self.assertEqual(result["tokens_per_second"], 60)
        self.assertEqual(result["min_seconds"], 0.1)
        for times, count in (([], 1), ([0], 1), ([float("nan")], 1), ([1], 0)):
            with self.assertRaises(ValueError):
                summarize_timings(times, count)

    def test_real_cpu_benchmark_preserves_model_and_counts_batch_tokens(self):
        model = MiniGPT(ModelConfig(8, 8, 8, 2, 1, 0.1)).train()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        prompt = torch.tensor([[1, 2], [2, 3]])
        report = benchmark_generation(model, prompt, new_tokens=2, repeats=2, warmup=1)
        self.assertEqual(report["generated_tokens_per_run"], 4)
        self.assertEqual(report["output_ids"], model.generate(prompt, 2, greedy=True).tolist())
        self.assertGreater(report["median_seconds"], 0)
        self.assertFalse(report["kv_cache"])
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(before[n], v) for n, v in model.state_dict().items()))

    def test_quantization_is_bounded_and_does_not_claim_packed_storage(self):
        values = torch.tensor([-2., -0.3, 0., 0.7, 2.])
        before = values.clone()
        q, scale, restored = quantize_symmetric(values, 4)
        self.assertEqual(q.dtype, torch.int8)
        self.assertLessEqual(q.abs().max().item(), 7)
        self.assertEqual(q.element_size(), 1)
        self.assertLessEqual((restored - values).abs().max().item(), scale / 2 + 1e-6)
        self.assertTrue(torch.equal(before, values))
        zeros, zero_scale, reconstructed = quantize_symmetric(torch.zeros(3))
        self.assertEqual(zero_scale, 1)
        self.assertEqual(zeros.sum().item(), 0)
        self.assertTrue(torch.equal(reconstructed, torch.zeros(3)))
        for invalid in (torch.tensor([float("nan")]), torch.empty(0), torch.tensor([1])):
            with self.assertRaises(ValueError):
                quantize_symmetric(invalid)

    def test_request_limits_prevent_invalid_work_before_generation(self):
        tokenizer = CharacterTokenizer.from_text("abc")
        model = MiniGPT(ModelConfig(tokenizer.vocab_size, 8, 8, 2, 1, 0.0))
        session = InferenceSession(model, tokenizer, RequestLimits(12, 4))
        for prompt, tokens in (("", 1), ("X", 1), ("a" * 13, 1), ("a" * 8, 1),
                               ("abc", 0), ("abc", 5), ("abc", True)):
            with patch.object(model, "generate", wraps=model.generate) as generate:
                with self.assertRaises(ValueError):
                    session.request(prompt, new_tokens=tokens)
                generate.assert_not_called()
        self.assertEqual(session.request("abc", new_tokens=2)["generated_tokens"], 2)
        with self.assertRaises(ValueError):
            RequestLimits(0, 1)


if __name__ == "__main__":
    unittest.main()
