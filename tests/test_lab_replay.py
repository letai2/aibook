"""Replay small notebook continuations without Jupyter or plotting dependencies."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch

from book_src.lab_exercises.foundations import EXERCISES as FOUNDATION_LABS
from book_src.lab_exercises.reviews import EXERCISES as REVIEW_LABS
from book_src.lab_exercises.tools_reasoning import EXERCISES as SYSTEM_LABS
from mini_gpt.dataset import NextTokenDataset
from mini_gpt.sampling import sampling_distribution
from mini_gpt.tokenizer import CharacterTokenizer


ROOT = Path(__file__).resolve().parents[1]


class LabReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def original_cell(self, relative_path, cell_id, namespace):
        notebook = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
        self.assertEqual(len(cells), 1, (relative_path, cell_id))
        self.assertEqual(cells[0]["cell_type"], "code")
        source = "".join(cells[0]["source"])
        # The probability fixture imports plotting alongside its pure-Python math.
        # No plot-producing cell is executed by this suite.
        source = source.replace("import matplotlib.pyplot as plt\n", "")
        exec(compile(source, f"{relative_path}:{cell_id}", "exec"), namespace)

    def replay(self, spec, namespace, solutions):
        for field in ("setup", "starter", "check", "vary_code", "bug_code", "fix", "fix_check"):
            if solutions:
                field = {"starter": "solution", "fix": "fix_solution"}.get(field, field)
            exec(compile(spec[field], f"lab:{field}", "exec"), namespace)
        self.assertIs(namespace["exercise_complete"], solutions)
        self.assertIs(namespace["repair_complete"], solutions)

    def namespace(self):
        # Original non-plot cells only need these entries from their setup cell.
        return {"ROOT": ROOT, "torch": torch, "inspect": lambda name, value: None}

    def test_matrix_review_preserves_original_operands(self):
        path = "notebooks/mathematics/01_matrix_products.ipynb"
        for solutions in (False, True):
            with self.subTest(solutions=solutions), redirect_stdout(StringIO()):
                namespace = self.namespace()
                self.original_cell(path, "cell-04", namespace)
                a, b = namespace["a"], namespace["b"]
                saved_a, saved_b = deepcopy(a), deepcopy(b)
                product = deepcopy(namespace["product"])
                self.replay(REVIEW_LABS["01_matrix_products"], namespace, solutions)
                self.assertIs(namespace["a"], a)
                self.assertIs(namespace["b"], b)
                self.assertEqual(a, saved_a)
                self.assertEqual(b, saved_b)
                self.assertEqual(namespace["matmul"](a, b), product)
                # Replay the numerical part of the original plotting cell.
                changed = deepcopy(a)
                changed[0][1] += 1
                new_product = namespace["matmul"](changed, b)
                difference = [[new_product[i][j] - product[i][j] for j in range(2)]
                              for i in range(2)]
                self.assertEqual(difference, [[0, 1], [0, 0]])
                self.original_cell(path, "cell-08", namespace)

    def test_probability_review_preserves_original_distribution(self):
        path = "notebooks/mathematics/02_probability_loss.ipynb"
        for solutions in (False, True):
            with self.subTest(solutions=solutions), redirect_stdout(StringIO()):
                namespace = self.namespace()
                self.original_cell(path, "cell-04", namespace)
                probabilities = namespace["probabilities"]
                saved = probabilities[:]
                self.replay(REVIEW_LABS["02_probability_loss"], namespace, solutions)
                self.assertIs(namespace["math"], math)
                self.assertIs(namespace["probabilities"], probabilities)
                self.assertEqual(probabilities, saved)
                self.assertEqual(namespace["softmax"](namespace["logits"]), probabilities)
                self.assertLess(namespace["softmax"]([2., 3., 0.])[0], probabilities[0])
                self.original_cell(path, "cell-08", namespace)

    def test_embedding_review_preserves_tokenizer_ids_and_table(self):
        path = "notebooks/nlp/06_tokens_embeddings.ipynb"
        for solutions in (False, True):
            with self.subTest(solutions=solutions), redirect_stdout(StringIO()):
                namespace = self.namespace()
                self.original_cell(path, "cell-04", namespace)
                self.original_cell(path, "cell-06", namespace)
                tokenizer, ids = namespace["tokenizer"], namespace["ids"]
                saved_ids = ids[:]
                embedding = namespace["embedding"]
                saved_weight = embedding.weight.detach().clone()
                self.replay(REVIEW_LABS["06_tokens_embeddings"], namespace, solutions)
                self.assertIs(namespace["tokenizer"], tokenizer)
                self.assertIs(namespace["ids"], ids)
                self.assertEqual(ids, saved_ids)
                self.assertEqual(tokenizer.decode(ids), namespace["reference"])
                self.assertIs(namespace["embedding"], embedding)
                torch.testing.assert_close(embedding.weight, saved_weight, rtol=0, atol=0)
                self.original_cell(path, "cell-08", namespace)

    def test_training_review_preserves_original_model_and_optimizer(self):
        path = "notebooks/mini_gpt/11_train_inspect.ipynb"
        for solutions in (False, True):
            with self.subTest(solutions=solutions), redirect_stdout(StringIO()):
                namespace = self.namespace()
                self.original_cell(path, "cell-04", namespace)
                self.original_cell(path, "cell-06", namespace)
                model, optimizer = namespace["model"], namespace["optimizer"]
                # Populate AdamW state with one real update, not the full training run.
                optimizer.zero_grad(set_to_none=True)
                model(namespace["x"], namespace["y"])[1].backward()
                optimizer.step()
                original_objects = {name: namespace[name]
                                    for name in ("model", "optimizer", "config", "tokenizer")}
                weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
                optimizer_state = deepcopy(optimizer.state_dict())
                self.replay(REVIEW_LABS["11_train_inspect"], namespace, solutions)
                for name, value in original_objects.items():
                    self.assertIs(namespace[name], value, name)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
                after = optimizer.state_dict()
                self.assertEqual(after["param_groups"], optimizer_state["param_groups"])
                self.assertEqual(after["state"].keys(), optimizer_state["state"].keys())
                for parameter_id, values in optimizer_state["state"].items():
                    for name, value in values.items():
                        torch.testing.assert_close(after["state"][parameter_id][name], value,
                                                   rtol=0, atol=0)
                self.original_cell(path, "cell-06", namespace)
                self.assertEqual(namespace["logits"].shape[-1], namespace["tokenizer"].vocab_size)

    def test_sampling_review_preserves_original_logits_and_distribution(self):
        path = "notebooks/mini_gpt/12_sampling.ipynb"
        for solutions in (False, True):
            with self.subTest(solutions=solutions), redirect_stdout(StringIO()):
                namespace = self.namespace()
                self.original_cell(path, "cell-04", namespace)
                model = namespace["model"].eval()
                with torch.no_grad():
                    namespace["logits"] = model(namespace["prompt"])[0][:, -1, :]
                namespace["probabilities"] = sampling_distribution(namespace["logits"], top_p=0.8)[0]
                original_objects = {name: namespace[name] for name in
                                    ("model", "optimizer", "config", "tokenizer", "logits", "probabilities")}
                logits = namespace["logits"].clone()
                probabilities = namespace["probabilities"].clone()
                weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
                self.replay(REVIEW_LABS["12_sampling"], namespace, solutions)
                for name, value in original_objects.items():
                    self.assertIs(namespace[name], value, name)
                torch.testing.assert_close(namespace["logits"], logits, rtol=0, atol=0)
                torch.testing.assert_close(namespace["probabilities"], probabilities, rtol=0, atol=0)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
                self.original_cell(path, "cell-11", namespace)

    def test_shift_lab_starts_with_raw_text_and_real_dataset(self):
        for solutions in (False, True):
            with self.subTest(solutions=solutions), redirect_stdout(StringIO()):
                namespace = {}
                self.replay(FOUNDATION_LABS["23-shift"], namespace, solutions)
                self.assertEqual(namespace["text"], "abcde")
                self.assertEqual(namespace["T"], 3)
                self.assertIsInstance(namespace["tokenizer"], CharacterTokenizer)
                self.assertIsInstance(namespace["reference"], NextTokenDataset)
                self.assertEqual(namespace["ids"], [1, 2, 3, 4, 5])
                self.assertEqual(namespace["tokenizer"].encode(namespace["text"]), namespace["ids"])
                self.assertEqual(namespace["tokenizer"].decode(namespace["first_x"].tolist()), "abc")
                self.assertEqual(namespace["tokenizer"].decode(namespace["first_y"].tolist()), "bcd")
                self.assertEqual(len(namespace["reference"]), 2)

    def test_candidate_lab_reports_actual_verifier_calls(self):
        expected = [
            {"candidate_budget": 1, "voted_answer": 85, "votes": 1,
             "verified_answer": None, "verifier_calls": 1},
            {"candidate_budget": 3, "voted_answer": 135, "votes": 2,
             "verified_answer": 135, "verifier_calls": 3},
            {"candidate_budget": 5, "voted_answer": 85, "votes": 3,
             "verified_answer": 135, "verifier_calls": 5},
        ]
        spec = SYSTEM_LABS["77-candidates"]
        for solutions in (False, True):
            with self.subTest(solutions=solutions), redirect_stdout(StringIO()):
                namespace = {}
                self.replay(spec, namespace, solutions)
                self.assertEqual(namespace["candidate_reports"], expected)
                measured = Mock(wraps=namespace["verify_study_answer"])
                namespace["verify_study_answer"] = measured
                exec(compile(spec["vary_code"], "77-candidates:vary_code", "exec"), namespace)
                self.assertEqual(namespace["candidate_reports"], expected)
                self.assertEqual(measured.call_count, 9)
                self.assertEqual([call.args for call in measured.call_args_list],
                                 [([25, 20], 3, answer) for answer in
                                  [85, 85, 135, 135, 85, 135, 135, 85, 85]])


if __name__ == "__main__":
    unittest.main()
