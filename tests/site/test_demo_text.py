"""Keep incidental teaching data ASCII without banning Persian instruction.

Only executable string literals are checked: prose, comments and docstrings
remain Persian. Exact exceptions identify experiments where Unicode is the
subject, or prompts for the intentionally Persian Mini-GPT training corpus.
"""
import ast
import json
from pathlib import Path
import re
import unittest

from book_src.curriculum import LESSONS
from book_src.lab_exercises import exercises
from book_src.lab_exercises.reviews import EXERCISES as REVIEWS


ROOT = Path(__file__).resolve().parents[2]
ARABIC_SCRIPT = re.compile(
    r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]')
CODE_FIELDS = ('setup', 'starter', 'check', 'solution', 'vary_code',
               'bug_code', 'fix', 'fix_check', 'fix_solution')
INTENTIONAL = {
    # Compare a space with ZWNJ; inspect unknown Unicode characters explicitly.
    '02-token': {'می روم', 'می‌روم'},
    '21-tokenizer': {'می‌روم home', 'ژ', 'ژچ'},
    # The original review compares Persian/Arabic yeh and ZWNJ tokenization.
    '06_tokens_embeddings': {'مدل می‌رود. مدل می‌آید.'},
    # These original notebooks train on data/sample.txt, not a synthetic demo.
    '11_train_inspect': {'مدل '},
    '12_sampling': {'مدل '},
}


def persian_literals(source):
    tree = ast.parse(source)
    # A standalone string expression is documentation, not example data.
    documentation = {id(node.value) for node in ast.walk(tree)
                     if isinstance(node, ast.Expr)
                     and isinstance(node.value, ast.Constant)}
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in documentation and ARABIC_SCRIPT.search(node.value)}


class DemoTextTests(unittest.TestCase):
    def check_example(self, identity, label, source):
        unexpected = persian_literals(source) - INTENTIONAL.get(identity, set())
        self.assertFalse(unexpected, f'{label}: unexpected Persian demo data {unexpected!r}')

    def test_lesson_reference_programs(self):
        for lesson in LESSONS:
            with self.subTest(lesson=lesson.id):
                self.check_example(lesson.id, lesson.id, lesson.code)

    def test_authored_exercises_and_solutions(self):
        for identity, spec in {**exercises(), **REVIEWS}.items():
            for field in CODE_FIELDS:
                with self.subTest(exercise=identity, field=field):
                    self.check_example(identity, f'{identity}/{field}', spec.get(field, ''))

    def test_all_notebook_code(self):
        paths = [p for p in (ROOT / 'notebooks').rglob('*.ipynb')
                 if '.ipynb_checkpoints' not in p.parts]
        self.assertEqual(len(paths), 104)
        for path in paths:
            notebook = json.loads(path.read_text(encoding='utf-8'))
            metadata = notebook['metadata']['book']
            identity = metadata.get('primary_lesson') or metadata.get('exercise_id')
            self.assertIsNotNone(identity, str(path))
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    with self.subTest(notebook=str(path), cell=index):
                        self.check_example(identity, f'{path}:{index}', ''.join(cell['source']))

    def test_synthetic_runtime_and_training_fixtures(self):
        # UI messages, Unicode-path regressions, and the Persian model corpus
        # are outside this all-ASCII synthetic-data subset.
        for relative in ('mini_gpt/instruction.py', 'mini_gpt/retrieval.py',
                         'mini_gpt/evaluation_system.py', 'mini_gpt/smoke_test.py',
                         'tests/test_model.py', 'tests/test_training.py',
                         'tests/test_inspection_schedule.py'):
            with self.subTest(path=relative):
                self.assertFalse(persian_literals((ROOT / relative).read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
