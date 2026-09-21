"""Execute every standalone reference example, not shell command blocks."""
import contextlib
import importlib
import io
import unittest
from importlib.util import find_spec

import torch


class LessonExamples(unittest.TestCase):
    @unittest.skipUnless(find_spec("book_src"), "standalone project ZIP does not include the book sources")
    def test_all_standalone_examples(self):
        torch.set_num_threads(1)
        count = 0
        for part in range(1, 11):
            for lesson in importlib.import_module(f"book_src.part{part:02}").LESSONS:
                if lesson.code and lesson.code_kind.startswith("برنامه"):
                    with self.subTest(lesson=lesson.id), contextlib.redirect_stdout(io.StringIO()):
                        torch.manual_seed(123)
                        exec(compile(lesson.code, f"lesson:{lesson.id}", "exec"), {"__name__": "__main__"})
                        count += 1
        self.assertGreaterEqual(count, 40)


if __name__ == "__main__":
    unittest.main()
