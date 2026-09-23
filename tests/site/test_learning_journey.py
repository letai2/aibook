"""Whole-path invariants: dependencies, stable identities and honest workloads."""
import re
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from book_src.curriculum import LESSONS, BY_ID, CHAPTERS, PATHS, LEGACY_PATHS
from book_src.schema import PARTS
from book_src.learning_time import (LESSON_PROFILES, CHECKPOINT_MINUTES, estimate,
                                    aggregate, add_ranges)
from book_src.checkpoints import CHECKPOINTS


class LearningJourneyTests(unittest.TestCase):
    def test_all_previous_lesson_identities_and_urls_survive(self):
        self.assertEqual(len(LEGACY_PATHS), 76)
        for identifier, route in LEGACY_PATHS.items():
            self.assertIn(identifier, BY_ID)
            self.assertEqual(PATHS[identifier], route)
        self.assertEqual(len(LESSONS), 92)
        self.assertEqual(len(PARTS), 15)
        self.assertEqual(len(CHECKPOINTS), len(PARTS))
        self.assertEqual([l.part for l in LESSONS], sorted(l.part for l in LESSONS))

    def test_advanced_topics_follow_their_actual_dependencies(self):
        ids = [l.id for l in LESSONS]
        chain = ['54-generate','62b-lifecycle','65-sft','65a-sft-lab',
                 '65b-lora','66-preference','68-context','67-rag','69-chunks',
                 '70-vectors','71-grounding','72-history','73-summary',
                 '74-memory','75-tools','76-reasoning','77-candidates',
                 '80-controller','81-system-eval','63-scale','64-cache',
                 '82-performance','83-deployment','84-capstone']
        positions = [ids.index(identifier) for identifier in chain]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(set(l.id for _, ls in CHAPTERS.values() for l in ls), set(BY_ID))

    def test_time_profiles_are_complete_and_nontrivial(self):
        self.assertEqual(set(LESSON_PROFILES), set(BY_ID))
        self.assertEqual(set(CHECKPOINT_MINUTES), set(range(1,len(PARTS)+1)))
        for lesson in LESSONS:
            estimate_data = estimate(lesson)
            low, high = estimate_data['minutes']
            self.assertGreaterEqual(low, 25, lesson.id)
            self.assertGreater(high, low)
            self.assertLessEqual(high, 210, lesson.id)
            self.assertEqual(set(estimate_data['components']),
                             {'reading','understanding','code','laboratory','exercise','recall'})
            subtotal = add_ranges(*estimate_data['components'].values())
            self.assertTrue(all(0 <= rounded-raw < 5 for rounded, raw in zip((low,high),subtotal)))
        # Content volume can affect reading, but never substitutes for lab work.
        lesson = BY_ID['80-controller']
        short = estimate(replace(lesson, body='<p>کوتاه</p>'))
        self.assertEqual(short['components']['laboratory'], estimate(lesson)['components']['laboratory'])

    def test_time_aggregates_exactly_before_display_rounding(self):
        parts = [aggregate([l for l in LESSONS if l.part==part], checkpoints=[part])
                 for part in range(1,len(PARTS)+1)]
        self.assertEqual(add_ranges(*parts),aggregate(LESSONS,checkpoints=range(1,len(PARTS)+1)))
        self.assertEqual(add_ranges(*(aggregate(ls) for _,ls in CHAPTERS.values())),aggregate(LESSONS))

    def test_no_early_review_introduces_torch_before_the_tool(self):
        from book_src.lab_exercises.reviews import EXERCISES
        for identifier in ('01_matrix_products','02_probability_loss'):
            self.assertNotIn('import torch',EXERCISES[identifier]['setup'])

    def test_header_estimates_and_all_part_counts_are_rendered(self):
        from tools import build_book as book
        previous = book.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                book.main(root)
                targets = ['index.html', *PATHS.values(), *(path for path,_ in CHAPTERS.values()),
                           *(f'part-{part:02}/index.html' for part in range(1,len(PARTS)+1)),
                           *(f'part-{part:02}/checkpoint.html' for part in range(1,len(PARTS)+1))]
                for route in targets:
                    source = (root/route).read_text(encoding='utf-8')
                    self.assertEqual(source.count('class="learning-time"'),1,route)
                    self.assertIn('data-minutes-low=',source)
                    self.assertIn('data-minutes-high=',source)
                    self.assertIn('learning-time.html',source)
                self.assertEqual(len(book.UNITS),len(LESSONS)+len(PARTS))
                self.assertEqual(len(book.LABS),len(LESSONS)+12)
        finally:
            book.OUT = previous


if __name__ == '__main__':
    unittest.main()
