"""No browser or PyTorch required: preserve code and complete design coverage."""
from html.parser import HTMLParser
import unittest

from tools.build_book import LESSONS, ROOT, code_block
from book_src.visuals import BRIEFS, concept_visual, opening


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []

    def handle_data(self, data):
        self.text.append(data)


class VisualDesignTests(unittest.TestCase):
    def test_every_lesson_has_an_explicit_unique_brief(self):
        self.assertEqual(set(BRIEFS), {lesson.id for lesson in LESSONS})
        self.assertEqual(len(set(BRIEFS.values())), len(LESSONS))

    def test_controlled_compositions_and_themes(self):
        self.assertEqual(len({brief[1] for brief in BRIEFS.values()}), 6)
        self.assertEqual(len({brief[0] for brief in BRIEFS.values()}), 12)
        for left, right in zip(LESSONS, LESSONS[1:]):
            self.assertNotEqual(BRIEFS[left.id][1:], BRIEFS[right.id][1:])

    def test_opening_preserves_title_and_objective(self):
        for index, lesson in enumerate(LESSONS, 1):
            parser = TextOnly()
            parser.feed(opening(lesson, index, len(LESSONS)))
            text = ''.join(parser.text)
            self.assertIn(lesson.title, text)
            self.assertIn(lesson.objective, text)
            self.assertIn('<figcaption>', concept_visual(lesson.id))

    def test_highlighting_preserves_every_example_exactly(self):
        for lesson in LESSONS:
            if lesson.code:
                parser = TextOnly()
                parser.feed(code_block(lesson.code))
                self.assertEqual(''.join(parser.text), lesson.code.strip(), lesson.id)

    def test_highlighting_preserves_all_model_source(self):
        for source in (ROOT / 'mini_gpt').rglob('*.py'):
            text = source.read_text(encoding='utf-8').strip()
            parser = TextOnly()
            parser.feed(code_block(text))
            self.assertEqual(''.join(parser.text), text, source.name)


if __name__ == '__main__':
    unittest.main()
