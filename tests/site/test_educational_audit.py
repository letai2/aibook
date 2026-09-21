"""Pedagogical order and rendering regressions; no PyTorch dependency."""
import re
import tempfile
from pathlib import Path
import unittest
import zipfile
import posixpath

from tools import build_book as book
from book_src.terminology import annotate_html, normalize_text
from tools.validate_book import validate


class EducationalReleaseTests(unittest.TestCase):
    def test_review_ledger_covers_every_lesson(self):
        source = (book.ROOT/'docs/COVERAGE.md').read_text(encoding='utf-8')
        ids = re.findall(r'^\| `([a-z0-9-]+)` \|', source, re.M)
        self.assertEqual(ids, [lesson.id for lesson in book.LESSONS])

    def test_prerequisite_bridges_and_attention_lab_order(self):
        ids = [lesson.id for lesson in book.LESSONS]
        self.assertLess(ids.index('05-shape'), ids.index('05a-vector-operations'))
        self.assertLess(ids.index('05a-vector-operations'), ids.index('06-dot'))
        self.assertLess(ids.index('12-sgd'), ids.index('12b-neuron'))
        self.assertLess(ids.index('12b-neuron'), ids.index('13-torch'))
        self.assertIn('33-mask', book.INLINE_LABS)
        self.assertNotIn('29-scores', book.INLINE_LABS)
        self.assertNotIn('experiments import future', book.BY_ID['34-causal-test'].code)
        self.assertIn('SingleHead', book.BY_ID['34-causal-test'].code)
        self.assertIn('masked_fill', book.BY_ID['33-mask'].code)
        self.assertIsNone(book.stage_for('12b-neuron')[0])
        self.assertIsNone(book.stage_for('13-torch')[0])
        self.assertEqual(book.stage_for('21-tokenizer')[0], 1)

    def test_explanations_precede_their_abstractions(self):
        text = book.BY_ID['08-probability'].body
        self.assertLess(text.index('توان را'), text.index('<dfn>Natural logarithm'))
        text = book.BY_ID['19-network'].body
        self.assertLess(text.index('6x-1'), text.index('<code>nn.Sequential'))
        self.assertIn('--steps 6', book.BY_ID['47-loop'].body)
        self.assertIn('TemporaryDirectory', book.BY_ID['50-checkpoint'].code)
        self.assertIn('count(two.blocks[1])', book.BY_ID['41-stack'].code)
        self.assertIn('<code>attention(embedding)</code>', book.BY_ID['45-trace'].body)
        self.assertNotIn('پیش از head', book.BY_ID['46-gradient-path'].answer)

    def test_prose_calls_indexing_and_abbreviations_survive_formatting(self):
        source = '<p>model(ids) logits[b,t,v] transpose(1,2) weight[ids]</p>'
        direction = book.MathDirection()
        direction.feed(source)
        rendered = annotate_html(''.join(direction.parts), 'index.html')
        self.assertEqual(rendered, source)
        definition = annotate_html('<dfn>Byte Pair Encoding یا BPE (ادغام جفت‌ها)</dfn>', 'index.html')
        self.assertIn('>BPE</bdi>', definition)
        self.assertIn('data-term="mini-batch"', annotate_html('<p>Mini-batch</p>', 'index.html'))
        for phrase in ('نشانهٔ مشکل', 'نشانهٔ صریح ترتیب', 'نشانهٔ ′'):
            self.assertEqual(normalize_text(phrase), phrase)
        self.assertNotIn('Positional Embedding', normalize_text('نمایش موقعیت‌های یک لایه'))

    def test_glossary_uses_actual_implementation_names_and_context(self):
        text = ' '.join(str(term) for term in book.TERMS.values())
        self.assertNotIn('CharTokenizer', text)
        self.assertNotIn('block_size', text)
        self.assertIn('CharacterTokenizer', text)
        self.assertIn('h_t =', book.TERMS['recurrent-neural-network'].technical)
        self.assertIn('0.8413', book.TERMS['gelu'].example)
        self.assertIn('[2,1,1]', book.TERMS['language-model-head'].example)

    def test_windows_guide_is_packaged_and_paths_are_case_sensitive(self):
        previous = book.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/'site'
                book.main(root)
                self.assertTrue((root/'windows.html').is_file())
                with zipfile.ZipFile(root/'downloads/mini-gpt-project.zip') as archive:
                    self.assertIn('docs/WINDOWS_SETUP.md', archive.namelist())
                    self.assertIn('mini_gpt/console.py', archive.namelist())
                    names = set(archive.namelist())
                    for name in names:
                        if not name.endswith('.md'):
                            continue
                        text = archive.read(name).decode('utf-8')
                        for href in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
                            if '://' in href or href.startswith('#'):
                                continue
                            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), href.split('#')[0]))
                            self.assertIn(target, names, (name, href))
                index = root/'index.html'
                text = index.read_text(encoding='utf-8')
                self.assertIn('assets/book.css', text)
                index.write_text(text.replace('assets/book.css', 'assets/Book.css'), encoding='utf-8')
                with self.assertRaisesRegex(AssertionError, 'wrong-case'):
                    validate(root)
        finally:
            book.OUT = previous


if __name__ == '__main__':
    unittest.main()
