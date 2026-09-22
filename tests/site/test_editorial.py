"""Whole-book navigation, prose-only linking and offline typography contracts."""
from html.parser import HTMLParser
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import parse_qs, unquote, urlsplit

from tools import build_book as book
from book_src.terminology import annotate_html, normalize_text
from tools.prepare_release import inventory


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.ids, self.code, self.text = [], set(), [], []
        self.stack = []
        self.nested = False

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if data.get('id'):
            self.ids.add(data['id'])
        if tag == 'a':
            self.nested |= 'a' in self.stack
            self.links.append(data)
        if tag not in {'meta','link','input','br','hr','img'}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.stack:
            index = len(self.stack)-1-self.stack[::-1].index(tag)
            del self.stack[index:]

    def handle_data(self, text):
        self.text.append(text)
        if 'pre' in self.stack:
            self.code.append(text)


class EditorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.previous_output = book.OUT
        cls.root = Path(cls.temporary.name)/'public'
        book.main(cls.root)
        cls.pages = {}
        for path in cls.root.rglob('*.html'):
            parser = Links()
            parser.feed(path.read_text(encoding='utf-8'))
            cls.pages[path.relative_to(cls.root).as_posix()] = parser

    @classmethod
    def tearDownClass(cls):
        book.OUT = cls.previous_output
        cls.temporary.cleanup()

    def test_all_learning_units_have_correct_previous_next_and_parent(self):
        for index, (unit,path,_) in enumerate(book.UNITS):
            links = self.pages[path].links
            following = [a for a in links if a.get('rel') == 'next']
            previous = [a for a in links if a.get('rel') == 'prev']
            self.assertEqual(len(following),1,path)
            self.assertEqual(len(previous),1,path)
            self.assertEqual(len([a for a in links if 'data-return-section' in a]),1,path)
            expected_next = book.UNITS[index+1][1] if index+1<len(book.UNITS) else 'project.html'
            expected_previous = book.UNITS[index-1][1] if index else 'guide.html'
            for actual,expected in [(following[0],expected_next),(previous[0],expected_previous)]:
                target = (self.root/path).parent/urlsplit(actual['href']).path
                self.assertEqual(target.resolve(),(self.root/expected).resolve(),unit)

    def test_chapters_parts_answers_and_major_references_have_a_next_step(self):
        paths = [p for p,_ in book.CHAPTERS.values()]
        paths += [f'part-{n:02}/index.html' for n in range(1,11)]
        paths += [p for p in self.pages if p.startswith(('answers/','glossary/','code/'))]
        paths += ['guide.html','windows.html','project.html','api.html','lab.html','journal.html','glossary.html']
        for path in paths:
            self.assertEqual(sum('data-next-step' in a for a in self.pages[path].links),1,path)

    def test_every_concept_has_real_content_related_terms_and_lesson_routes(self):
        required = {'Tensor','Embedding','Token','Tokenizer','Attention','Self-Attention','Transformer',
                    'Layer','Batch','Epoch','Gradient','Backpropagation','Optimizer','Learning Rate',
                    'Loss','Logits','Softmax','Parameter','Dataset','Checkpoint','Fine-Tuning',
                    'Inference','Sampling','Context Window','Layer Normalization','Residual Connection'}
        self.assertTrue(required <= {t.name for t in book.TERMS.values()})
        for slug,term in book.TERMS.items():
            self.assertIn('glossary/'+slug+'.html',self.pages)
            for content in (term.meaning,term.intuition,term.technical,term.example,term.project):
                self.assertTrue(content.strip(),slug)
            self.assertTrue(term.related.split(),slug)
            self.assertTrue(set(term.related.split()) <= book.TERMS.keys(),slug)
            self.assertTrue(set(term.lessons.split()) <= book.BY_ID.keys(),slug)

    def test_technical_links_and_return_anchors_are_valid_without_nested_links(self):
        count = 0
        for path,parser in self.pages.items():
            self.assertFalse(parser.nested,path)
            for anchor in parser.links:
                if 'data-term' not in anchor:
                    continue
                count += 1
                url = urlsplit(anchor['href'])
                destination = ((self.root/path).parent/unquote(url.path)).resolve()
                self.assertTrue(destination.is_relative_to(self.root))
                self.assertTrue(destination.is_file())
                if url.query:
                    origin = urlsplit(parse_qs(url.query)['return'][0])
                    target = (destination.parent/origin.path).resolve()
                    route = target.relative_to(self.root).as_posix()
                    self.assertIn(origin.fragment,self.pages[route].ids,path)
        self.assertGreater(count,1000)

    def test_prose_normalization_preserves_code_attributes_formulas_and_calls(self):
        code = '<pre><code>loss = torch.tensor([1.])\n# نشانه، توجه و گرادیان\n</code></pre>'
        source = '<p>تنسور، گرادیان و Self-attention.</p>'+code+'<p><code>self.attention</code> softmax([1,2]) attention.py</p><div class="math">loss = x</div>'
        rendered = annotate_html(source,'part-01/chapter-01/01-model.html')
        self.assertIn(code,rendered)
        self.assertIn('softmax([1,2]) attention.py',rendered)
        self.assertIn('<div class="math">loss = x</div>',rendered)
        self.assertIn('data-term="tensor"',rendered)
        self.assertIn('data-term="self-attention"',rendered)
        self.assertEqual(normalize_text('خودتوجهی'),'Self-Attention')
        self.assertEqual(normalize_text('توجه کنید؛ نشانهٔ کیفیت نیست؛ پیام خطا'),'توجه کنید؛ نشانهٔ کیفیت نیست؛ پیام خطا')
        self.assertNotIn('Tokenٔ',normalize_text('نشانهٔ بعدی'))
        self.assertEqual(normalize_text('پارامترهایش'), 'Parameterهایش')
        self.assertEqual(normalize_text('مدل دوپارامتری'), 'مدل با دو Parameter')
        self.assertEqual(normalize_text('بهینه‌سازی؛ نمونه‌برداری'), 'بهینه‌سازی؛ Sampling')
        self.assertEqual(normalize_text('مبتنی بر توجهی'), 'مبتنی بر Attention')
        self.assertEqual(normalize_text('توجه نکرده؛ دسته‌بندی؛ نشانه‌گذاری'), 'توجه نکرده؛ دسته‌بندی؛ نشانه‌گذاری')
        self.assertIn('id="term-ref-tensor-1"',rendered)
        heading = annotate_html('<h1>چرا Attention لازم است؟</h1><title>Attention</title>','index.html')
        self.assertIn('<h1>چرا <bdi dir="ltr">Attention</bdi> لازم است؟</h1>',heading)
        self.assertIn('<title>Attention</title>',heading)
        option = annotate_html('<select><option>پرسش، کلید، مقدار</option></select>','lab.html')
        self.assertEqual(option, '<select><option>Query، Key، Value</option></select>')

    def test_supplemental_definitions_keep_complete_focused_sentences(self):
        from book_src.glossary import definition_sentence
        paragraph = '<dfn>Scalar</dfn> is one number. <dfn>Vector</dfn> is a sequence. <dfn>Matrix</dfn> is a table.'
        self.assertEqual(definition_sentence(paragraph,'Vector'), '<dfn>Vector</dfn> is a sequence.')
        combined = '<dfn>AdamW</dfn> combines Adam and <dfn>Weight decay</dfn>. Another sentence.'
        self.assertEqual(definition_sentence(combined,'AdamW'), '<dfn>AdamW</dfn> combines Adam and <dfn>Weight decay</dfn>.')
        code = '<dfn>Tensor</dfn> works with <code>torch.tensor([1.0])</code>. Another sentence.'
        self.assertEqual(definition_sentence(code,'Tensor'), '<dfn>Tensor</dfn> works with <code>torch.tensor([1.0])</code>.')

    def test_conventional_persian_and_first_introduction_glosses_survive(self):
        from book_src.terminology import normalize_html
        self.assertEqual(normalize_text('شبکهٔ عصبی، بردار، ماتریس و مشتق'),
                         'شبکهٔ عصبی، بردار، ماتریس و مشتق')
        source = '<p><dfn>Loss (زیان)</dfn> با خطای Python یکی نیست.</p>'
        parser = Links()
        parser.feed(annotate_html(source,'index.html'))
        self.assertIn('Loss (زیان)', ''.join(parser.text))
        self.assertIn('خطای Python', ''.join(parser.text))
        code = '<code>embedding.weight[ids]</code><pre>loss = x\n# گرادیان</pre>'
        self.assertIn(code,normalize_html('<p>تنسور</p>'+code))
        self.assertIn('Tensor',normalize_html('<p>تنسور</p>'))

    def test_terminology_inventory_covers_actual_concepts_and_evidence(self):
        from book_src.glossary import terminology_inventory, CORRECTED_CONCEPTS
        audit = terminology_inventory()
        self.assertEqual(set(audit),set(book.TERMS))
        self.assertEqual(len(CORRECTED_CONCEPTS),20)
        self.assertTrue(CORRECTED_CONCEPTS <= audit.keys())
        self.assertEqual(audit['context-window']['category'],'E')
        for slug in ('supervised-learning','self-supervised-learning','fine-tuning','sft'):
            self.assertGreaterEqual(len(audit[slug]['evidence']),2)
        for phrase in ('یادگیری نظارت‌شده','یادگیری خودنظارتی'):
            self.assertIn(phrase, book.BY_ID['01-learning'].body)
        self.assertNotIn('یادگیری با هدف مرجع',book.BY_ID['01-learning'].body)

    def test_all_executable_examples_still_match_original_source(self):
        for lesson in book.LESSONS:
            if lesson.code:
                html = (self.root/'answers'/f'{lesson.id}.html').read_text(encoding='utf-8')
                reference = Links()
                reference.feed(re.search(r'<pre\b[^>]*>.*?</pre>',html,re.S)[0])
                self.assertEqual(''.join(reference.code),lesson.code.strip(),lesson.id)
        reference = (self.root/'project.html').read_text(encoding='utf-8')
        for command in ('attention','sampling','network','normalization'):
            self.assertIn(f'<code>{command}</code>', reference)

    def test_local_font_and_license_are_packaged_without_allowing_arbitrary_text(self):
        font = self.root/'assets/fonts/Vazirmatn.woff2'
        self.assertEqual(font.read_bytes()[:4],b'wOF2')
        self.assertIn('SIL OPEN FONT LICENSE',(self.root/'assets/fonts/OFL-Vazirmatn.txt').read_text())
        self.assertIn('assets/fonts/Vazirmatn.woff2',inventory(self.root))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'index.html').write_text('<!doctype html>')
            (path/'private-notes.txt').write_text('not public')
            with self.assertRaisesRegex(ValueError,'Unexpected font/license'):
                inventory(path)


if __name__ == '__main__':
    unittest.main()
