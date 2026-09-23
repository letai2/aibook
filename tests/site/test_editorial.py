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
        paths += [f'part-{n:02}/index.html' for n in range(1,len(book.PARTS)+1)]
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
        self.assertEqual(normalize_text('نشانهٔ بهترشدن متن نیست'),'نشانهٔ بهترشدن متن نیست')
        self.assertEqual(normalize_text('پوشش سؤال، پوشش متن‌های تازه و پوشش همهٔ موضوع‌ها'),
                         'پوشش سؤال، پوشش متن‌های تازه و پوشش همهٔ موضوع‌ها')
        self.assertEqual(normalize_text('شکل پوشش؛ توجه تک‌سر؛ توجه چندسر'),
                         'شکل Mask؛ Single-Head Attention؛ Multi-Head Attention')
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
        from book_src.glossary import terminology_inventory, CORRECTED_CONCEPTS, FINAL_TERM_REVIEW
        audit = terminology_inventory()
        self.assertEqual(set(audit),set(book.TERMS))
        self.assertEqual(len(CORRECTED_CONCEPTS),20)
        self.assertTrue(CORRECTED_CONCEPTS <= audit.keys())
        self.assertEqual(len(FINAL_TERM_REVIEW),60)
        self.assertTrue(FINAL_TERM_REVIEW.keys() <= audit.keys())
        self.assertTrue(all(audit[key]['final_review'] for key in FINAL_TERM_REVIEW))
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

    def test_notebook_typography_aligns_prose_without_changing_code_or_math(self):
        from tools.build_notebooks import render_markdown
        source = ('<div dir="rtl"><h1>عنوان Tensor</h1><p>تابع `encode(value, vocabulary)` را بنویسید.</p>'
                  '<ul><li>یک مورد</li></ul><blockquote><p>یادداشت</p></blockquote>'
                  '<p>English only paragraph.</p><pre><code>loss = value\n# نویسه</code></pre>'
                  '<div class="math">x = y + 1</div></div>')
        rendered = render_markdown(source)
        self.assertIn('<h1 style="text-align:right">',rendered)
        self.assertIn('<p style="text-align:right">',rendered)
        self.assertIn('encode(value, vocabulary)</code>',rendered)
        self.assertIn('loss = value\n# نویسه',rendered)
        self.assertRegex(rendered,r'<p[^>]*text-align:left[^>]*dir="ltr"')
        self.assertIn('direction:ltr;text-align:left;unicode-bidi:isolate',rendered)
        self.assertIn('border-right:3px',rendered)
        self.assertIn('border-left:0',rendered)
        self.assertIn('padding-right:1.5em;padding-left:0',rendered)
        self.assertEqual(render_markdown(rendered),rendered)

    def test_notebook_table_cells_use_content_direction(self):
        from tools.build_notebooks import render_markdown
        rendered = render_markdown('<div dir="rtl"><table><tr><th>نام</th><th>Score</th></tr>'
                                   '<tr><td>نمونه</td><td>0.25</td></tr></table></div>')
        self.assertIn('<th dir="rtl" style="text-align:right">نام</th>',rendered)
        self.assertIn('<td dir="ltr" style="text-align:left">0.25</td>',rendered)
        self.assertEqual(render_markdown(rendered),rendered)

    def test_short_math_and_label_groups_do_not_force_whole_sentences(self):
        from book_src.terminology import typography_html
        source = '<p>نکته مهم: مدل هنوز چیزی یاد نگرفته است. شکل (B,T,C) است.</p>'
        rendered = typography_html(source)
        self.assertIn('white-space:nowrap">نکته مهم: مدل</span> هنوز',rendered)
        self.assertIn('class="inline-math" style="white-space:nowrap">(B,T,C)</bdi>',rendered)
        self.assertIn('>[1,0]·[0,1]=0</bdi>',typography_html('<p>حاصل [1,0]·[0,1]=0 است.</p>'))
        long_math = '['+','.join(str(n) for n in range(30))+']'
        self.assertNotIn('nowrap',typography_html('<p>'+long_math+'</p>'))
        formula = '<div class="math">QKᵀ = [[1,0],[1,1]]<br>O = AV</div>'
        self.assertEqual(typography_html(formula),formula)
        self.assertEqual(typography_html(rendered),rendered)

    def test_shared_backticks_cover_headers_and_html_prose(self):
        from tools.build_notebooks import make_notebook
        from book_src.lab_exercises import exercises
        from book_src.terminology import inline_code_html
        spec = dict(exercises()['02-token'], prerequisite="`reduction='none'` و `encode(value, vocabulary)`")
        _, notebook = make_notebook(book.BY_ID['02-token'],3,spec)
        header = ''.join(notebook['cells'][0]['source'])
        self.assertIn('encode(value, vocabulary)</code>',header)
        self.assertNotIn('Reduction',header)
        self.assertIn('reduction=',header)
        rendered = annotate_html(inline_code_html('<p>`project(x, weight, bias)`</p>'),'index.html')
        self.assertIn('project(x, weight, bias)</code>',rendered)

    def test_every_notebook_markdown_cell_uses_the_shared_typography_policy(self):
        import json
        from tools.build_notebooks import render_markdown
        count = 0
        for lab in book.LABS:
            notebook = json.loads((book.ROOT/lab['path']).read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type'] != 'markdown':
                    continue
                count += 1
                source = ''.join(cell['source'])
                self.assertEqual(render_markdown(source),source,(lab['id'],cell['id']))
                self.assertIn('text-align:right',source,(lab['id'],cell['id']))
        self.assertEqual(count,10*len(book.LESSONS)+187)

    def test_inline_comparisons_entities_and_protected_subtrees(self):
        from book_src.terminology import inline_code_html, typography_html
        from tools.build_notebooks import render_markdown
        source = '<p>شرط `0<p<=1` و `1<=k<=V` و `value &lt; target` و `<|unk|>`.</p>'
        rendered = render_markdown('<div dir="rtl">'+source+'</div>')
        for code in ('0&lt;p&lt;=1','1&lt;=k&lt;=V','value &lt; target','&lt;|unk|&gt;'):
            self.assertIn('>'+code+'</code>',rendered)
        self.assertNotIn('`',rendered)
        self.assertEqual(render_markdown(rendered),rendered)
        opaque = ('<!-- `weight` [1,2] --><script>const x = `value`;</script>'
                  '<style>/* `loss` */</style><pre>`value &lt; target`</pre>'
                  '<div class="math">`weight` [1,2]</div><code>`loss`</code>')
        self.assertEqual(inline_code_html(opaque),opaque)
        self.assertEqual(typography_html(opaque),opaque)
        self.assertEqual(inline_code_html('<p title="`weight`">یک ` نشانه</p>'),
                         '<p title="`weight`">یک ` نشانه</p>')
        table = render_markdown('<table><tr><td dir="rtl" style="color:red;text-align:center">12</td></tr></table>')
        self.assertIn('dir="rtl" style="color:red;text-align:right"',table)
        self.assertEqual(render_markdown(table),table)


if __name__ == '__main__':
    unittest.main()
