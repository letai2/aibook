"""Notebook curriculum, clean-source, links and public-package contracts."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit, unquote
import zipfile

from tools import build_book as book
from tools.prepare_release import inventory
from book_src.terminology import annotate_html


class LaboratoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.previous = book.OUT
        cls.root = Path(cls.temporary.name)
        book.main(cls.root)

    @classmethod
    def tearDownClass(cls):
        book.OUT = cls.previous
        cls.temporary.cleanup()

    def test_catalog_covers_clean_independent_notebooks(self):
        self.assertEqual(len(book.LABS), len(book.LESSONS)+12)
        self.assertEqual(len({lab['id'] for lab in book.LABS}), len(book.LABS))
        primary = [lab for lab in book.LABS if lab['kind'] == 'lesson']
        self.assertEqual({lab['primary_lesson'] for lab in primary},set(book.BY_ID))
        self.assertEqual(len(primary),len(book.LESSONS))
        for lab in book.LABS:
            self.assertTrue(lab['independent'])
            self.assertTrue(lab['goal'])
            self.assertTrue(set(lab['lessons']) <= book.BY_ID.keys())
            if lab['kind'] == 'lesson':
                self.assertEqual(lab['lessons'],[lab['primary_lesson']])
                self.assertEqual(lab['html'],book.PATHS[lab['primary_lesson']])
            notebook = json.loads((book.ROOT/lab['path']).read_text(encoding='utf-8'))
            ids = [cell['id'] for cell in notebook['cells']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(notebook['nbformat'],4)
            for cell in notebook['cells']:
                source = ''.join(cell['source'])
                if cell['cell_type'] == 'code':
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'],[])
                    compile(source,lab['path'],'exec')
                    self.assertNotIn('%pip',source)
            sources = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
            self.assertIn('sys.executable',sources)
            self.assertIn('Path.cwd().parents',sources)
            if lab['id'].startswith(('08','09','10','11','12')):
                self.assertIn('from mini_gpt.',sources)

    def test_build_rejects_missing_or_duplicate_primary_mapping(self):
        first = next(lab for lab in book.LABS if lab['kind'] == 'lesson')
        for labs in ([lab for lab in book.LABS if lab is not first], book.LABS+[first]):
            with patch.object(book,'LABS',labs), self.assertRaisesRegex(ValueError,'exactly one'):
                book.render_lessons()

    def test_personal_unmapped_notebook_is_not_a_public_course_asset(self):
        from book_src.laboratories import catalog
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'notebooks').mkdir()
            (root/'notebooks/personal.ipynb').write_text(json.dumps(
                {'metadata': {}, 'cells': [], 'nbformat':4, 'nbformat_minor':5}))
            self.assertEqual(catalog(root),[])

    def test_notebook_backlinks_and_docs_resolve(self):
        for lab in book.LABS:
            notebook_path = book.ROOT/lab['path']
            notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type'] != 'markdown':
                    continue
                for href in re.findall(r'href="([^"]+)"',''.join(cell['source'])):
                    parts = urlsplit(href)
                    if parts.hostname == '127.0.0.1':
                        self.assertEqual(parts.port,8000)
                        self.assertIn(parts.path.lstrip('/'),set(book.PATHS.values()) |
                                      {'answers/'+identifier+'.html' for identifier in book.BY_ID} |
                                      {'answers/lab-'+item['id']+'.html' for item in book.LABS if item['kind'] == 'review'})
                    elif not parts.scheme:
                        self.assertTrue((notebook_path.parent/unquote(parts.path)).is_file(),href)

    def test_each_mapping_is_rendered_and_downloaded(self):
        manifest = json.loads((self.root/'assets/manifest.js').read_text(encoding='utf-8')
                              .removeprefix('window.BOOK = ').strip().removesuffix(';'))
        self.assertEqual(manifest['laboratories'],book.LABS)
        landing = (self.root/'notebooks.html').read_text(encoding='utf-8')
        for lab in book.LABS:
            self.assertIn('id="'+lab['id']+'"',landing)
            self.assertEqual((self.root/lab['path']).read_bytes(),(book.ROOT/lab['path']).read_bytes())
            for identifier in lab['lessons']:
                html = (self.root/book.PATHS[identifier]).read_text(encoding='utf-8')
                self.assertIn('notebooks.html#'+lab['id'],html)
                self.assertIn('rel="next"',html)
                self.assertIn('rel="prev"',html)

    def test_download_contains_labs_docs_and_only_model_tests(self):
        with zipfile.ZipFile(self.root/'downloads/mini-gpt-project.zip') as archive:
            names = set(archive.namelist())
            self.assertTrue({'docs/NOTEBOOKS.md','docs/WINDOWS_SETUP.md',
                             'requirements.txt','requirements-notebooks.txt'} <= names)
            self.assertTrue({lab['path'] for lab in book.LABS} <= names)
            self.assertTrue({'run.py','tools/build_book.py','tools/learning_server.py','book_src/curriculum.py'} <= names)
            self.assertFalse(any(name.startswith(('tests/site/','tests/browser/','dist/','runs/','.venv/','.openai/')) for name in names))
            self.assertTrue(all('__pycache__' not in name and '.ipynb_checkpoints' not in name for name in names))

    def test_student_notebooks_have_real_work_without_reference_fallback(self):
        from book_src.lab_exercises import exercises
        specs = exercises()
        self.assertEqual(set(specs),set(book.BY_ID))
        for lab in book.LABS:
            if lab['kind'] != 'lesson':
                continue
            notebook = json.loads((book.ROOT/lab['path']).read_text(encoding='utf-8'))
            tagged = {tag:cell for cell in notebook['cells'] for tag in cell['metadata'].get('tags',[])}
            self.assertTrue({'exercise','repair','check','repair-check','experiment','deliberate-bug'} <= set(tagged))
            self.assertIn('TODO',''.join(tagged['exercise']['source']))
            self.assertIn('TODO',''.join(tagged['repair']['source']))
            self.assertIn('INCOMPLETE',''.join(tagged['check']['source']))
            sources = '\n'.join(''.join(c['source']) for c in notebook['cells'])
            self.assertNotIn('from book_src.lab_exercises',sources)
            self.assertNotIn(specs[lab['primary_lesson']]['solution'].strip(),sources)
            lesson_html=(self.root/lab['html']).read_text(encoding='utf-8')
            self.assertIn('launch.html?lesson='+lab['primary_lesson'],lesson_html)
            self.assertLess(lesson_html.index('lesson-lab'),lesson_html.index('class="practice"'))

    def test_public_inventory_rejects_unlisted_or_executed_notebooks(self):
        stray = self.root/'private.ipynb'
        stray.write_text('{}',encoding='utf-8')
        try:
            with self.assertRaisesRegex(ValueError,'Unexpected public notebook'):
                inventory(self.root)
        finally:
            stray.unlink()
        path = self.root/book.LABS[0]['path']
        original = path.read_text(encoding='utf-8')
        notebook = json.loads(original)
        cell = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
        cell['execution_count'] = 1
        path.write_text(json.dumps(notebook),encoding='utf-8')
        try:
            with self.assertRaisesRegex(ValueError,'Executed notebook'):
                inventory(self.root)
        finally:
            path.write_text(original,encoding='utf-8')

    def test_new_bridges_precede_their_dependents(self):
        ids = list(book.BY_ID)
        self.assertLess(ids.index('26a-sequence-models'),ids.index('26b-sequence-memory'))
        self.assertLess(ids.index('62b-lifecycle'),ids.index('65-sft'))
        self.assertIn('T_out',book.BY_ID['42-families'].body)
        self.assertIn('Foundation model',book.BY_ID['62b-lifecycle'].body)
        self.assertIn('Alignment',book.BY_ID['62b-lifecycle'].body)
        self.assertIn('104',book.BY_ID['26a-sequence-models'].answer)

    def test_glossary_links_are_selective_but_old_return_anchors_survive(self):
        source = '<p>Tensor سپس Tensor</p><dfn>Tensor</dfn><code>Tensor</code>'
        rendered = annotate_html(source,'part-01/chapter-01/01-model.html')
        self.assertEqual(rendered.count('class="term-link"'),2)
        self.assertIn('id="term-ref-tensor-2"',rendered)
        self.assertIn('<code>Tensor</code>',rendered)


if __name__ == '__main__':
    unittest.main()
