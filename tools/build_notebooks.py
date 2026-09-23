"""Explicit author command: generate clean lesson notebooks from reviewed labs.

Never run this from the learner launcher: it would overwrite learner edits.
"""
import argparse
import html
import json
import re
from pathlib import Path

from book_src.curriculum import LESSONS, PATHS
from book_src.experience import stage_for
from book_src.learning_time import estimate, duration
from book_src.lab_exercises import exercises
from book_src.lab_exercises.reviews import EXERCISES as REVIEWS
from book_src.glossary import extend_from_lessons
from book_src.terminology import configure_terms, normalize_html, inline_code_html, typography_html

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'title', 'goal', 'prerequisite', 'predict', 'setup', 'task', 'starter',
            'check', 'solution', 'vary', 'vary_code', 'debug', 'bug_code', 'fix',
            'fix_check', 'fix_solution', 'connection', 'takeaway'}

SETUP = '''from pathlib import Path
import sys

project_root = next((p for p in (Path.cwd(), *Path.cwd().parents)
                     if (p / "mini_gpt").is_dir() and (p / "book_src").is_dir()), None)
if project_root is None:
    raise RuntimeError("Extract the complete learning project; open this notebook inside it.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print("Python:", sys.executable)
print("Project:", project_root)
'''

def cell(kind, source, identifier, tags=()):
    value = {'cell_type': kind, 'id': identifier,
             'metadata': {'tags': list(tags)} if tags else {},
             'source': source.strip().splitlines(keepends=True)}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
        compile(source, identifier, 'exec')
    return value

def prose(title, body):
    return f'<div dir="rtl">\n<h2>{title}</h2>\n<p>{body}</p>\n</div>'

def render_markdown(source):
    """One policy for headers, exercises, reflection and preserved review cells."""
    return typography_html(normalize_html(inline_code_html(source)), notebook=True)

def make_notebook(lesson, number, spec):
    if set(spec) != REQUIRED or not all(isinstance(v, str) and v.strip() for v in spec.values()):
        raise ValueError(f'Invalid exercise contract: {lesson.id}: {set(spec) ^ REQUIRED}')
    path = f'notebooks/lessons/{lesson.id}/lab.ipynb'
    title = html.escape(spec['title'])
    header = f'''<div dir="rtl">
<h1>{title}</h1>
<p>درس {number} از {len(LESSONS)} · {html.escape(lesson.title)} · <code dir="ltr">{lesson.id}</code></p>
<p><a href="http://127.0.0.1:8000/{PATHS[lesson.id]}">📖 بازگشت به همین درس</a></p>
<p>{spec['goal']}</p><p>پیش‌نیاز: {spec['prerequisite']}</p>
<p>زمان یادگیری درس همراه با همین دفتر: حدود {duration(estimate(lesson)['minutes'])}. زمان دفتر دوباره به زمان درس اضافه نمی‌شود؛ نصب و تمرین اختیاری جداست.</p>
<p>این دفتر نیمهٔ عملی درس است. مثال‌ها آمادهٔ اجرا هستند؛ دو Cell با برچسب TODO را خودتان کامل کنید. پیام INCOMPLETE یعنی هنوز چیزی ننوشته‌اید، نه اینکه پاسخ درست است. جواب مرجع در این دفتر پنهان نشده است.</p>
<p>از بالا به پایین اجرا کنید. پس از تغییر هر تابع، Cell آن و سپس Cell آزمون را دوباره اجرا کنید. برای بررسی نهایی، از منوی <code>Kernel → Restart Kernel and Run All Cells</code> استفاده کنید.</p>
</div>'''
    cells = [cell('markdown', header, 'orientation'),
             cell('code', SETUP, 'environment', ('setup',)),
             cell('markdown', prose('قبل از اجرا، پیش‌بینی کنید', spec['predict']), 'predict'),
             cell('markdown', '<div dir="rtl"><p>پیش‌بینی من: …</p></div>', 'prediction-notes'),
             cell('code', spec['setup'], 'example', ('example',)),
             cell('markdown', prose('این بار شما کد بنویسید', spec['task']), 'exercise'),
             cell('code', spec['starter'], 'student', ('exercise',)),
             cell('code', spec['check'], 'check', ('check',)),
             cell('markdown', prose('فقط یک عامل را تغییر دهید', spec['vary']), 'variation'),
             cell('code', spec['vary_code'], 'vary', ('experiment',)),
             cell('markdown', prose('خرابی را پیدا کنید', spec['debug']), 'debugging'),
             cell('code', spec['bug_code'], 'bug', ('deliberate-bug',)),
             cell('markdown', prose('اصلاح را خودتان بنویسید', 'علت را توضیح دهید، سپس تابع زیر را کامل کنید. خطای عمدی بالا یک نمونهٔ آموزشی است؛ آزمون پایین باید اصلاح شما را بسنجد.'), 'repair'),
             cell('code', spec['fix'], 'repair-code', ('repair',)),
             cell('code', spec['fix_check'], 'repair-check', ('repair-check',)),
             cell('markdown', prose('در Mini-GPT کجا به کار می‌آید؟', spec['connection']), 'project'),
             cell('markdown', prose('با زبان خودتان توضیح دهید', spec['takeaway'])+'\n<div dir="rtl"><p>پیش‌بینی و مشاهدهٔ من: …</p><p>علت خرابی و اصلاح من: …</p></div>', 'reflection'),
             cell('markdown', f'<div dir="rtl"><p><a href="http://127.0.0.1:8000/{PATHS[lesson.id]}">بازگشت به درس و ادامهٔ مسیر</a> · <a href="http://127.0.0.1:8000/answers/{lesson.id}.html#lab-solution">فقط پس از تلاش: راه‌حل مرجع آزمایشگاه</a></p></div>', 'return')]
    stage, stage_title = stage_for(lesson.id)
    for entry in cells:
        if entry['cell_type'] == 'markdown':
            source = render_markdown(''.join(entry['source']))
            entry['source'] = source.replace('<a href="http://127.0.0.1:8000/', '<a target="_self" href="http://127.0.0.1:8000/').splitlines(keepends=True)
    return path, {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'AI Book (project Python)', 'language': 'python', 'name': 'aibook'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'book': {'id': 'lesson-'+lesson.id, 'kind': 'lesson', 'primary_lesson': lesson.id,
                 'lessons': [lesson.id], 'lesson_number': number, 'html': PATHS[lesson.id],
                 'goal': spec['goal'], 'independent': True, 'stage': stage,
                 'stage_title': stage_title, 'transition': spec['task']}},
        'nbformat': 4, 'nbformat_minor': 5}


def upgrade_review(path, notebook, spec):
    """Keep all original demonstration cells; replace only our authored extension."""
    metadata = notebook['metadata']['book']
    lesson = next(l for l in LESSONS if l.id == metadata['lessons'][-1])
    _, extension = make_notebook(lesson, LESSONS.index(lesson)+1, spec)
    identifier = path.stem
    cells = extension['cells']
    cells[0] = cell('markdown', prose('تمرین تکمیلی: '+spec['title'],
        spec['goal']+' پیش‌نیاز: '+spec['prerequisite']+
        ' مثال‌های قبلی این دفتر را نگه داشته‌ایم. اکنون دو تابع TODO را خودتان بنویسید؛ INCOMPLETE یعنی کار هنوز تمام نشده است.'), 'orientation')
    cells[-1] = cell('markdown', f'<div dir="rtl"><p><a href="http://127.0.0.1:8000/{PATHS[lesson.id]}">بازگشت به درس مرتبط</a> · <a href="http://127.0.0.1:8000/answers/lab-{identifier}.html">فقط پس از تلاش: پاسخ مرجع تمرین تکمیلی</a></p></div>', 'return')
    original = [c for c in notebook['cells'] if not c.get('metadata', {}).get('book_review_extension')]
    for entry in cells:
        entry['id'] = 'review-'+entry['id']
        entry['metadata']['book_review_extension'] = True
    for entry in original+cells:
        if entry['cell_type'] == 'markdown':
            source = render_markdown(''.join(entry['source']))
            entry['source'] = source.replace('<a href="http://127.0.0.1:8000/', '<a target="_self" href="http://127.0.0.1:8000/').splitlines(keepends=True)
    notebook['cells'] = original+cells
    metadata.update(kind='review', exercise_id=identifier)
    notebook['metadata']['kernelspec'] = extension['metadata']['kernelspec']
    return notebook

def main(check=False):
    extend_from_lessons(LESSONS)
    configure_terms()
    specs = exercises()
    if set(specs) != {l.id for l in LESSONS}:
        raise ValueError('Every lesson must have exactly one authored exercise')
    for number, lesson in enumerate(LESSONS, 1):
        relative, notebook = make_notebook(lesson, number, specs[lesson.id])
        path = ROOT / relative
        content = json.dumps(notebook, ensure_ascii=False, indent=1)+'\n'
        if check:
            if not path.is_file() or json.loads(path.read_text(encoding='utf-8')) != notebook:
                raise ValueError(f'Notebook differs from authored exercise: {relative}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
    found = set()
    for path in sorted((ROOT/'notebooks').rglob('*.ipynb')):
        if 'lessons' in path.parts or '.ipynb_checkpoints' in path.parts:
            continue
        original = json.loads(path.read_text(encoding='utf-8'))
        notebook = upgrade_review(path, json.loads(json.dumps(original)), REVIEWS[path.stem])
        found.add(path.stem)
        if check:
            if original != notebook:
                raise ValueError(f'Review extension differs: {path}')
        else:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    if found != set(REVIEWS):
        raise ValueError('Every existing review notebook needs one synthesis exercise')
    print(f'{"Checked" if check else "Generated"} {len(specs)} dedicated lesson laboratories and {len(found)} preserved reviews.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
