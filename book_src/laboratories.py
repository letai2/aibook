"""Notebook metadata is the single source for laboratory/lesson connections."""
import json
import html
from pathlib import Path
import re


def catalog(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    labs = []
    for path in (root / 'notebooks').rglob('*.ipynb'):
        if '.ipynb_checkpoints' in path.parts:
            continue
        notebook = json.loads(path.read_text(encoding='utf-8'))
        if not notebook.get('metadata', {}).get('book'):
            continue  # Personal notebooks are not authored curriculum assets.
        metadata = dict(notebook['metadata']['book'])
        identifier = metadata.pop('id', path.stem)
        metadata.setdefault('kind', 'review')
        heading = re.search(r'<h1\b[^>]*>(.*?)</h1>', ''.join(notebook['cells'][0]['source']), re.S).group(1)
        title = html.unescape(re.sub(r'<[^>]+>', '', heading))
        labs.append(dict(id=identifier, path=path.relative_to(root).as_posix(),
                         title=title, **metadata))
    return sorted(labs, key=lambda lab: (lab['kind'] != 'lesson', lab.get('lesson_number', 999), lab['id']))


def lesson_labs(root=None):
    """One primary notebook per lesson; review notebooks are additional assets."""
    result = {}
    for lab in catalog(root):
        if lab['kind'] == 'lesson':
            identifier = lab['primary_lesson']
            if identifier in result or lab['lessons'] != [identifier]:
                raise ValueError(f'Duplicate/ambiguous primary laboratory: {identifier}')
            result[identifier] = lab
    return result


INTRO = r'''<p class="objective">هر درس، یک آزمایشگاه: ابتدا ایده را بفهمید؛ سپس خودتان کد بنویسید و نتیجه را بسنجید.</p>
<p>{lesson_count} دفتر درس‌به‌درس در مسیر اصلی و ۱۲ دفتر مرور چنددرس داریم. دفتر هر درس مستقل است و ورودی‌ها را خودش می‌سازد؛ لازم نیست Kernel یا Checkpoint جلسهٔ قبل را نگه دارید. دفترهای مرور اختیاری‌اند و پیش‌نیازشان در فهرست آمده است.</p>
<h2 id="setup">یک نصب، یک فرمان برای کتاب و Jupyter</h2>
<p>پروژهٔ کامل را دریافت و استخراج کنید؛ <code>run.py</code>، <code>book_src</code>، <code>mini_gpt</code> و <code>notebooks</code> باید کنار هم بمانند. در ریشهٔ پروژه، با همان محیط مجازی این فرمان را اجرا کنید:</p>
<pre><code>python run.py</code></pre>
<p>این فرمان کتاب را در صورت نیاز می‌سازد، سرور کتاب و Jupyter را با همان Python شروع می‌کند و [[start.html|میز کار یادگیری]] را باز می‌کند. دکمهٔ هر آزمایشگاه شما را مستقیم به دفتر مربوط می‌برد. از نسخهٔ ایستای سایت، ابتدا باید پروژه را روی رایانهٔ خودتان اجرا کنید.</p>
<p>[[downloads/mini-gpt-project.zip|دریافت پروژهٔ کامل یادگیری]] · [[windows.html|نصب یک‌باره در Windows]]</p>
<p>سرورها فقط روی <code>127.0.0.1</code> هستند. نشانی خصوصی Jupyter که ترمینال نشان می‌دهد را به اشتراک نگذارید. ترمینال را باز نگه دارید و پس از ذخیرهٔ کارتان، با Ctrl+C هر دو سرویس را متوقف کنید.</p>
<h2>تمرین ناتمام، خطای نصب نیست</h2>
<p>مثال‌ها آمادهٔ اجرا هستند، اما در هر دفتر دو تابع TODO دارید: تمرین و اصلاح خرابی. تا وقتی آن‌ها را ننوشته‌اید، آزمون پیام <code>INCOMPLETE</code> می‌دهد؛ این پیام موفقیت نیست. پس از نوشتن تابع، Cell آن و Cell آزمون را دوباره اجرا کنید. <code>PASS</code> یعنی آزمون‌های مشخص همان تمرین گذشته‌اند. جواب مرجع جدا از دفتر، در صفحهٔ پاسخ درس است.</p>
<p>پیش از اجرا پیش‌بینی بنویسید، یک عامل را تغییر دهید، خرابی عمدی را بررسی کنید و دلیل اصلاح را توضیح دهید. برای آزمون نهایی از <code>Kernel → Restart Kernel and Run All Cells</code> استفاده کنید. خطای قرمزِ مدیریت‌نشده نیاز به بررسی دارد. همهٔ مثال‌های پیش‌فرض کوچک و CPU هستند.</p>
<p>دریافت تکی دفتر برای جایگزینی همان فایل در ساختار پروژه است؛ کل وابستگی‌ها را همراه ندارد. راهنمای همراه پروژه: <code>docs/NOTEBOOKS.md</code>.</p>
<h2>دفترهای درس‌به‌درس و مرور</h2>'''
