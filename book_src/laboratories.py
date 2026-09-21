"""Notebook metadata is the single source for laboratory/lesson connections."""
import json
from pathlib import Path
import re


def catalog(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    labs = []
    for path in (root / 'notebooks').rglob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        metadata = notebook['metadata']['book']
        title = re.search(r'<h1>(.*?)</h1>', ''.join(notebook['cells'][0]['source'])).group(1)
        labs.append(dict(id=path.stem, path=path.relative_to(root).as_posix(),
                         title=title, **metadata))
    return sorted(labs, key=lambda lab: lab['id'])


INTRO = '''<p class="objective">کتاب مسیر اصلی است؛ اینجا توقف می‌کنیم تا یک ایده را اجرا، دست‌کاری و بررسی کنیم.</p>
<p>هر دفتر یک پرسش دارد. ابتدا درس‌های مرتبط را بخوانید، نتیجه را پیش‌بینی کنید و سپس Cellها را از بالا به پایین اجرا کنید. دفترها مستقل‌اند؛ لازم نیست <code>Kernel</code> یا Checkpoint دفتر قبلی را نگه دارید. بعد از آزمایش به همان درس برگردید.</p>
<section class="note"><h2>فایل را دریافت کنید؛ کد در این صفحه اجرا نمی‌شود</h2>
<p>Jupyter به Python محلی نیاز دارد. برای شروع، بستهٔ کامل پروژه را دریافت و کامل استخراج کنید تا mini_gpt، data و notebooks کنار هم بمانند. دریافت تکی برای جایگزین‌کردن یک دفتر در همان ساختار است، نه اجرای بدون وابستگی.</p>
<p>[[downloads/mini-gpt-project.zip|دریافت پروژه همراه با همهٔ دفترها]] · [[windows.html|آماده‌کردن Python و محیط در Windows]]</p></section>
<h2 id="setup">نصب افزونهٔ آزمایشگاه در همان محیط پروژه</h2>
<p>اگر محیط کتاب را ساخته‌اید، محیط دیگری نسازید. فرمان‌های زیر از ریشهٔ پروژه و در PowerShell اجرا می‌شوند. requirements-notebooks.txt همان requirements.txt اصلی را وارد می‌کند؛ فقط JupyterLab و Matplotlib برای اجرای دفتر و نمودار اضافه می‌شوند.</p>
<pre><code>.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-notebooks.txt
.\\.venv\\Scripts\\python.exe -m jupyterlab --notebook-dir=. --ip=127.0.0.1</code></pre>
<p>اگر هنوز محیط ندارید، ابتدا راهنمای Windows بالا را انجام دهید. راهنمای کامل همراه بسته در <code>docs/NOTEBOOKS.md</code> است. از فهرست فایل‌های Jupyter پوشهٔ notebooks را باز کنید و <code>Kernel</code> همان محیط را برگزینید؛ نخستین Cell مسیر مفسر را نشان می‌دهد.</p>
<p>برای شروع پاک، از منوی <code>Kernel</code> گزینهٔ Restart <code>Kernel</code> and Run All Cells را بزنید. خطاهای عمدی داخل try/except گرفته می‌شوند؛ خطای قرمزِ گرفته‌نشده را طبیعی فرض نکنید. دفترها Checkpoint مدل نمی‌خواهند و فایل خروجی نمی‌نویسند. خروجی اجرای خود را در نسخهٔ شخصی نگه دارید؛ پیش از ثبت نسخهٔ تألیفی در Git، Restart <code>Kernel</code> and Clear Outputs و سپس Save کنید.</p>
<p>لینک بازگشت داخل دفترها به کتابِ محلی روی پورت ۸۰۰۰ است. کتاب HTML را در پنجرهٔ ترمینال جدا سرو کنید؛ پوشهٔ کتاب در ZIP مستقل مدل نیست. همهٔ محاسبات CPU و داده‌های کوچک‌اند. سنجش مدل زبانی، کیفیت یک دستیار عمومی را تضمین نمی‌کند.</p>
<h2>توقف‌های پیشنهادی در مسیر کتاب</h2>'''
