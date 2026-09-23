"""ساخت قطعی کتاب ایستا از محتوای درس‌ها؛ بدون وابستگی خارجی."""

import argparse
import html
import hashlib
import io
import keyword
from html.parser import HTMLParser
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import tokenize
import zipfile

from book_src.checkpoints import CHECKPOINTS
from book_src.schema import PARTS
from book_src.experience import (INLINE_LABS, JOURNAL, MODE_LABELS, lab_page,
                                 lesson_mode, pipeline, stage_for)
from book_src.visuals import BRIEFS, opening
from book_src.laboratories import catalog as notebook_catalog, INTRO as NOTEBOOK_INTRO
from book_src.curriculum import LESSONS, BY_ID, CHAPTERS, PATHS
from book_src.learning_time import estimate_panel, METHOD as TIME_METHOD

LABS = notebook_catalog()
from book_src.glossary import TERMS, SUPPLEMENTAL, extend_from_lessons
from book_src.terminology import (annotate_html, normalize_text, configure_terms,
                                  inline_code_html, typography_html)
from book_src.windows import WINDOWS

ROOT = Path(__file__).resolve().parents[1]
# Deterministic cache key: a rebuilt book must not keep an older unit manifest.
ASSET_VERSION = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'book_src').rglob('*'))
    if p.is_file() and p.suffix in {'.py','.css','.js'}) + (ROOT/'data/inspection-sample.json').read_bytes()).hexdigest()[:12]
OUT = ROOT / "dist"
extend_from_lessons(LESSONS)
configure_terms()
UNITS = []
for part in range(1, len(PARTS) + 1):
    UNITS.extend((l.id, PATHS[l.id], l.title) for l in LESSONS if l.part == part)
    UNITS.append((f"checkpoint-{part:02}", f"part-{part:02}/checkpoint.html", f"ایستگاه عملی {part}"))


def escaped(value):
    return html.escape(str(value), quote=True)


def display_title(value):
    return escaped(value).replace('Mini-GPT', '<bdi dir="ltr" class="nowrap">Mini-GPT</bdi>').replace('Python', '<bdi dir="ltr">Python</bdi>')


class MathDirection(HTMLParser):
    """Keep inline ASCII arrays/shape tuples readable inside Persian paragraphs."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())
        if tag in ('code', 'pre'):
            self.skip += 1

    def handle_endtag(self, tag):
        self.parts.append(f'</{tag}>')
        if tag in ('code', 'pre'):
            self.skip -= 1

    def handle_data(self, data):
        pattern = r'(?<![A-Za-z0-9_.)\]])(?:\b[A-Za-z][A-Za-z0-9_]*=)?(?:\[[A-Za-z0-9_.,\[\] +*/=-]+\]|\([A-Za-z0-9_=,/*× .-]+\))'
        if not self.skip:
            data = re.sub(pattern, lambda m: '<bdi dir="ltr">'+m[0]+'</bdi>', data)
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append('&'+name+';')

    def handle_charref(self, name):
        self.parts.append('&#'+name+';')


def relative(current, target):
    return os.path.relpath(OUT / target, (OUT / current).parent).replace("\\", "/")


def link(current, target, label, unit=""):
    current_attr = ' aria-current="page"' if current == target else ""
    unit_attr = f' data-unit="{escaped(unit)}"' if unit else ""
    navigation_attr = ' data-book-link' if target.endswith('.html') else ''
    return (f'<a href="{escaped(relative(current, target))}"{navigation_attr}'
            f'{current_attr}{unit_attr}>{escaped(label)}</a>')


def lesson_link(current, lesson_id):
    return link(current, PATHS[lesson_id], BY_ID[lesson_id].title, lesson_id)


def sidebar(current, part):
    result = ['<nav class="atlas" aria-label="اطلس مسیر"><details><summary>فهرست درس‌ها</summary><div class="atlas-content"><label>پیداکردن درس <input type="search" data-lesson-search placeholder="مثلاً بلوک یا mask"></label><div data-search-results aria-live="polite"></div><div class="atlas-grid">']
    for n, (title, _) in enumerate(PARTS, 1):
        result.append(f'<details><summary>بخش {n} · {escaped(title)}</summary>')
        result.append(link(current, f"part-{n:02}/index.html", "صفحهٔ بخش"))
        result.append("<ol>")
        for (p, chapter), (path, _) in CHAPTERS.items():
            if p == n:
                result.append(f"<li>{link(current, path, chapter)}</li>")
        result.append(f'<li>{link(current, f"part-{n:02}/checkpoint.html", "ایستگاه عملی", f"checkpoint-{n:02}")}</li></ol></details>')
    result.append('</div></div></details></nav>')
    return "".join(result)


def resolve(text, current):
    return re.sub(r'\[\[([a-zA-Z0-9][a-zA-Z0-9/_\-.]*)(?:\|([^]]+))?\]\]',
                  lambda m: link(current,PATHS.get(m[1],m[1]),m[2] or BY_ID[m[1]].title), text)


def page(current, title, body, *, part=0, crumbs=(), unit="", kind="reference", own_term=None):
    trail = [link(current, "index.html", "خانه")]
    trail.extend(link(current, target, label) for target, label in crumbs)
    trail.append(f'<span aria-current="page">{escaped(title)}</span>')
    progress = (f'<details class="progress-panel"><summary>پیشرفت مسیر <span data-progress-count>0 / {len(UNITS)}</span></summary><div class="progress-tools"><progress value="0" max="{len(UNITS)}" aria-label="درس‌ها و ایستگاه‌های انجام‌شده"></progress>'
                '<button type="button" data-export>خروجی پیشرفت</button>'
                '<label class="import-label">ورود پیشرفت <input type="file" accept=".json,application/json" data-import></label></div>'
                '<details data-export-preview hidden><summary>نسخهٔ متنی پیشرفت؛ اگر دانلود انجام نشد، این JSON را کپی و در فایل نگه دارید</summary><pre><code data-export-code></code></pre></details>'
                '<p class="progress-note">علامت انجام فقط با انتخاب شما ثبت می‌شود. برای نگهداری مطمئن یا انتقال به مرورگر دیگر، خروجی پیشرفت بگیرید. دفتر آزمایش جداست.</p>'
                '<p class="status" role="status" aria-live="polite" data-status></p></details>')
    completion = (f'<button class="done-button" data-complete="{escaped(unit)}" aria-pressed="false">'
                  'تمرین و خودسنجی را انجام دادم</button>') if unit else ""
    body = inline_code_html(resolve(body, current))
    body = re.sub(r"<table>(.*?)</table>", r'<div class="table-wrap"><table>\1</table></div>', body, flags=re.S)
    body = typography_html(body)
    mode = lesson_mode(unit) if kind == 'lesson' else kind
    stage, stage_title = stage_for(unit) if kind == 'lesson' else (None, '')
    order = next((i+1 for i,l in enumerate(LESSONS) if l.id == unit), None)
    stage_label = f'مرحلهٔ {stage} · {stage_title}' if stage is not None else stage_title
    context = (f'<details class="lesson-context"><summary>جای این درس در پروژه <span>{stage_label}</span></summary><div><p>{MODE_LABELS[mode]} · شناسهٔ ثابت: <bdi dir="ltr">{unit}</bdi></p>{link(current,"project.html","بازگشت به نقشهٔ ساخت")}</div></details>') if kind == 'lesson' else ''
    theme = BRIEFS[unit][0] if kind == 'lesson' else {1:'foundations',2:'math',3:'code',4:'language',5:'attention',6:'architecture',7:'model',8:'training',9:'debug',10:'research'}.get(part,'model')
    masthead = opening(BY_ID[unit], order, len(LESSONS)) if kind == 'lesson' else f'<header class="page-masthead"><p class="eyebrow">از Python تا Mini-GPT</p><h1>{display_title(title)}</h1></header>'
    time_panel = ''
    if kind == 'lesson':
        time_panel = estimate_panel([BY_ID[unit]], detailed=True)
    elif kind == 'part':
        time_panel = estimate_panel([l for l in LESSONS if l.part == part], checkpoints=[part])
    elif kind == 'chapter':
        chapter_lessons = next(ls for path, ls in CHAPTERS.values() if path == current)
        time_panel = estimate_panel(chapter_lessons)
    elif kind == 'checkpoint':
        time_panel = estimate_panel([], checkpoints=[part])
    elif current == 'index.html':
        time_panel = estimate_panel(LESSONS, checkpoints=range(1, len(PARTS)+1))
    time_panel = typography_html(resolve(time_panel, current))
    journal_target = relative(current, 'journal.html') + (f'?lesson={unit}' if unit else '')
    scripts = '<script src="'+relative(current,'assets/experience.js')+'?v='+ASSET_VERSION+'" defer></script>'
    if current == 'lab.html':
        scripts = '<script src="assets/inspection-demo.js?v='+ASSET_VERSION+'" defer></script>' + scripts
    if current == 'journal.html':
        scripts += '<script src="assets/journal.js?v='+ASSET_VERSION+'" defer></script>'
    document = f'''<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped(title)} | از Python تا Mini-GPT</title><meta name="description" content="کتاب فارسیِ ساخت تدریجی یک مدل زبان؛ درس، آزمایش، کد و خودسنجی.">
<link rel="stylesheet" href="{relative(current, 'assets/book.css')+'?v='+ASSET_VERSION}">
<link rel="preload" href="{relative(current, 'assets/fonts/Vazirmatn.woff2')}" as="font" type="font/woff2" crossorigin>
<script src="{relative(current, 'assets/manifest.js')+'?v='+ASSET_VERSION}" defer></script><script src="{relative(current, 'assets/book.js')+'?v='+ASSET_VERSION}" defer></script>{scripts}</head>
<body data-page-kind="{kind}" data-theme="{theme}" data-mode="{mode}" data-unit-id="{escaped(unit)}" data-root="{relative(current,'index.html')}"><a class="skip" href="#main">رفتن به متن</a>
<header class="topbar"><a class="brand" href="{relative(current,'index.html')}" data-book-link>از <bdi dir="ltr">Python</bdi> تا <bdi dir="ltr" class="nowrap">Mini-GPT</bdi></a>
<div class="header-actions">{sidebar(current,part)}<details class="book-tools"><summary>میز کار</summary><nav class="toplinks" aria-label="فضاهای کتاب">{link(current,'guide.html','راهنمای آغاز')}{link(current,'lab.html','بازرس مدل')}{link(current,'notebooks.html','آزمایشگاه‌های Jupyter')}<a href="{journal_target}" data-book-link>دفتر آزمایش</a>{link(current,'project.html','پروژه')}{link(current,'glossary.html','واژه‌ها')}{link(current,'api.html','مرجع کد')}</nav></details></div></header>
<div class="layout"><main id="main"><nav class="breadcrumbs" aria-label="مسیر صفحه">{' <span aria-hidden="true">/</span> '.join(trail)}</nav>
{masthead}{time_panel}{context}<div class="page-content">{body}</div><div class="reading-finish">{completion}<a href="{journal_target}" data-book-link>ثبت در دفتر آزمایش</a></div>{progress}<footer class="footer"><span>از یک نشانه، تا یک مدل؛ از مدل، تا یک سامانه.</span>{link(current,'guide.html','راهنما')} · {link(current,'api.html','کد و منابع')}</footer></main></div>
<noscript><p>خواندن و ناوبری بدون JavaScript هم کار می‌کند؛ کپی خودکار و ثبت پیشرفت به آن نیاز دارند.</p></noscript></body></html>'''
    document = annotate_html(document, current, own_term)
    destination = OUT / current
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def continuation(current, following, context, *, previous=None, parent=None, position=''):
    next_link = link(current, following[1], following[2], following[0]).replace('<a ', '<a rel="next" data-next-step ')
    nearby = ''
    if previous:
        nearby += link(current, previous[1], 'جلسهٔ قبل: '+previous[2], previous[0]).replace('<a ', '<a rel="prev" ')
    if parent:
        nearby += link(current, parent[0], parent[1]).replace('<a ', '<a data-return-section ')
    return (f'<nav class="pager learning-continuation" aria-label="ادامهٔ مسیر یادگیری">'
            f'<div class="next-step"><p class="eyebrow">گام بعد <span>{position}</span></p>'
            f'<p class="next-context">{context}</p>{next_link}</div>'
            f'<div class="nearby-steps">{nearby}</div></nav>')


def pager(current, unit):
    index = next(i for i, item in enumerate(UNITS) if item[0] == unit)
    previous = UNITS[index - 1] if index else ("", "guide.html", "راهنمای آغاز")
    following = UNITS[index + 1] if index + 1 < len(UNITS) else ("", "project.html", "پروژهٔ شخصی بعدی")
    if following[0] in BY_ID:
        context = 'در جلسهٔ بعد: ' + BY_ID[following[0]].objective
    elif following[0].startswith('checkpoint-'):
        context = 'حالا قطعه‌های این بخش را کنار هم بگذارید. در ایستگاه بعد، با یک کار عملی و معیار تحویل مشخص فهم خودتان را می‌سنجید.'
    else:
        context = 'مسیر اصلی تمام شد؛ یک تغییر کوچک برای پروژه انتخاب کنید، پیش‌بینی بنویسید و نتیجه را با اجرای مرجع بسنجید.'
    if unit in BY_ID:
        lesson = BY_ID[unit]
        parent = (CHAPTERS[(lesson.part, lesson.chapter)][0], 'بازگشت به فصل: '+lesson.chapter)
    else:
        parent = (f'part-{int(unit[-2:]):02}/index.html', 'بازگشت به این بخش')
    return continuation(current, following, context, previous=previous, parent=parent,
                        position=f'واحد {index+1} از {len(UNITS)} در مسیر اصلی')


def code_block(code, language='python'):
    source = code.strip()
    formatted = escaped(source)
    if language == 'python':
        try:
            lines = source.splitlines(keepends=True)
            offsets = [0]
            for line in lines:
                offsets.append(offsets[-1] + len(line))
            cursor, pieces = 0, []
            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                style = ('keyword' if tok.type == tokenize.NAME and keyword.iskeyword(tok.string)
                         else {tokenize.STRING:'string', tokenize.COMMENT:'comment', tokenize.NUMBER:'number'}.get(tok.type))
                if not style:
                    continue
                start = offsets[tok.start[0]-1] + tok.start[1]
                end = offsets[tok.end[0]-1] + tok.end[1]
                pieces.extend([escaped(source[cursor:start]), f'<span class="syntax-{style}">', escaped(source[start:end]), '</span>'])
                cursor = end
            formatted = ''.join(pieces) + escaped(source[cursor:])
        except (tokenize.TokenError, IndentationError):
            pass  # A command or intentionally incomplete example stays verbatim.
    return '<pre data-code-language="'+language+'"><code>' + formatted + '</code></pre>'


def render_lessons():
    primary = {lab['primary_lesson']: lab for lab in LABS if lab['kind'] == 'lesson'}
    if (set(primary) != set(BY_ID) or len(primary) != sum(lab['kind'] == 'lesson' for lab in LABS)
            or any(lab['lessons'] != [identifier] or lab['html'] != PATHS[identifier]
                   for identifier, lab in primary.items())):
        raise ValueError('Each lesson needs exactly one correctly mapped primary notebook')
    for lesson in LESSONS:
        path = PATHS[lesson.id]
        chapter_path, _ = CHAPTERS[(lesson.part, lesson.chapter)]
        crumbs = [(f"part-{lesson.part:02}/index.html", f"بخش {lesson.part}"), (chapter_path, lesson.chapter)]
        body = '<article class="lesson-prose">' + lesson.body + '</article>'
        if lesson.id in primary:
            lab = primary[lesson.id]
            body += ('<aside class="notebook-link lesson-lab" aria-label="آزمایشگاه این جلسه">'
                     '<span class="eyebrow">نیمهٔ عملی همین درس · شما کد می‌نویسید</span>'
                     '<h2>آزمایشگاه این جلسه</h2><p>'+lab['goal']+'</p><p>'+lab['transition']+'</p>'
                     '<p>'+link(path, 'launch.html?lesson='+lesson.id, 'باز کردن آزمایشگاه Jupyter ←')+
                     ' · '+link(path, 'notebooks.html#'+lab['id'], 'راهنما و دریافت دفتر')+'</p></aside>')
        if lesson.id in INLINE_LABS:
            body += INLINE_LABS[lesson.id]()
        body += f'<section class="practice"><div class="practice-heading"><p class="eyebrow">از خواندن به آزمودن</p><h2>حالا نوبت شماست.</h2><ol class="experiment-steps"><li>پیش‌بینی</li><li>اجرا</li><li>مشاهده</li><li>توضیح</li></ol></div><div class="practice-work"><section class="task"><h3>مسئلهٔ این درس</h3><p>{lesson.task}</p>'
        body += f'<p>{link(path, "answers/" + lesson.id + ".html", "پس از کوشش: پاسخ و اجرای مرجع")}</p></section>'
        body += f'<details class="check"><summary>خودسنجی · آیا می‌توانم توضیح بدهم؟</summary><p>{lesson.check}</p></details></div></section>'
        body += f'<details class="project"><summary>این ایده کجای Mini-GPT است؟</summary><p>{lesson.project}</p></details>'
        if lesson.review:
            body += f'<p class="note">{lesson.review}</p>'
        for lab in LABS:
            if lab['kind'] == 'review' and lesson.id in lab['lessons']:
                ready = lab['lessons'][-1]
                timing = ('اکنون می‌توانید این ایده را در دفتر مستقل آزمایش کنید.' if lesson.id == ready else
                          'این آزمایش را پس از '+lesson_link(path,ready)+' اجرا کنید؛ فعلاً مسیر اصلی کتاب را ادامه دهید.')
                body += '<aside class="notebook-link"><span class="eyebrow">آزمایشگاه مرور چند درس · اختیاری</span><h2>' + link(path, 'notebooks.html#'+lab['id'], lab['title']) + '</h2><p>'+lab['goal']+'</p><p>'+timing+'</p></aside>'
        body += pager(path, lesson.id)
        page(path, lesson.title, body, part=lesson.part, crumbs=crumbs, unit=lesson.id, kind="lesson")
        answer_path = f"answers/{lesson.id}.html"
        answer = '<p class="answers-warning">این صفحه را بعد از نوشتن پیش‌بینی و تلاش خودتان بخوانید. تفاوت پاسخ خود و مرجع را در دفتر ثبت کنید.</p>'
        answer += f'<h2>صورت تکلیف</h2><p>{lesson.task}</p><h2>مقایسه با مرجع</h2><p>{lesson.answer}</p>'
        if lesson.code:
            answer += f'<h2>اجرای مرجع</h2><p class="meta">{escaped(lesson.code_kind)}. فرمان را از ریشهٔ پروژه اجرا کنید. نام تازه‌ای دیدید؟ {link(answer_path,"api.html","راهنمای APIها و قرارداد نام‌ها")} را ببینید.</p>' + code_block(lesson.code)
        if lesson.source:
            source_path = "code/" + lesson.source + ".html"
            answer += f'<p>قطعهٔ مرتبط: {link(answer_path,source_path,lesson.source)}. فایل نهایی بخش‌های جلسه‌های بعد را هم دارد؛ خواندن کامل آن اکنون پیش‌نیاز نیست.</p>'
        if lesson.id in primary:
            from book_src.lab_exercises import exercises
            spec = exercises()[lesson.id]
            answer += '<section id="lab-solution"><h2>راه‌حل مرجع آزمایشگاه</h2><p>ابتدا TODOها را خودتان کامل کنید. این تعریف‌ها جایگزین تابع‌های ناتمام همان دفتر می‌شوند؛ مثال و ورودی‌ها در دفتر هستند.</p>'
            answer += '<h3>تمرین</h3>'+code_block(spec['solution'])+'<h3>اصلاح خرابی</h3>'+code_block(spec['fix_solution'])+'</section>'
        answer += f'<p>{lesson_link(answer_path,lesson.id)} · {link(answer_path,"project.html","دریافت پروژه و دستور اجرا")}</p>'
        answer += pager(answer_path, lesson.id)
        page(answer_path, "پاسخ: " + lesson.title, answer, part=lesson.part, crumbs=crumbs, kind='answer')


def render_structure():
    toc = []
    for n, ((part_title, description), checkpoint) in enumerate(zip(PARTS, CHECKPOINTS), 1):
        path = f"part-{n:02}/index.html"
        part_body = f'<p class="objective">{description}</p><p>زمان هر درس شامل کار دفتر هم هست. درس طولانی را در دو نشست انجام دهید؛ پس از کار عملی، یک روز فاصله بدهید و دوباره توضیح دهید.</p>'
        home_section = f'<details class="toc-part"><summary>بخش {n} · {part_title}</summary><p class="meta">{description}</p><p>{link("index.html",path,"ورود به این بخش")}</p>'
        for (part, title), (chapter_path, lessons) in CHAPTERS.items():
            if part != n:
                continue
            def items(current):
                return '<ol>' + ''.join(f'<li>{lesson_link(current,l.id)}</li>' for l in lessons) + '</ol>'
            part_body += f'<h2>{link(path,chapter_path,title)}</h2>' + items(path)
            home_section += f'<p class="route-chapter">{link("index.html",chapter_path,title)}</p>'
            chapter_body = '<p>این فصل را یک‌جا تمام نکنید. درس‌ها را به‌ترتیب بخوانید و پیش از پاسخ مرجع، چیزی بنویسید یا اجرا کنید.</p>' + items(chapter_path)
            chapter_body += '<h2>کارهایی که در این فصل انجام می‌دهید</h2><ul>'
            chapter_body += ''.join(f'<li>{l.task}</li>' for l in lessons) + '</ul>'
            chapter_body += '<p>کار چهارسطحیِ مفهومی، محاسباتی، ساخت و پژوهش با معیار تحویل مشخص در ' + link(chapter_path,f"part-{n:02}/checkpoint.html","ایستگاه این بخش") + ' منتظر شماست؛ پس از درس‌های بخش سراغ آن بروید.</p>'
            start = lessons[0]
            chapter_body += continuation(chapter_path, (start.id, PATHS[start.id], 'شروع فصل: '+start.title),
                                        start.objective, parent=(path, 'بازگشت به بخش'), position=f'بخش {n} از {len(PARTS)}')
            page(chapter_path,title,chapter_body,part=n,crumbs=[(path,f"بخش {n}")],kind="chapter")
        part_body += '<h2>پیش از بخش بعد</h2>' + link(path,f"part-{n:02}/checkpoint.html",checkpoint[0],f"checkpoint-{n:02}")
        start = next(l for l in LESSONS if l.part == n)
        part_body += continuation(path, (start.id, PATHS[start.id], 'شروع این بخش: '+start.title),
                                  start.objective, parent=('index.html#route','نقشهٔ کامل مسیر'), position=f'بخش {n} از {len(PARTS)}')
        page(path, f"بخش {n} · {part_title}",part_body,part=n,kind="part")
        home_section += f'<p>{link("index.html",f"part-{n:02}/checkpoint.html",checkpoint[0],f"checkpoint-{n:02}")}</p></details>'
        toc.append(home_section)
        cp_path = f"part-{n:02}/checkpoint.html"
        cp_title, revisit, tasks, criterion, answer = checkpoint
        cp_body = '<p class="objective">پایان این بخش، زمان آزمودن فهم است؛ نه صرفاً علامت‌زدن صفحه‌ها. برای این کار یک یا دو نشست مستقل بگذارید.</p>'
        cp_body += '<ol>' + ''.join(f'<li>{task}</li>' for task in tasks) + '</ol>'
        cp_body += f'<section class="check"><h2>معیار عبور</h2><p>{criterion}</p></section>'
        cp_body += '<p>اگر گیر کردید، از ' + lesson_link(cp_path,revisit) + ' برگردید و فقط قطعهٔ مبهم را بازسازی کنید.</p>'
        cp_body += '<p>' + link(cp_path,f"answers/checkpoint-{n:02}.html","پس از تحویل: راهنمای سنجش") + '</p>' + pager(cp_path,f"checkpoint-{n:02}")
        page(cp_path, f"ایستگاه {n} · {cp_title}",cp_body,part=n,crumbs=[(path,f"بخش {n}")],unit=f"checkpoint-{n:02}",kind="checkpoint")
        answer_path = f"answers/checkpoint-{n:02}.html"
        page(answer_path,f"راهنمای ایستگاه {n}",f'<p>{answer}</p><p>{criterion}</p><p>{link(answer_path,cp_path,"بازگشت به تکلیف")}</p>'+pager(answer_path,f'checkpoint-{n:02}'),part=n,kind='answer')
    intro = f'''<div class="home-opening"><p class="objective">جعبه را باز کنیم.<br>از اولین حدس تا مدل و سامانه‌ای که خودمان می‌سازیم.</p><p>برای کسی که Python می‌نویسد و می‌خواهد بفهمد پشت ادامهٔ یک جمله و پاسخ یک دستیار چه اتفاقی می‌افتد. عددها را دنبال می‌کنیم، کد می‌سازیم، گاهی عمداً خرابش می‌کنیم و برای هر نتیجه شاهد می‌آوریم.</p><div class="book-stats"><span><b>{len(LESSONS)}</b>درس</span><span><b>{len(PARTS)}</b>ایستگاه عملی</span><span><b>۲۳</b>مرحلهٔ هستهٔ مدل</span><span><b>۱</b>پروژهٔ پیوسته</span></div></div>'''
    intro += f'<p>{link("index.html","guide.html","از اینجا آغاز کنید: روش کار و نصب")} · {lesson_link("index.html","01-model")} · {link("index.html","project.html","نقشهٔ نسخه‌های پروژه")}</p>'
    intro += '<p>'+link("index.html","notebooks.html",f"{len(LESSONS)} آزمایشگاه درس‌به‌درس + ۱۲ دفتر مرور")+' · '+link('index.html','start.html','شروع محیط کتاب و Jupyter')+'</p>'
    intro += '<p><a data-resume hidden>ادامه از آخرین درس</a></p>'
    intro += '<div class="three-spaces"><section><span>۰۱ / بخوان و بساز</span><h2>مسیر مطالعه</h2><p>ریاضی و PyTorch از نخستین نیاز معرفی می‌شوند. پاسخ‌ها جدا هستند تا جا برای فکرکردن بماند.</p></section><section><span>۰۲ / دست‌کاری کن</span><h2>'+link('index.html','lab.html','آزمایشگاه')+'</h2><p>Mask را خاموش کنید، دما را تغییر دهید و عددهای واقعیِ مدل خودتان را بررسی کنید.</p></section><section><span>۰۳ / شاهد نگه دار</span><h2>'+link('index.html','journal.html','دفتر آزمایش')+'</h2><p>حدس، تنظیمات، خطا و کشف را ثبت کنید؛ محلی، قابل خروجی، بدون حساب.</p></section></div>'
    intro += '<h2 id="route">اطلس مسیر یادگیری</h2><p>دانش قبلی یادگیری عمیق لازم نیست. بر اساس زمان بخش‌ها و فرصت هفتگی خود برنامه بریزید؛ معیار عبور، توان توضیح و تغییر کد است.</p><div class="route-grid">' + ''.join(toc) + '</div>'
    intro += '<details class="model-map-disclosure"><summary>نمای کامل مسیر یک نشانه در GPT</summary>'+pipeline()+'</details>'
    page("index.html","از Python تا ساخت Mini-GPT",intro,kind="home")


def render_glossary():
    glossary = '<p class="objective">نام فنی را همان‌طور یاد بگیرید که در کد و مقاله می‌بینید؛ معنی‌اش را با مثال و در مسیر ساخت مدل دنبال کنید.</p><p>در متن درس، واژه‌های دارای خط نقطه‌چینِ کم‌رنگ به این توضیح‌ها می‌رسند. در هر مدخل می‌توانید به همان جای مطالعه برگردید.</p><label class="glossary-search">پیداکردن مفهوم <input type="search" data-glossary-search placeholder="مثلاً Embedding یا Gradient"></label><p data-glossary-count class="meta" role="status"></p><h2>مفهوم‌های اصلی</h2><dl class="concept-index">'
    supplemental_index = []
    for slug, term in sorted(TERMS.items(), key=lambda item:item[1].name.casefold()):
        target = f'glossary/{slug}.html'
        overview = term.meaning if slug not in SUPPLEMENTAL else BY_ID[term.lessons.split()[0]].objective
        item = f'<div data-concept-entry data-concept-search="{escaped(term.name+" "+term.aliases)}"><dt>{link("glossary.html",target,term.name)}</dt><dd>{overview}</dd></div>'
        if slug in SUPPLEMENTAL:
            supplemental_index.append(item)
        else:
            glossary += item
        body = '<article class="glossary-article">'
        first = term.lessons.split()[0]
        body += f'<p class="glossary-return"><a data-glossary-return href="{relative(target,PATHS[first])}">بازگشت به درس معرفی این مفهوم</a></p>'
        body += f'<p class="objective">{term.meaning}</p><h2>چرا به آن نیاز داریم؟</h2><p>{term.intuition}</p>'
        example_title = 'تمرین و مثال در زمینهٔ درس' if slug in SUPPLEMENTAL else 'یک مثال کوچک'
        body += f'<h2>دقیق‌تر نگاه کنیم</h2><div class="concept-context">{term.technical}</div><h2>{example_title}</h2><div class="concept-example">{term.example}</div>'
        body += f'<h2>در Mini-GPT</h2><p>{term.project}</p><h2>مفهوم‌های مرتبط</h2><ul class="glossary-related">'
        for related in term.related.split():
            body += '<li>'+link(target,f'glossary/{related}.html',TERMS[related].name)+'</li>'
        body += '</ul><h2>از آشنایی تا اجرا</h2><ol class="concept-journey">'
        for i, lesson_id in enumerate(term.lessons.split()):
            label = ['آشنایی در یک مسئله','محاسبه و ساخت','بررسی در مدل کامل'][min(i,2)]
            body += f'<li><span>{label}</span> {lesson_link(target,lesson_id)}</li>'
        body += '</ol><noscript><p>برای برگشت دقیق به صفحه و محل قبلی، از دکمهٔ Back مرورگر استفاده کنید.</p></noscript></article>'
        if slug in SUPPLEMENTAL:
            body += '<p class="note">توضیح، مثال و ارتباط با پروژه در این مدخل از '+lesson_link(target,first)+' آمده است. برای اجرای کامل مثال، پاسخ همان درس را ببینید.</p>'
        body += continuation(target,(first,PATHS[first],'این مفهوم را در درس ببینید: '+BY_ID[first].title),
                             BY_ID[first].objective,parent=('glossary.html','همهٔ مفهوم‌ها'))
        page(target,term.name,body,kind='glossary',crumbs=[('glossary.html','واژه‌نامهٔ فنی')],own_term=slug)
    glossary += '</dl><details class="glossary-more"><summary>تعریف‌های بیشتر از سراسر کتاب</summary><dl class="concept-index">'+''.join(supplemental_index)+'</dl></details>' + continuation('glossary.html',('01-model',PATHS['01-model'],'شروع مسیر: '+BY_ID['01-model'].title),
                                      'اگر هنوز درس‌ها را شروع نکرده‌اید، از یک مدل تک‌پارامتری آغاز کنید.',parent=('index.html#route','نقشهٔ مسیر'))
    page("glossary.html","مفهوم‌ها به هم وصل‌اند",glossary,kind='glossary-index')


def render_references():
    render_glossary()
    notebook_body = NOTEBOOK_INTRO.replace('{lesson_count}', str(len(LESSONS)))
    time_body = TIME_METHOD + estimate_panel(LESSONS, checkpoints=range(1, len(PARTS)+1))
    time_body += '<h2>برنامهٔ بخش‌به‌بخش</h2>'
    for part, (title, _) in enumerate(PARTS, 1):
        time_body += '<h3>'+link('learning-time.html', f'part-{part:02}/index.html', title)+'</h3>'
        time_body += estimate_panel([l for l in LESSONS if l.part == part], checkpoints=[part])
    time_body += continuation('learning-time.html', ('01-model', PATHS['01-model'], 'آغاز مسیر'),
                              'زمان واقعی سه درس اول را ثبت کنید و برنامه را با تجربهٔ خود تنظیم کنید.',
                              parent=('guide.html', 'راهنمای مطالعه'))
    page('learning-time.html', 'زمان یادگیری، نه فقط زمان خواندن', time_body)
    for lab in LABS:
        label = ('درس '+str(lab['lesson_number'])+' · دفتر تمرین' if lab['kind'] == 'lesson' else 'مرور چند درس · اختیاری')
        target = 'launch.html?lesson='+lab['primary_lesson'] if lab['kind'] == 'lesson' else 'launch.html?review='+lab['id']
        notebook_body += '<section class="notebook-link" id="'+lab['id']+'"><p class="eyebrow">'+label+'</p><h2>'+lab['title']+'</h2><p>'+lab['goal']+'</p><p>درس‌های مرتبط، به ترتیب مطالعه: '+ ' · '.join(lesson_link('notebooks.html', identifier) for identifier in lab['lessons'])+'</p><p>'+link('notebooks.html',target,'باز کردن آزمایشگاه Jupyter ←')+' · <a download href="'+lab['path']+'">دریافت این دفتر (.ipynb)</a> · <a href="#setup">روش بازکردن و اجرا</a></p></section>'
    from book_src.lab_exercises.reviews import EXERCISES as review_specs
    for lab in LABS:
        if lab['kind'] != 'review':
            continue
        spec = review_specs[lab['id']]
        target = 'answers/lab-'+lab['id']+'.html'
        content = '<p class="answers-warning">ابتدا دو تابع TODO دفتر مرور را خودتان کامل کنید؛ این پاسخ‌ها جایگزین همان تابع‌ها هستند.</p>'
        content += '<h2>تمرین</h2><p>'+spec['task']+'</p>'+code_block(spec['solution'])
        content += '<h2>اصلاح خرابی</h2><p>'+spec['debug']+'</p>'+code_block(spec['fix_solution'])
        content += continuation(target,(lab['lessons'][-1],PATHS[lab['lessons'][-1]],'بازگشت به درس مرتبط'),spec['goal'],parent=('notebooks.html#'+lab['id'],'دفتر مرور'))
        page(target,'پاسخ دفتر مرور: '+lab['title'],content,kind='answer')
    page('notebooks.html','آزمایشگاه‌های Jupyter در مسیر کتاب',resolve(notebook_body,'notebooks.html')+continuation('notebooks.html',('01-model',PATHS['01-model'],'نخستین آزمایش: زمان خواندن'),BY_ID['01-model'].objective,parent=('index.html#route','مسیر اصلی کتاب')),kind='laboratory')
    start = '<p class="objective">یک درس بخوانید، همان ایده را در Python بسازید و نتیجه را به کتاب برگردانید.</p><div class="three-spaces"><section><h2>'+link('start.html','index.html','📖 باز کردن کتاب')+'</h2><p>'+lesson_link('start.html','01-model')+'</p></section><section><h2>'+link('start.html','launch.html','🧪 باز کردن Jupyter')+'</h2><p>'+link('start.html','launch.html?lesson=01-model','اولین آزمایشگاه: زمان خواندن')+'</p></section><section><h2>'+link('start.html','notebooks.html','فهرست آزمایشگاه‌ها')+'</h2><p>'+link('start.html','index.html#route','فهرست درس‌ها')+'</p></section></div><p>'+link('start.html','windows.html','نصب و راهنمای شروع')+'</p><p>این صفحه با <code>python run.py</code> به هر دو سرویس محلی وصل می‌شود. تا پایان کار، ترمینال را باز نگه دارید. برای خروج، Ctrl+C هر دو سرویس را متوقف می‌کند.</p>'
    page('start.html','میز کار یادگیری',start)
    launch = '<p class="objective">برای نوشتن و اجرای Python، محیط محلی را باز کنید.</p><p>این نسخهٔ ایستا خودش Jupyter اجرا نمی‌کند. پروژهٔ کامل را استخراج کنید، وابستگی‌ها را یک بار نصب کنید و از ریشهٔ پروژه فرمان زیر را اجرا کنید.</p><pre><code>python run.py</code></pre><p>سپس کتاب را روی <a href="http://127.0.0.1:8000/start.html">میز کار محلی</a> باز کنید و دکمهٔ آزمایشگاه همان درس را بزنید؛ دفتر درست مستقیم باز می‌شود.</p><p>'+link('launch.html','windows.html','نصب یک‌باره در Windows')+' · '+link('launch.html','downloads/mini-gpt-project.zip','دریافت پروژهٔ کامل یادگیری')+' · '+link('launch.html','notebooks.html','فهرست همهٔ دفترها')+'</p>'
    page('launch.html','باز کردن آزمایشگاه محلی',launch)
    from book_src.references import GUIDE, PROJECT, API
    for path, title, body in [("guide.html","چطور با این کتاب کار کنیم؟",GUIDE),
                              ("project.html","پروژه‌ای که همراه شما رشد می‌کند",PROJECT),
                              ("api.html","راهنمای خواندن کد و APIها",API)]:
        first = {'guide.html':'01-model','project.html':'52-first-run','api.html':'13-torch'}[path]
        body += continuation(path,(first,PATHS[first],BY_ID[first].title),BY_ID[first].objective,parent=('index.html#route','نقشهٔ مسیر'))
        page(path,title,resolve(body,path))
    lab_next = continuation('lab.html',('29-scores',PATHS['29-scores'],'عددها را در درس دنبال کنید'),BY_ID['29-scores'].objective,parent=('index.html#route','نقشهٔ مسیر'))
    journal_next = continuation('journal.html',('62-journal',PATHS['62-journal'],'از یادداشت به یک آزمایش قابل بازسازی'),BY_ID['62-journal'].objective,parent=('index.html#route','نقشهٔ مسیر'))
    page('lab.html','جعبهٔ مدل را باز کنیم',lab_page()+'<p>'+link('lab.html','notebooks.html','آزمایشگاه‌های Python و Jupyter: از ماتریس تا آموزش مدل')+'</p>'+lab_next,kind='laboratory')
    page('journal.html','دفتر آزمایش شما',JOURNAL+journal_next,kind='journal')
    source_files = sorted((ROOT / "mini_gpt").rglob("*.py")) + sorted((ROOT / "mini_gpt").rglob("*.md"))
    source_index = '<p>مرجع نهایی کد؛ ترتیب مطالعه را از درس‌ها بگیرید، نه از این فهرست الفبایی. نام‌های کتابخانه در '+link('code/index.html','api.html','راهنمای کد')+' توضیح داده شده‌اند.</p><ul>'
    for source in source_files:
        source_name = source.relative_to(ROOT).as_posix()
        path = "code/" + source_name + ".html"
        source_index += '<li>'+link('code/index.html',path,source_name)+'</li>'
        related = [l for l in LESSONS if l.source == source_name]
        body = '<p>این فایل مرجع است. متن فارسی درون کد ممکن است به‌علت چیدمان چپ‌به‌راست جدا دیده شود؛ نسخهٔ دانلودی UTF-8 است.</p>'
        body += '<p>'+link(path,'api.html','توضیح نام‌ها و دستورها')+' · '+link(path,'project.html','دریافت همهٔ فایل‌ها')+'</p>'
        if related:
            body += '<p>درس‌های مرتبط: '+ ' · '.join(lesson_link(path,l.id) for l in related)+'</p>'
        body += code_block(source.read_text(encoding='utf-8'), 'python' if source.suffix == '.py' else 'text')
        first = related[0] if related else BY_ID['13-torch']
        body += continuation(path,(first.id,PATHS[first.id],'کد را در یک مسئله دنبال کنید: '+first.title),
                             first.objective,parent=('code/index.html','همهٔ فایل‌های مرجع'))
        page(path,source_name,body)
    source_index += '</ul>'+continuation('code/index.html',('', 'api.html','راهنمای خواندن کد و قرارداد نام‌ها'),
                                        'پیش از خواندن فایل کامل، قرارداد Shapeها، ورودی‌ها و خروجی‌ها را در راهنما مرور کنید.',
                                        parent=('project.html','نقشهٔ پروژه'))
    page('code/index.html','فایل‌های مرجع پروژه',source_index)


def main(output=None):
    global OUT
    OUT = Path(output).resolve() if output is not None else ROOT / 'dist'
    if len(BY_ID) != len(LESSONS) or len(CHECKPOINTS) != len(PARTS):
        raise ValueError("duplicate lesson IDs or incomplete checkpoints")
    if set(BRIEFS) != set(BY_ID):
        raise ValueError('Every lesson must have an authored visual brief.')
    assets = OUT / 'assets'
    assets.mkdir(parents=True,exist_ok=True)
    (assets/'fonts').mkdir(exist_ok=True)
    for font in (ROOT/'book_src'/'assets'/'fonts').iterdir():
        if font.suffix in ('.woff2', '.txt'):
            shutil.copyfile(font, assets/'fonts'/font.name)
    for name in ('book.css','book.js','experience.js','journal.js'):
        (assets/name).write_text((ROOT/'book_src'/'assets'/name).read_text(encoding='utf-8'),encoding='utf-8')
    sample = json.loads((ROOT/'data'/'inspection-sample.json').read_text(encoding='utf-8'))
    (assets/'inspection-demo.js').write_text('window.INSPECTION_DEMO = '+json.dumps(sample,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    for lab in LABS:
        target = OUT / lab['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / lab['path'], target)
    render_lessons()
    render_structure()
    render_references()
    windows_body = resolve(WINDOWS, 'windows.html') + continuation(
        'windows.html', ('01-model', PATHS['01-model'], 'شروع از اولین مدل'),
        BY_ID['01-model'].objective, parent=('guide.html', 'راهنمای مسیر'))
    page('windows.html', 'نصب و اجرای کتاب در Windows', windows_body)
    manifest = {'edition':3,'units':[{'id':i,'path':p,'title':normalize_text(t)} for i,p,t in UNITS],
                'laboratories':LABS,
                'concepts':{term.name:slug for slug,term in TERMS.items()},
                'pages':sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob('*.html'))}
    (assets/'manifest.js').write_text('window.BOOK = '+json.dumps(manifest,ensure_ascii=False)+';\n',encoding='utf-8')
    (OUT/'downloads').mkdir(exist_ok=True)
    packaged = [ROOT/name for name in ('run.py','requirements.txt','requirements-notebooks.txt','docs/WINDOWS_SETUP.md','docs/NOTEBOOKS.md','docs/LEARNING_TIME.md','tools/__init__.py','tools/build_book.py','tools/learning_server.py')]
    for directory in ('mini_gpt','data','notebooks','book_src'):
        packaged.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.txt','.ipynb','.json','.css','.js','.woff2') and '.ipynb_checkpoints' not in p.parts and '__pycache__' not in p.parts
                        and (directory != 'notebooks' or p.relative_to(ROOT).as_posix() in {lab['path'] for lab in LABS}))
    packaged.extend((ROOT/'tests').glob('test_*.py'))
    with zipfile.ZipFile(OUT/'downloads'/'mini-gpt-project.zip','w',zipfile.ZIP_DEFLATED) as archive:
        readme = zipfile.ZipInfo('README.md', date_time=(2026,9,19,0,0,0))
        readme.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(readme, (ROOT/'book_src/download_readme.md').read_bytes())
        for path in sorted(set(packaged)):
            # Fixed metadata makes a rebuild byte-for-byte reproducible.
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(2026,9,19,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes())
    print(f"Built {len(LESSONS)} lessons, {len(CHAPTERS)} chapters, {len(PARTS)} parts, {len(UNITS)} progress units.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Output directory (default: dist).')
    main(parser.parse_args().output)
