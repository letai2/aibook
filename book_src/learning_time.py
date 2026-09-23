"""Transparent editorial workload estimates, not measured learner statistics.

Each lesson has an explicitly reviewed activity profile. Word count only estimates
the first reading; conceptual difficulty, code tracing, writing/repair, experiments,
and prerequisite recall have separate allowances. See docs/LEARNING_TIME.md.
"""
from dataclasses import dataclass
import html
import math
import re


@dataclass(frozen=True)
class Profile:
    # Minutes before applying the conceptual difficulty multiplier.
    concept: tuple[int, int]
    code: tuple[int, int]
    lab: tuple[int, int]
    exercise: tuple[int, int]
    recall: tuple[int, int]
    difficulty: float


PROFILES = {
    'introduce': Profile((5, 10), (3, 6), (10, 18), (5, 10), (0, 5), 1.0),
    'calculate': Profile((8, 14), (5, 10), (15, 25), (8, 15), (3, 7), 1.15),
    'construct': Profile((8, 15), (8, 15), (20, 35), (10, 18), (5, 10), 1.25),
    'investigate': Profile((10, 18), (8, 15), (20, 35), (10, 20), (5, 10), 1.35),
    'integrate': Profile((12, 20), (12, 20), (30, 50), (15, 25), (8, 15), 1.5),
}

# Membership is intentional, not inferred from a title or from the lesson number.
PROFILE_IDS = {
    'introduce': '''01-model 01-learning 02-token 08-probability 14-index-device
        24-data-contract 26a-sequence-models 27-attention-why 42-families
        62b-lifecycle 63-scale 67-rag''',
    'calculate': '''05-shape 05a-vector-operations 06-dot 07-matmul 09-softmax
        10-entropy 11-derivative 12-chain 12-sgd 12b-neuron 15-broadcast
        25-embedding 28-qkv 29-scores 30-scaling 31-values 35-split-heads
        36-merge-heads 39-layernorm 43-lm-head 44-parameters 49b-schedule
        55-temperature 56-topkp 64-cache 65-sft 65b-lora 66-preference''',
    'construct': '''03-counts 13-torch 17-autograd 18-module 19-network 20-loader
        21-tokenizer 22-bpe 23-shift 32-self 33-mask 37-ffn 41-stack 47-loop
        54-generate 68-context 69-chunks 72-history 75-tools''',
    'investigate': '''04-splits 16-reshape 26-positions 26b-sequence-memory
        34-causal-test 38-residual 46-gradient-path 48-evaluate 49-rate
        53-curves 57-prompts 58-ablation 59-capacity 60-bug-clinic 61-one-batch
        62-journal 70-vectors 71-grounding 73-summary 76-reasoning 77-candidates
        82-performance''',
    'integrate': '''40-block 45-trace 50-checkpoint 51-resume 52-first-run
        65a-sft-lab 74-memory 80-controller 81-system-eval 83-deployment 84-capstone''',
}
LESSON_PROFILES = {identifier: name for name, identifiers in PROFILE_IDS.items()
                   for identifier in identifiers.split()}
if len(LESSON_PROFILES) != sum(len(ids.split()) for ids in PROFILE_IDS.values()):
    raise ValueError('Duplicate learning-time profile assignment')


def add_ranges(*values):
    return tuple(sum(v[i] for v in values) for i in (0, 1))


def rounded(value):
    return int(math.ceil(value / 5) * 5)


def estimate(lesson):
    profile = PROFILES[LESSON_PROFILES[lesson.id]]
    prose = ' '.join((lesson.objective, lesson.body, lesson.task, lesson.check))
    prose = re.sub(r'<(?:pre|code)\b[^>]*>.*?</(?:pre|code)>', '', prose, flags=re.S)
    words = len(html.unescape(re.sub(r'<[^>]+>', ' ', prose)).split())
    # Planning assumptions, not claimed Persian population reading speeds.
    reading = (max(3, math.ceil(words / 180)), max(5, math.ceil(words / 100)))
    understanding = tuple(math.ceil(x * profile.difficulty) for x in profile.concept)
    components = {'reading': reading, 'understanding': understanding,
                  'code': profile.code, 'laboratory': profile.lab,
                  'exercise': profile.exercise, 'recall': profile.recall}
    subtotal = add_ranges(*components.values())
    total = tuple(rounded(x) for x in subtotal)
    return {'minutes': total, 'components': components,
            'profile': LESSON_PROFILES[lesson.id], 'difficulty': profile.difficulty,
            'prose_words': words}


# Extra synthesis, not a second charge for the same lesson notebook work.
CHECKPOINT_MINUTES = {part: ((45, 75) if part <= 4 else (60, 100) if part <= 9
                           else (60, 105) if part < 15 else (90, 150))
                      for part in range(1, 16)}


def aggregate(lessons, *, checkpoints=()):
    values = [estimate(lesson)['minutes'] for lesson in lessons]
    values += [CHECKPOINT_MINUTES[part] for part in checkpoints]
    return add_ranges(*values)


def fa(value):
    return str(value).translate(str.maketrans('0123456789.', '۰۱۲۳۴۵۶۷۸۹٫'))


def duration(minutes, *, hours=False):
    low, high = minutes
    if hours:
        # Outward half-hour rounding avoids a false impression of minute precision.
        low, high = math.floor(low / 30) / 2, math.ceil(high / 30) / 2
        return f'{fa(f"{low:g}")}–{fa(f"{high:g}")} ساعت'
    return f'{fa(low)}–{fa(high)} دقیقه'


def estimate_panel(lessons, *, checkpoints=(), detailed=False):
    lessons = list(lessons)
    total = aggregate(lessons, checkpoints=checkpoints)
    result = (f'<aside class="learning-time" aria-label="برآورد زمان یادگیری" '
              f'data-minutes-low="{total[0]}" data-minutes-high="{total[1]}">'
              f'<p>زمان یادگیری: حدود <bdi dir="rtl">{duration(total, hours=len(lessons)>1 or bool(checkpoints))}</bdi></p>')
    if detailed and len(lessons) == 1:
        data = estimate(lessons[0])
        labels = {'reading':'خواندن اولیه', 'understanding':'فهم و محاسبهٔ دستی',
                  'code':'خواندن کد', 'laboratory':'نوشتن، آزمایش و رفع خرابی',
                  'exercise':'خودسنجی و توضیح', 'recall':'مرور پیش‌نیاز'}
        result += '<details><summary>این زمان شامل چه کارهایی است؟</summary><dl>'
        result += ''.join(f'<div><dt>{labels[key]}</dt><dd>{duration(value)}</dd></div>'
                          for key, value in data['components'].items())
        result += '</dl><p>بازهٔ کل رو به بالا به مضرب پنج دقیقه گرد شده است؛ زمان اجرای طولانی مدل و نصب جداست.</p></details>'
    else:
        result += '<p class="meta">جمع درس‌ها'+(' و زمان جداگانهٔ ایستگاه‌های عملی' if checkpoints else '')+'؛ دفترهای مرور اختیاری در این جمع نیستند.</p>'
    return result + '<p class="time-caveat">برآورد تحریریه است، نه زمان‌سنجیِ یادگیرندگان. مکث و تکرار طبیعی است. [[learning-time.html|روش برآورد و برنامه‌ریزی]]</p></aside>'


METHOD = '''<p class="objective">خواندن یک صفحه با یادگرفتن آن یک کار نیست.</p>
<p>این بازه‌ها برای کسی است که تابع، حلقه و کلاس سادهٔ Python را می‌شناسد اما تازه وارد یادگیری عمیق شده است. هر درس شامل خواندن، دنبال‌کردن مثال دستی، خواندن کد، نوشتن دو تمرین دفتر، آزمایش یک تغییر و توضیح نتیجه است. جواب مرجع جای این کارها را نمی‌گیرد.</p>
<p>تعداد واژه فقط سهم خواندن اولیه را تخمین می‌زند؛ فرض برنامه‌ریزی ما ۱۰۰ تا ۱۸۰ واژه در دقیقه است و ادعای اندازه‌گیری سرعت خواندن فارسی نیست. زمان فهم، خواندن کد، کار دفتر، خودسنجی و یادآوری پیش‌نیاز جدا اضافه می‌شوند. برای درس محاسباتی، ساخت، بررسی خطا و یکپارچه‌سازی، ضریب دشواری فهم به‌ترتیب ۱٫۱۵، ۱٫۲۵، ۱٫۳۵ و ۱٫۵ است؛ درس معرفی ضریب یک دارد. این ضرایب انتخاب تحریریه‌اند، نه ثابت‌های علمی.</p>
<p>هر درس به‌طور صریح یکی از پنج الگوی فعالیت را دارد. جمع درس‌ها زمان فصل را می‌سازد؛ زمان بخش، ایستگاه عملی همان بخش را هم حساب می‌کند. جمع بخش‌ها زمان مسیر اصلی است. تمرین دفتر در زمان درس حساب شده و دوباره در ایستگاه شمرده نمی‌شود؛ ایستگاه یک تکلیف ترکیبی تازه دارد.</p>
<p>نصب و رفع مشکل محیط را جدا برنامه‌ریزی کنید: معمولاً یک نشست ۱ تا ۳ ساعته برای نصب نخستین، بدون تضمین برای شبکه یا دستگاه شما. ۱۲ دفتر مرور اختیاری، هرکدام حدود ۴۵ تا ۹۰ دقیقهٔ افزوده‌اند. اجرای آموزش طولانی، مطالعهٔ مقاله‌ها و پروژهٔ شخصی باز در جمع اصلی نیستند.</p>
<p>برای سه درس اول، زمان واقعی خود را ثبت کنید. اگر مرتب بیرون بازه بود، برنامه را متناسب با تجربهٔ خود تنظیم کنید؛ سرعت کمتر نشانهٔ ضعف نیست. درس طولانی را در دو نشست انجام دهید و میان آن‌ها توضیحِ بدون نگاه‌کردن به پاسخ را امتحان کنید.</p>
<p>در <a href="https://cte.rice.edu/resources/workload-estimator">راهنمای برآورد بار کاری Rice University</a> نیز هدف خواندن، دشواری و نوع تکلیف از هم جدا می‌شوند. ما از این اصل استفاده کرده‌ایم، نه از نرخ‌های انگلیسی آن به‌عنوان دادهٔ معتبر برای این کتاب فارسی. برای دقیق‌ترکردن نسخهٔ بعد به مشاهدهٔ داوطلبانهٔ زمان و دشواری درس‌ها نیاز داریم؛ فعلاً چنین مطالعه‌ای انجام نشده است.</p>'''
