"""Purpose-built learning surfaces; shared HTML, no external display assets."""

MODES = {
    'calculate': {'05a-vector-operations','12b-neuron','05-shape','06-dot','07-matmul','08-probability','09-softmax','10-entropy','11-derivative','12-chain','12-sgd','15-broadcast','16-reshape','28-qkv','29-scores','30-scaling','31-values','35-split-heads','36-merge-heads','39-layernorm','43-lm-head','44-parameters','49b-schedule','56-topkp','65b-lora'},
    'workshop': {'03-counts','13-torch','17-autograd','18-module','19-network','20-loader','21-tokenizer','22-bpe','23-shift','25-embedding','37-ffn','40-block','41-stack','45-trace','47-loop','50-checkpoint','51-resume','52-first-run','54-generate'},
    'investigate': {'04-splits','24-data-contract','26-positions','32-self','33-mask','34-causal-test','38-residual','46-gradient-path','48-evaluate','49-rate','53-curves','57-prompts','58-ablation','59-capacity','60-bug-clinic','61-one-batch'},
    'reflect': {'62-journal','63-scale','64-cache','65-sft','66-preference','67-rag'},
}
MODE_LABELS = {'concept':'فهم یک ایده','calculate':'روی میز محاسبه','workshop':'در کارگاه ساخت','investigate':'پروندهٔ یک آزمایش','reflect':'یادداشت‌های پژوهش'}


def lesson_mode(identifier):
    return next((mode for mode, ids in MODES.items() if identifier in ids), 'concept')


def stage_for(identifier):
    if identifier in {'26a-sequence-models', '26b-sequence-memory'}:
        return None, 'راه‌های پردازش دنباله'
    n = int(identifier[:2]) if identifier[:2].isdigit() else 0
    if 5 <= n <= 12:
        return None, 'پیش‌نیازهای ریاضی و شبکه'
    if 13 <= n <= 20:
        return None, 'پیش‌نیازهای PyTorch'
    if 21 <= n <= 22:
        return 1, 'Tokenization'
    for limit, stage, title in [(4,0,'مدل شمارشی'),(24,2,'متن و شناسه'),(25,3,'جدول نمایش'),(26,4,'اطلاعات موقعیت'),(32,5,'توجه'),(34,6,'پوشش علّی'),(36,7,'چندسر'),(37,8,'پیش‌خور'),(38,9,'مسیر جمع'),(39,10,'نرمال‌سازی'),(40,11,'یک بلوک'),(42,12,'چند بلوک'),(46,13,'خروجی مدل'),(47,14,'حلقهٔ آموزش'),(49,15,'سنجش آموزش'),(53,16,'ذخیره و ادامه'),(54,17,'تولید'),(55,19,'دما'),(56,21,'انتخاب نشانه'),(99,22,'مدل و پژوهش شخصی')]:
        if n <= limit:
            return stage, title


def pipeline():
    return '''<section class="model-map" aria-label="نقشهٔ کامل مدل زبان">
<div class="map-heading"><span>یک گام، یک نشانه</span><span class="meta">مدل احتمال می‌سازد؛ انتخاب، مرحله‌ای جداست.</span></div>
<ol class="pipeline">
<li><span class="node-tag">۰۱ · متن</span><b>مدلِ ما</b><small>نویسه‌ها → شناسه‌ها</small><code>(B,T)</code></li>
<li><span class="node-tag">۰۲ · نمایش</span><b>نشانه + موقعیت</b><small>دو جدول یادگرفتنی</small><code>(B,T,C)</code></li>
<li><span class="node-tag">۰۳ · تبدیل</span><b>بلوک‌های علّی</b><small>خواندن گذشته + پردازش ویژگی</small><code>(B,T,C)</code></li>
<li><span class="node-tag">۰۴ · امتیاز</span><b>خروجی واژگان</b><small>نرمال‌سازی نهایی و تبدیل خطی</small><code>(B,T,V)</code></li>
<li><span class="node-tag">۰۵ · انتخاب</span><b>نشانهٔ بعدی</b><small>احتمال → نمونه‌گیری → تکرار</small><code>(B,T+1)</code></li></ol>
<p class="map-caption">در آموزش، هر موقعیت با هدفِ یک‌خانه جلوتر سنجیده می‌شود. در تولید، فقط آخرین موقعیت برای انتخاب نشانهٔ تازه به کار می‌رود.</p></section>'''


def residual_diagram():
    return '''<figure class="residual-diagram"><svg class="residual-desktop" viewBox="0 0 760 310" role="img" aria-labelledby="residual-title residual-desc"><title id="residual-title">دو مسیر جمع در بلوک Pre-Norm</title><desc id="residual-desc">ورودی به نرمال‌سازی و توجه می‌رود و همچنین مستقیم به جمع اول. نتیجه به نرمال‌سازی و پیش‌خور و نیز مستقیم به جمع دوم می‌رود.</desc>
<defs><marker id="res-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#res-arrow)"><path d="M30 140 H90"/><path d="M200 140 H245"/><path d="M365 140 H405"/><path d="M445 140 H480"/><path d="M590 140 H625"/><path d="M700 140 H741"/><path d="M55 140 V50 H425 V120"/><path d="M460 140 V245 H718 V160"/></g>
<g class="svg-box"><rect x="90" y="105" width="110" height="70" rx="8"/><rect x="245" y="105" width="120" height="70" rx="8"/><rect x="480" y="105" width="110" height="70" rx="8"/><rect x="625" y="105" width="75" height="70" rx="8"/><circle cx="425" cy="140" r="20"/><circle cx="720" cy="140" r="20"/></g>
<g text-anchor="middle" fill="currentColor" font-size="17"><text x="145" y="135">LayerNorm</text><text x="145" y="160">ویژگی‌ها</text><text x="305" y="135">توجه علّی</text><text x="305" y="160">گذشتهٔ مجاز</text><text x="535" y="135">LayerNorm</text><text x="535" y="160">ویژگی‌ها</text><text x="662" y="135">FFN</text><text x="662" y="160">پیش‌خور</text><text x="425" y="147">+</text><text x="720" y="147">+</text><text x="240" y="35">میان‌بر اول: خودِ ورودی</text><text x="590" y="275">میان‌بر دوم: نتیجهٔ جمع اول</text></g></svg>
<div class="residual-mobile"><ol><li><bdi dir="ltr">x</bdi> · ورودی بلوک</li><li>نرمال‌سازی ← توجه علّی</li><li><bdi dir="ltr">y = x + Attention(LN(x))</bdi><br>جمع با خودِ ورودی</li><li>نرمال‌سازی ← شبکهٔ پیش‌خور</li><li><bdi dir="ltr">output = y + FFN(LN(y))</bdi><br>جمع با نتیجهٔ مرحلهٔ سوم</li></ol></div>
<figcaption>در مدل ما نرمال‌سازی پیش از هر زیرلایه است. پیکان میان‌بر، مقدار را بدون آن زیرلایه به جمع می‌رساند؛ تضمینی برای مصونیت کامل از مشکل گرادیان نیست.</figcaption></figure>'''


def token_lab():
    return '''<section class="interactive" data-token-lab><p class="lab-kicker">آزمایش ۰۱ / متن زیر ذره‌بین</p><h2>یک «حرف» همیشه یک خانه نیست</h2>
<p>اینجا مانند نشانه‌بند پروژه، هر نقطه‌کد Unicode یک نشانه است؛ نه واژه، نه لزوماً یک نویسهٔ دیداری. واژگان از متن مرجع ثابت زیر ساخته می‌شود و شناسهٔ صفر برای ناشناخته است. جدول نمایش سه‌ستونیِ بعدی فقط مثال دستی است، نه وزن آموزش‌دیده.</p>
<p class="meta">متن مرجع واژگان: <bdi>مدل می‌رود. cat cats سلام!</bdi></p>
<label>متن آزمایش <input data-token-text value="مدل cat!" maxlength="80"></label><div class="control-row"><button type="button" data-token-example="می‌رود">نیم‌فاصلهٔ فارسی</button><button type="button" data-token-example="cats 🐈">انگلیسی و ناشناخته</button><button type="button" data-token-example="ي ک ی ك">ی و ک متفاوت</button></div>
<div class="token-strip" data-token-strip aria-label="نشانه‌ها و شناسه‌ها"></div><div data-token-detail class="readout" aria-live="polite"></div><p class="meta">یکی از نشانه‌ها را انتخاب کنید. شماره، معنی نشانه نیست؛ نشانی سطر جدول است. بدون نرمال‌سازیِ صریح، «ي» و «ی» یک نشانه نیستند.</p></section>'''


def softmax_lab():
    return '''<section class="interactive" data-probability-lab><p class="lab-kicker">آزمایش ۰۲ / میز احتمال</p><h2>امتیاز را با اطمینان اشتباه نگیریم</h2><p>سه نشانهٔ فرضی داریم. امتیازهای خام را تغییر دهید؛ softmax از آن‌ها توزیع می‌سازد. هدف این مثال «الف» است. «دما» عدد مثبتی است که پیش از softmax امتیازها را بر آن تقسیم می‌کنیم؛ در درس تولید دوباره به آن برمی‌گردیم. نخست حدس بزنید دما چه اثری روی احتمال هدف دارد. خطای نمایش‌داده‌شده همان منفی لگاریتم این احتمال است و در درس بعد بازش می‌کنیم.</p>
<div class="control-row"><label>امتیاز الف <input type="number" data-logit="0" value="2" min="-20" max="20" step="0.5"></label><label>امتیاز ب <input type="number" data-logit="1" value="1" min="-20" max="20" step="0.5"></label><label>امتیاز ج <input type="number" data-logit="2" value="0" min="-20" max="20" step="0.5"></label></div>
<label>دما <input type="range" data-temperature min="0.2" max="3" step="0.1" value="1"><output data-temperature-value>1</output></label>
<div data-probability-bars class="probability-bars"></div><p class="readout" data-probability-result aria-live="polite"></p><p class="meta">دما در تولید به توزیع دست می‌زند؛ وزن شبکه را آموزش نمی‌دهد. کم‌شدن خطا با انتخاب هدف دلخواه، شاهد تعمیم نیست.</p></section>'''


def attention_lab():
    return '''<section class="interactive" data-attention-lab><p class="lab-kicker">آزمایش ۰۳ / اتاق توجه</p><h2>از یک پرسش تا یک جمع وزن‌دار</h2><p>مثال دستی با سه موقعیت و دو ویژگی؛ نه وزن‌های ذخیره‌شدهٔ مدل. هر سطر یک پرسش و هر ستون یک کلید است. سطر سوم را انتخاب و پیش از تغییر پوشش حدس بزنید کدام سطرها تغییر می‌کنند. «پوشش علّی» یعنی منع دیدن موقعیت‌های بعدی؛ چند درس جلوتر آن را در کد می‌سازیم.</p>
<div class="qkv-legend"><div><b>Q · پرسش</b><code>[[1,0],[0,1],[1,1]]</code><small>دنبال چه ترکیبی می‌گردم؟</small></div><div><b>K · کلید</b><code>[[1,1],[2,0],[0,1]]</code><small>با چه ویژگی‌هایی سنجیده می‌شوم؟</small></div><div><b>V · مقدار</b><code>[[1,0],[0,2],[3,1]]</code><small>چه عددهایی منتقل می‌کنم؟</small></div></div>
<div class="control-row"><label><input type="checkbox" data-causal checked> پوشش علّی روشن</label><label>سطر پرسش <select data-query><option value="0">۰ · نخستین</option><option value="1" selected>۱ · میانی</option><option value="2">۲ · آخرین</option></select></label></div>
<div class="matrix-pair"><div><h3>امتیاز مقیاس‌شده · QKᵀ / √2</h3><div data-attention-scores></div></div><div><h3>وزن پس از پوشش و softmax</h3><div data-attention-weights></div></div></div><p class="readout" data-attention-result aria-live="polite"></p><p class="meta">Query، Key و Value خروجی سه Projection آموختنی‌اند، نه سه نوع کلمه. Parameterهای Projection آموخته می‌شوند؛ Q/K/V با ورودی تازه دوباره محاسبه می‌شوند. خانهٔ ممنوع پیش از softmax برابر منفی بی‌نهایت می‌شود؛ صفرکردن امتیاز کافی نیست.</p></section>'''


def shape_lab():
    return '''<section class="interactive" data-shape-lab><p class="lab-kicker">آزمایش ۰۴ / جابه‌جایی محورها</p><h2>عددها ثابت؛ قرارداد محور عوض می‌شود</h2><p>B تعداد نمونه، T طول زمینه، C ویژگی‌ها، H سرها و D ویژگی هر سر است. این ابزار شکل‌ها و هزینهٔ یک ماتریس امتیاز float32 را حساب می‌کند، نه حافظهٔ کل آموزش.</p>
<div class="control-row"><label>B <input type="number" data-dimension="B" value="2" min="1" max="16"></label><label>T <input type="number" data-dimension="T" value="8" min="1" max="1024"></label><label>C <input type="number" data-dimension="C" value="32" min="1" max="2048"></label><label>H <input type="number" data-dimension="H" value="4" min="1" max="32"></label></div>
<div data-shape-result class="shape-route" aria-live="polite"></div><p class="meta">reshape محور تازه را معرفی می‌کند؛ transpose جای محور زمان و سر را عوض می‌کند. عوض‌کردن نام شکل بدون جابه‌جایی درست داده، چندسری نمی‌سازد.</p></section>'''


INLINE_LABS = {'02-token':token_lab,'09-softmax':softmax_lab,'33-mask':attention_lab,'35-split-heads':shape_lab,'40-block':residual_diagram,'45-trace':pipeline}

INSPECTOR = '''<section class="inspector" data-inspector><p class="lab-kicker">میز بازکردن مدل / محاسبهٔ واقعی Python</p><h2>درون فایل ذخیرهٔ مدل خودتان را ببینید</h2>
<p>این میز برای پس از درس ردیابی مدل است. checkpoint یعنی فایل وضعیت ذخیره‌شدهٔ مدل. ابزار Python یک «گذر رو به جلو» یا forward، یعنی محاسبهٔ خروجی از ورودی با وزن‌های فعلی، انجام می‌دهد و عددها را در قالب متنی JSON صادر می‌کند. این بخش شبکه را در مرورگر اجرا نمی‌کند؛ وزن‌ها یا فایل .pt را اینجا وارد نکنید. نمونهٔ همراه از اجرای کوچک روی دادهٔ همین کتاب است و نشانهٔ کیفیت زبان عمومی نیست.</p>
<div class="control-row"><button type="button" data-inspect-demo>بازکردن نمونهٔ واقعی همراه</button><label class="file-label">ورود JSON بازرسی <input type="file" data-inspect-file accept=".json,application/json"></label></div><p data-inspect-status role="status" aria-live="polite"></p>
<pre><code>python -m mini_gpt.inspect --checkpoint runs/first/best.pt --prompt "مدل " --output runs/first/inspection.json --layer 0 --head 0 --max-tokens 16 --generate-tokens 4 --metrics runs/first/metrics.csv --device cpu</code></pre>
<div data-inspect-output hidden><div class="source-stamp" data-inspect-source></div><div class="control-row"><label>موقعیت متن <select data-inspect-position></select></label><label>پنجرهٔ محاسبه <select data-inspect-view><option value="embeddings">نشانه و نمایش</option><option value="qkv">پرسش، کلید، مقدار</option><option value="scores">امتیاز خام و مقیاس‌شده</option><option value="weights">پوشش و وزن توجه</option><option value="logits">امتیاز و احتمال خروجی</option><option value="generation">گام‌های تولید</option><option value="metrics">منحنی آموزش</option></select></label></div>
<div data-inspect-content></div></div><p class="meta">هر فایل یک لایه و یک سر منتخب را نشان می‌دهد؛ برای مقایسه دوباره با --layer و --head خروجی بگیرید. وزن توجه توضیح کامل علت پاسخ نیست. همهٔ بارگذاری‌ها محلی‌اند؛ داده‌ای فرستاده نمی‌شود.</p></section>'''


def lab_page():
    stations = [('tokens','۰۱ · متن و نشانه',token_lab),('probability','۰۲ · امتیاز و احتمال',softmax_lab),('attention','۰۳ · پرسش و توجه',attention_lab),('shapes','۰۴ · شکل و محور',shape_lab),('model','۰۵ · درون مدل واقعی',lambda:INSPECTOR)]
    return '<p class="objective">اول پیش‌بینی، بعد دست‌کاری، بعد مشاهده. یک میز را باز کنید و فقط یک چیز را تغییر دهید.</p>' + '''<nav class="lab-index" aria-label="میزهای آزمایش"><a href="#tokens">نویسه</a><a href="#probability">احتمال</a><a href="#attention">توجه</a><a href="#shapes">شکل‌ها</a><a href="#model">مدل واقعی</a><a href="#thinking">«فکرکردن» یعنی چه؟</a></nav>''' + ''.join(f'<details class="lab-station" id="{identifier}"'+(' open' if identifier=='tokens' else '')+f'><summary>{title}</summary>{fn()}</details>' for identifier,title,fn in stations) + '<details class="model-map-disclosure"><summary>نقشهٔ کامل یک گام تولید</summary>'+pipeline()+'</details>' + '''<section id="thinking"><h2>پس GPT چطور «فکر» می‌کند؟</h2><p>در این صفحه «فکرکردن» فقط نام محاوره‌ایِ زنجیرهٔ محاسبه است: متن به شناسه می‌رود؛ وزن‌های آموخته نمایش آن را تبدیل می‌کنند؛ در پایان برای نشانهٔ بعدی امتیاز می‌سازند. softmax آن‌ها را به احتمال تبدیل می‌کند و قاعدهٔ انتخاب یک شناسه برمی‌گزیند. با افزودن آن شناسه، همین فرایند دوباره اجرا می‌شود.</p><p>توزیع احتمال می‌تواند الگوهای پیچیده را بازتاب دهد، اما نمودار ما تجربهٔ ذهنی، قصد یا روش فکر انسان را نشان نمی‌دهد. همچنین احتمال بیشتر الزاماً حقیقت بیشتر نیست: مدل برای پیش‌بینی نشانه آموزش دیده، نه برای تضمین درستی هر ادعا.</p><p>با عوض‌کردن دما وزن‌ها ثابت می‌مانند، ولی انتخاب تغییر می‌کند. با آموزش، خود وزن‌ها عوض می‌شوند. این دو مداخله را در دفتر با دو نام جدا ثبت کنید.</p></section>'''


JOURNAL = '''<p class="objective">دفتر قرار نیست مرتب‌تر از فکر شما باشد. یک پیش‌بینی دقیقِ اشتباه، از «همه‌چیز را فهمیدم» مفیدتر است.</p>
<section class="journal-sheet" data-journal><p class="privacy-note">فقط در همین مرورگر ذخیره می‌شود؛ نه حساب دارد، نه ارسال به سرور. حالت خصوصی یا پاک‌کردن دادهٔ مرورگر می‌تواند آن را حذف کند. مرتب خروجی بگیرید. روی HTTP محلی، همهٔ صفحه‌ها مبدأ مشترک دارند؛ ذخیرهٔ file:// به مرورگر وابسته است.</p>
<form data-journal-form><div class="control-row"><label>درس یا ایستگاه <select name="lesson" data-journal-lesson></select></label><label>عنوان آزمایش <input name="title" required maxlength="180" placeholder="مثلاً چرا سطر اول فقط خودش را می‌بیند؟"></label></div>
<label>پیکربندی و فرمان اجرا <textarea name="config" rows="2" maxlength="4000" placeholder="seed، T/C/H/L، نرخ یادگیری، فایل داده و فرمان؛ اگر آزمایش عددی بود، ورودی‌ها" dir="auto"></textarea></label>
<label>پیش‌بینی من <textarea name="prediction" rows="3" maxlength="6000" placeholder="پیش از دیدن نتیجه، چه انتظاری دارم؟"></textarea></label>
<label>مشاهده و شاهد <textarea name="observation" rows="4" maxlength="8000" placeholder="شکل، عدد خطا، خروجی، پیام خطا؛ دقیقاً چه اتفاقی افتاد؟"></textarea></label>
<label>تفسیر، اشتباه و کشف <textarea name="interpretation" rows="3" maxlength="6000" placeholder="کدام بخش حدسم رد شد؟ چه نتیجه‌ای هنوز نمی‌توانم بگیرم؟"></textarea></label>
<label>پرسش و گام بعد <textarea name="question" rows="2" maxlength="4000" placeholder="آزمایش بعدی فقط چه چیزی را تغییر می‌دهد؟"></textarea></label>
<div class="control-row"><button type="submit">ثبت یادداشت</button><button type="button" data-journal-new>برگهٔ تازه</button><span data-draft-status role="status" aria-live="polite"></span></div></form>
<div class="journal-toolbar"><h2>برگه‌های ثبت‌شده</h2><label>جست‌وجو در دفتر <input type="search" data-journal-search maxlength="180"></label><button type="button" data-journal-export>خروجی همهٔ دفتر</button><label class="file-label">ورود دفتر <input type="file" data-journal-import accept=".json,application/json"></label></div>
<p data-journal-status role="status" aria-live="polite"></p><div data-journal-entries></div><details data-journal-preview hidden><summary>نسخهٔ متنی دفتر برای کپی</summary><textarea data-journal-json readonly rows="12" dir="ltr" aria-label="JSON دفتر"></textarea></details></section>'''
