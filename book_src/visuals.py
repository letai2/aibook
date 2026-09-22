"""Authored visual briefs: one subject-specific opening for every lesson.

Diagrams are explanatory, not model measurements. Their captions state their
scope; actual interactive/model values remain in experience.py/experience.js.
"""
from html import escape

# ID: subject, composition, visual grammar, exact labels, explanatory caption.
BRIEFS = {
 '26a-sequence-models': ('sequence','workbench','compare','Token-only|Fixed-window MLP|RNN','راه ورود گذشته را بررسی کنید؛ نام خانوادهٔ شبکه به‌تنهایی کافی نیست.'),
 '62b-lifecycle': ('research','diagram','flow','Pretraining|Base model|SFT|RLHF or DPO','یک مسیر رایج آموزش؛ LoRA انتخاب وزن‌ها و RAG افزودن زمینه است.'),
 '05a-vector-operations': ('math','statement','equation','0.25[1,2] + 0.75[3,−1] = [2.5,−0.25]','تعداد ویژگی‌ها حفظ می‌شود؛ روی سهم Vectorها جمع می‌زنیم.'),
 '12b-neuron': ('network','diagram','flow','x|w·x+b|ReLU|prediction|loss','ابتدا مقدارها را رو به جلو می‌سازیم؛ سپس حساسیت‌ها را رو به عقب حساب می‌کنیم.'),
 '01-model': ('foundations','statement','equation','prediction = x × w','ورودی ثابت است؛ این بار فقط وزن را تغییر می‌دهیم.'),
 '01-learning': ('foundations','diagram','flow','نمونه‌ها|مدل|پیش‌بینی|زیان','ویژگی وارد مدل می‌شود؛ هدف برای سنجش خروجی است.'),
 '02-token': ('language','question','tokens','م|د|ل| |c|a|t','در این پروژه، هر نقطه‌کد یک نشانه است؛ نه هر واژه.'),
 '03-counts': ('foundations','workbench','matrix','a→a|a→b|b→a|b→b','چهار حالت ممکن برای انتقال میان دو Token؛ تعداد هر حالت را از متن می‌شماریم.'),
 '04-splits': ('debug','case','compare','دادهٔ آموزش|دادهٔ ارزیابی','مدل فقط از قسمت آموزش می‌آموزد؛ ارزیابی کنار می‌ماند.'),
 '05-shape': ('math','diagram','coordinates','1|2|3|4|5|6','دو سطر و سه ستون: شکل این جدول (2,3) است.'),
 '06-dot': ('math','statement','equation','q · k = Σ qᵢkᵢ','هر جفت ویژگی سهمی در یک عدد نهایی دارد.'),
 '07-matmul': ('math','diagram','equation','(T,D) × (D,T) = (T,T)','شکل Q و Kᵀ در ضرب: دو محور داخلی حذف می‌شوند و دو محور بیرونی می‌مانند.'),
 '08-probability': ('math','editorial','bars','0.1|0.3|0.6','یک توزیع نمونه: احتمال‌ها نامنفی‌اند و مجموعشان یک است.'),
 '09-softmax': ('math','question','bars','2|1|0','سه امتیاز خام؛ بلندی میله‌ها هنوز احتمال نیست.'),
 '10-entropy': ('math','statement','equation','loss = −log p(target)','فقط احتمال نشانهٔ هدف وارد این زیان تک‌نمونه‌ای می‌شود.'),
 '11-derivative': ('math','editorial','equation','Δloss / Δw','تغییر کوچک وزن را با تغییر زیان مقایسه می‌کنیم.'),
 '12-chain': ('math','diagram','flow','w|prediction|loss','اثر تغییر از این مسیر می‌گذرد؛ مشتق‌ها در هم ضرب می‌شوند.'),
 '12-sgd': ('training','question','compare','همهٔ نمونه‌ها|یک دستهٔ کوچک','یک دستهٔ کوچک، برآوردی از گرادیان کل داده می‌دهد.'),
 '13-torch': ('code','workbench','flow','tensor|operation|gradient','آرایه و مسیر محاسبه، دو بخش یک ابزارند.'),
 '14-index-device': ('code','case','stack','shape: (2,3)|dtype: float32|device: cpu','سه قرارداد مستقل برای یک تنسور نمونه.'),
 '15-broadcast': ('math','question','compare','(B,T,C)|(1,1,C)','گسترش خودکار از محور آخر، سازگاری اندازه‌ها را بررسی می‌کند.'),
 '16-reshape': ('code','case','compare','reshape|transpose','گروه‌بندی تازه، با جابه‌جا‌کردن محور یکی نیست.'),
 '17-autograd': ('code','diagram','flow','forward|loss|backward|grad','محاسبهٔ مشتق، به‌تنهایی وزن را به‌روزرسانی نمی‌کند.'),
 '18-module': ('code','workbench','equation','y = xWᵀ + b','Module وزن و بایاس را به‌عنوان پارامتر ثبت می‌کند.'),
 '19-network': ('network','statement','flow','Linear|nonlinearity|Linear','بدون Activation غیرخطی، این دو Linear را می‌توان با یک تبدیل Affine جایگزین کرد.'),
 '20-loader': ('code','diagram','stack','sample 0|sample 1|sample 2|batch','نمونه‌ها در امتداد محور دسته کنار هم قرار می‌گیرند.'),
 '21-tokenizer': ('language','workbench','flow','encode|save|load|decode','شناسه باید بعد از ذخیره و بازیابی همان معنی قراردادی را داشته باشد.'),
 '22-bpe': ('language','diagram','tokens','c|a|t|→|ca|t','یک ادغام فرضی: دو قطعه به یک قطعه تبدیل می‌شوند.'),
 '23-shift': ('language','case','compare','input: a b c|target: b c d','هدف هر موقعیت، نشانهٔ بعدی است؛ نه خود نشانه.'),
 '24-data-contract': ('debug','editorial','stack','طول متن|مرز تقسیم|واژگان|هویت فایل','قبل از آموزش، چهار قرارداد داده را بررسی می‌کنیم.'),
 '25-embedding': ('language','statement','flow','ID: 3|row 3|vector C','شناسه نشانی سطر است، نه اندازهٔ معنای نشانه.'),
 '26-positions': ('language','question','equation','xₜ = tokenₜ + positionₜ','جمع دو نمایش Cتایی، شکل را تغییر نمی‌دهد.'),
 '26b-sequence-memory': ('sequence','diagram','flow','h₀|h₁|h₂|h₃','در مدل بازگشتی، گذشته از مسیر حالت‌های پیاپی می‌گذرد.'),
 '27-attention-why': ('attention','question','network','نشانهٔ اول|نشانهٔ دوم|نشانهٔ سوم','نمایش یک موقعیت می‌تواند از موقعیت‌های دیگر اطلاعات بگیرد.'),
 '28-qkv': ('attention','statement','stack','Q · درخواست تطبیق|K · معیار تطبیق|V · محتوای انتقال','تطبیق با Q و K انجام می‌شود؛ اطلاعات از V می‌آید.'),
 '29-scores': ('attention','workbench','matrix','q₀k₀|q₀k₁|q₁k₀|q₁k₁','سطر پرسش و ستون کلید است؛ هر خانه یک ضرب داخلی.'),
 '30-scaling': ('attention','editorial','equation','scores = QKᵀ / √D','D تعداد ویژگی هر سر است، نه طول جمله.'),
 '31-values': ('attention','diagram','flow','scores|softmax|weights × V','به‌جای انتخاب یک Value، ترکیب وزن‌دار Valueهای مجاز را می‌سازیم.'),
 '32-self': ('attention','statement','branch','Q|K|V','در خودتوجهی، هر سه تبدیل از همان دنباله آغاز می‌شوند.'),
 '33-mask': ('attention','case','mask','past|now|future','بالای قطر ممنوع است؛ پوشش قبل از softmax اعمال می‌شود.'),
 '34-causal-test': ('debug','question','compare','a b c x|a b c y','در حالت ارزیابی، خروجی پیشوند مشترک باید یکسان بماند.'),
 '35-split-heads': ('architecture','diagram','flow','(B,T,C)|(B,T,H,D)|(B,H,T,D)','شکستن ویژگی و جابه‌جایی محور، دو عمل جدا هستند.'),
 '36-merge-heads': ('architecture','workbench','flow','(B,H,T,D)|(B,T,H,D)|(B,T,C)','محور زمان باید پیش از چسباندن ویژگی‌ها به جای خود برگردد.'),
 '37-ffn': ('network','statement','flow','C|4C|GELU|C','ویژگی‌های هر موقعیت جداگانه پردازش می‌شوند.'),
 '38-residual': ('architecture','question','equation','y = x + F(x)','زیرلایه اصلاحی روی ورودی می‌سازد؛ شکل دو مسیر باید برابر باشد.'),
 '39-layernorm': ('math','workbench','equation','(x − mean) / √(var + ε)','در مدل ما میانگین و واریانس روی محور ویژگی است.'),
 '40-block': ('architecture','diagram','stack','y = x + Attention(LN(x))|out = y + FFN(LN(y))','بلوک Pre-Norm: دو زیرلایه و دو مسیر جمع.'),
 '41-stack': ('architecture','statement','stack','block 1|block 2|block 3','شکل ثابت می‌ماند؛ پارامترهای هر بلوک مستقل‌اند.'),
 '42-families': ('architecture','editorial','compare','دید دوطرفه|دید علّی','تفاوت مسیر اطلاعات، از شباهت نام کلاس‌ها مهم‌تر است.'),
 '43-lm-head': ('model','diagram','flow','(B,T,C)|Linear: C → V|(B,T,V)','برای هر موقعیت، یک امتیاز برای هر نشانهٔ واژگان.'),
 '44-parameters': ('model','workbench','stack','token: V × C|position: T × C|blocks + head','اندازهٔ جدول‌ها را بشمارید؛ نام مدل عدد پارامتر را تعیین نمی‌کند.'),
 '45-trace': ('model','diagram','flow','text|IDs|blocks|logits','هر توقف در این مسیر، نام و شکل و عدد قابل وارسی دارد.'),
 '46-gradient-path': ('training','case','flow','loss|head|blocks|embeddings','مسیر مشتق را از خروجی تا جدول ورودی دنبال می‌کنیم.'),
 '47-loop': ('training','workbench','flow','zero_grad|forward|backward|step','در هر گام، مشتق قبلی پاک و مشتق تازه محاسبه می‌شود.'),
 '48-evaluate': ('training','question','equation','mean = Σ(loss × count) / Σcount','دستهٔ کوچک‌تر آخر، وزن کمتری در میانگین دارد.'),
 '49-rate': ('debug','case','compare','گام کوچک|گام بزرگ','اندازهٔ گام را با روند ارزیابی بسنجید، نه یک عدد منفرد.'),
 '49b-schedule': ('training','diagram','schedule','warmup|cosine|floor','شکل شماتیک برنامهٔ نرخ است؛ منحنی یک اجرای واقعی نیست.'),
 '50-checkpoint': ('code','statement','stack','weights + optimizer|vocabulary + config|step + RNG','برای ادامهٔ همان آزمایش، تنها وزن‌ها کافی نیستند.'),
 '51-resume': ('training','case','compare','4 + 2 steps|6 steps','در آزمون کنترل‌شدهٔ CPU، دو مسیر باید به یک وزن برسند.'),
 '52-first-run': ('training','workbench','flow','seed|train|validate|save','اولین اجرا باید کوچک و قابل بازسازی باشد.'),
 '53-curves': ('training','editorial','curves','train|validation','شماتیک: خط پیوسته آموزش و خط‌چین ارزیابی است؛ نه اندازه‌گیری یک اجرای واقعی.'),
 '54-generate': ('language','diagram','tokens','context|+ token|+ token|…','هر بار فقط یک نشانه انتخاب و به زمینه اضافه می‌شود.'),
 '55-temperature': ('language','question','equation','softmax(logits / τ)','وزن مدل ثابت است؛ فقط توزیع انتخاب تغییر می‌کند.'),
 '56-topkp': ('language','workbench','compare','top-k: count|top-p: mass','تعداد نامزد و مجموع احتمال، دو آستانهٔ متفاوت‌اند.'),
 '57-prompts': ('debug','case','flow','prompt|unknown IDs|context window','قبل از تولید، ببینید چه چیزی واقعاً وارد مدل شده است.'),
 '58-ablation': ('debug','question','compare','مدل مرجع|تنها یک تغییر','فقط یک عامل را عوض کنید تا مقایسه قابل تفسیر بماند.'),
 '59-capacity': ('model','statement','stack','width C|depth L|heads H','سه تنظیم متفاوت؛ افزایش یکی تضمین بهبود نیست.'),
 '60-bug-clinic': ('debug','case','compare','expected: (B,T,C)|received: (B,C,T)','عددها را دنبال کنید؛ شکل درست‌نما می‌تواند محور غلط داشته باشد.'),
 '61-one-batch': ('debug','workbench','flow','one batch|repeat|loss ↓','اگر مدل یک دسته را حفظ نمی‌کند، مسئله را کوچک‌تر کنید.'),
 '62-journal': ('research','editorial','flow','پیش‌بینی|مشاهده|توضیح|بازبینی','یک نتیجه زمانی مفید است که بتوانید دوباره آن را بسازید.'),
 '63-scale': ('research','statement','stack','داده|پارامترها|حافظه|ارتباط دستگاه‌ها','بزرگ‌کردن مدل، مسئلهٔ هماهنگی و منابع را هم بزرگ می‌کند.'),
 '64-cache': ('research','diagram','equation','K,V = concat(cache, new)','کلید و مقدار تازه به بخش ذخیره‌شده افزوده می‌شوند؛ سپس توجه محاسبه می‌شود.'),
 '65-sft': ('research','editorial','flow','base model|instruction data|updated weights','در تنظیم نظارت‌شده، وزن‌ها با نمونه‌های دستور و پاسخ عوض می‌شوند.'),
 '65b-lora': ('research','workbench','equation','W′ = W + BA','در مثال آموزشی، W ثابت و دو ماتریس کوچک قابل آموزش‌اند.'),
 '66-preference': ('research','question','compare','پاسخ ترجیح‌داده‌شده|پاسخ دیگر','سیگنال ترجیح نسبی است؛ تضمین حقیقت یا بی‌خطایی نیست.'),
 '67-rag': ('research','statement','flow','retrieve|add context|generate','افزودن سند به زمینه، با به‌روزرسانی وزن تفاوت دارد.'),
}

THEME_LABELS = {'foundations':'از مسئله تا مدل','math':'زبان عددها','code':'میز کدنویسی','language':'متن و نمایش','sequence':'حافظه و دنباله','attention':'مسیر اطلاعات','network':'ساخت یک شبکه','architecture':'معماری مدل','model':'یک مدل کامل','training':'چرخهٔ یادگیری','debug':'ردیابی یک خطا','research':'افق پژوهش'}


def concept_visual(identifier):
    theme, pattern, grammar, content, caption = BRIEFS[identifier]
    values = content.split('|')
    # Text remains real selectable HTML; SVG is reserved for relationships.
    if grammar in ('equation', 'compare', 'stack', 'tokens'):
        visual = '<div class="visual-' + grammar + '" dir="ltr">' + ''.join(
            '<span dir="auto"'+(' class="visual-fa"' if any('\u0600'<=c<='\u06ff' for c in value) else '')+'>' + escape(value) + '</span>' for value in values) + '</div>'
    elif grammar == 'coordinates':
        visual = '<div class="coordinate-label">محور ستون · ۳ خانه</div><div class="visual-grid grid-3" dir="ltr">'+''.join('<span>'+v+'</span>' for v in values)+'</div><div class="coordinate-label">محور سطر · ۲ خانه</div>'
    elif grammar in ('matrix', 'mask'):
        n = 4 if grammar == 'mask' else 2
        cells = ''.join(f'<span class="{"blocked" if grammar == "mask" and j > i else "allowed"}">{"×" if j > i else "·"}</span>' if grammar == 'mask' else f'<span>{escape(values[i*n+j])}</span>' for i in range(n) for j in range(n))
        visual = f'<div class="visual-grid grid-{n}" dir="ltr">{cells}</div>'
    elif grammar == 'bars':
        heights = (22, 66, 132) if identifier == '08-probability' else (150, 75, 0)
        visual = '<svg viewBox="0 0 420 210" aria-hidden="true">' + ''.join(f'<rect class="bar-{i}" x="{60+i*105}" y="{170-height}" width="56" height="{height}"/><text x="{88+i*105}" y="198" text-anchor="middle">{escape(value)}</text>' for i,(value,height) in enumerate(zip(values,heights))) + '</svg>'
    elif grammar in ('schedule','curves'):
        paths = '<path class="plot-line" d="M35 170 L85 38 C180 40 180 165 355 165"/>' if grammar == 'schedule' else '<path class="plot-line" d="M35 38 Q120 160 355 170"/><path class="plot-line plot-secondary" d="M35 55 Q170 170 355 80"/>'
        visual = f'<svg viewBox="0 0 420 210" aria-hidden="true"><path class="plot-axis" d="M35 20 V180 H390"/>{paths}<text x="185" y="207">step</text></svg><div class="plot-legend">' + ' / '.join(map(escape, values)) + '</div>'
    elif grammar == 'branch':
        visual = '<svg viewBox="0 0 420 180" aria-hidden="true"><g class="network-path"><path d="M210 40 L70 135 M210 40 V135 M210 40 L350 135"/></g><g text-anchor="middle"><text x="210" y="27">X</text><text x="70" y="163">Q</text><text x="210" y="163">K</text><text x="350" y="163">V</text></g></svg>'
    elif grammar == 'network':
        visual = '<svg viewBox="0 0 420 160" aria-hidden="true"><g class="network-path"><path d="M70 110 Q210 -40 350 110"/><path d="M70 110 Q140 0 210 110"/><path d="M210 110 Q280 0 350 110"/></g><g class="network-nodes"><circle cx="70" cy="110" r="10"/><circle cx="210" cy="110" r="10"/><circle cx="350" cy="110" r="10"/></g></svg><div class="network-labels" dir="ltr">' + ''.join('<span dir="auto" class="visual-fa">'+escape(x)+'</span>' for x in values) + '</div>'
    else:
        visual = '<ol class="visual-flow" dir="ltr">' + ''.join('<li><bdi dir="auto"'+(' class="visual-fa"' if any('\u0600'<=c<='\u06ff' for c in x) else '')+'>'+escape(x)+'</bdi></li>' for x in values) + '</ol>'
    return f'<figure class="concept-visual grammar-{grammar}">{visual}<figcaption>{caption}</figcaption></figure>'


def opening(lesson, order, total):
    theme, pattern, *_ = BRIEFS[lesson.id]
    return (f'<header class="lesson-opening opening-{pattern}">'
            f'<div class="opening-copy"><p class="eyebrow">{THEME_LABELS[theme]}<span>درس {order:02} / {total}</span></p>'
            f'<h1>{escape(lesson.title)}</h1><p class="objective">{lesson.objective}</p></div>'
            + concept_visual(lesson.id) + '</header>')
