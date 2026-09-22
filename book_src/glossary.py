"""Authored concept reference; names are English, teaching prose is Persian.

Fields: name, accepted old spellings, meaning, motivation/intuitive model,
technical detail, small example, Mini-GPT connection, related slugs, lesson IDs.
Aliases deliberately exclude ambiguous Persian words such as خطا and نمایش.
"""
from dataclasses import dataclass, replace
import re


@dataclass(frozen=True)
class Term:
    name: str
    aliases: str
    meaning: str
    intuition: str
    technical: str
    example: str
    project: str
    related: str
    lessons: str


TERMS = {
 'mini-batch': Term('Mini-batch','مینی‌بچ|مینی بچ',
  'یک Batch کوچک‌تر از همهٔ Dataset آموزشی است.',
  'برای تغییر وزن لازم نیست پیش از هر گام تمام داده را بخوانیم؛ بخشی از داده برآورد ارزان‌تری از Gradient می‌دهد.',
  'Gradient میانگین Loss این گروه، میانگین Gradient نمونه‌های آن است. با نمونه‌گیری یکنواخت، این برآورد در امید ریاضی با Gradient کل داده سازگار است.',
  'از سه Gradient برابر [-4,-16,-36]، میانگین دو مورد اول −۱۰ و میانگین کل حدود −۱۸٫۶۶۷ است.',
  'در Training پروژه اندازهٔ این گروه با batch_size تعیین می‌شود و پنجره‌ها با جایگذاری انتخاب می‌شوند.',
  'batch gradient optimizer dataset', '12-sgd 20-loader 47-loop'),
 'tensor': Term('Tensor','تنسور|تنسوری|تانسور',
  'آرایه‌ای از عددها با تعداد مشخصی محور است؛ شکل آن می‌گوید روی هر محور چند عضو داریم.',
  'برای پردازش هم‌زمان چند متن، به ظرفی نیاز داریم که هم نمونه، هم موقعیت و هم ویژگی را جدا نگه دارد.',
  'Shape، dtype و device سه قرارداد مستقل‌اند. Tensor الزاماً Parameter نیست و همهٔ Tensorها Gradient ندارند.',
  '<code>torch.zeros(2, 3)</code> شش عدد در دو سطر و سه ستون می‌سازد.',
  'شناسه‌های ورودی شکل (B,T) و نمایش‌های داخل بلوک شکل (B,T,C) دارند.', 'batch embedding parameter', '05-shape 13-torch 45-trace'),
 'embedding': Term('Embedding','جاسازی|تعبیه|نمایش برداری یادگرفتنی|بردارهای جاسازی',
  'روشی برای نسبت‌دادن یک بردار قابل یادگیری به هر شناسهٔ گسسته است.',
  'Token ID فقط نشانی است؛ بزرگی عدد آن معنای زبانی ندارد. Embedding به مدل ویژگی‌های قابل تنظیم می‌دهد.',
  'جدول E با شکل (V,C) دارد. برای هر ID، سطر متناظر برداشته می‌شود؛ مقدارهای این جدول هنگام Training تغییر می‌کنند.',
  'اگر ID برابر ۳ باشد، <code>E[3]</code> یک Vector با C مؤلفه است؛ نه عدد ۳ ضرب‌در یک Vector.',
  'مدل Embedding مربوط به Token و موقعیت را جمع می‌کند و به اولین Transformer block می‌دهد.', 'token vocabulary positional-embedding parameter', '25-embedding 26-positions 46-gradient-path'),
 'token': Term('Token','نشانه',
  'واحدی است که Tokenizer از متن می‌سازد و مدل با شناسهٔ آن کار می‌کند.',
  'محاسبه روی متن به یک قرارداد عددی نیاز دارد. Token می‌تواند نویسه، زیرواژه یا قطعه‌ای دیگر باشد؛ الزاماً یک کلمه نیست.',
  'مرز Token به الگوریتم و Vocabulary بستگی دارد. یک نویسهٔ قابل دیدن ممکن است از چند نقطه‌کد Unicode ساخته شود.',
  'Tokenizer اصلی این کتاب «مدل» را به سه نقطه‌کد م، د و ل تقسیم می‌کند.',
  '<code>CharacterTokenizer</code> پروژه نقطه‌کدمحور است؛ تمرین BPE مسیر آموزشی جداگانه‌ای دارد.', 'tokenizer tokenization vocabulary embedding', '02-token 21-tokenizer 22-bpe'),
 'tokenizer': Term('Tokenizer','نشانه‌بند|نشانه‌ بند|توکنایزر',
  'ابزاری است که متن را به Token ID تبدیل می‌کند و تا حد قراردادش مسیر برگشت را فراهم می‌کند.',
  'مدل باید بداند هر عدد همیشه به کدام واحد متن اشاره می‌کند؛ عوض‌شدن این قرارداد، معنی وزن‌ها را عوض می‌کند.',
  'encode، decode و ذخیرهٔ Vocabulary باید با هم سازگار باشند. ID ناشناخته یا Tokenهای ویژه قواعد جداگانه دارند.',
  'با Vocabulary ثابت، هر بار encode کردن «مدل» باید همان IDها را بدهد.',
  'Vocabulary کنار Checkpoint ذخیره می‌شود؛ هنگام Inference نباید از متن تازه دوباره ساخته شود.', 'token tokenization vocabulary checkpoint', '21-tokenizer 24-data-contract 50-checkpoint'),
 'tokenization': Term('Tokenization','نشانه‌بندی|نشانه‌ بندی',
  'فرایند تقسیم متن به Tokenها و نگاشت آن‌ها به شناسه‌هاست.',
  'انتخاب قطعه‌های کوتاه یا بلند روی طول دنباله و اندازهٔ Vocabulary اثر می‌گذارد.',
  'در روش نقطه‌کدمحور واحدها از Unicode می‌آیند؛ در BPE جفت‌های پرتکرار طی مراحل مشخص ادغام می‌شوند.',
  'در یک ادغام فرضی، c و a می‌توانند به ca تبدیل شوند؛ این ادغام باید در قرارداد Tokenizer ثبت شود.',
  'برای اجرای اصلی از <code>CharacterTokenizer</code> و برای شناخت تفاوت روش‌ها از تمرین BPE استفاده می‌کنیم.', 'token tokenizer vocabulary context-window', '02-token 21-tokenizer 22-bpe'),
 'attention': Term('Attention','توجه محاسباتی|توجه',
  'محاسبه‌ای است که برای هر موقعیت، اطلاعات چند موقعیت را با وزن‌های وابسته به ورودی ترکیب می‌کند.',
  'Embedding ثابت یک Token به‌تنهایی نمی‌گوید در این جمله چه اطلاعاتی لازم است. Attention مسیر تبادل اطلاعات را می‌سازد، نه ذهن یا آگاهی.',
  'Query و Key امتیاز تطبیق می‌سازند؛ Softmax آن‌ها را به وزن تبدیل می‌کند و خروجی، جمع وزن‌دار Valueهاست.',
  'اگر وزن‌ها [0.25,0.75] و Valueها [2,0] و [0,4] باشند، خروجی [0.5,3] می‌شود.',
  'در attention.py امتیازها بر ریشهٔ اندازهٔ هر Head تقسیم می‌شوند و Causal Mask پیش از Softmax اعمال می‌شود.', 'query key value softmax causal-mask multi-head-attention', '27-attention-why 29-scores 31-values'),
 'self-attention': Term('Self-Attention','self attention|خودتوجهی|خود توجهی',
  'نوعی Attention است که Query، Key و Value از یک دنبالهٔ مشترک ساخته می‌شوند.',
  '«Self» به مشترک‌بودن منبع اشاره دارد؛ به این معنی نیست که هر موقعیت فقط خودش را می‌بیند.',
  'برای X مشترک، سه Projection جدا Q، K و V را می‌سازند. علّی‌بودن خاصیت اضافه‌ای است و به Mask بستگی دارد.',
  'بدون Causal Mask، Token اول می‌تواند از Token سوم اطلاعات بگیرد.',
  'نسخهٔ v2 برای نشان‌دادن نشت آینده است؛ v3 محدودیت علّی را اضافه می‌کند.', 'attention query key value causal-mask', '32-self 33-mask 34-causal-test'),
 'transformer': Term('Transformer','تبدیل‌گر|ترنسفورمر',
  'خانواده‌ای از معماری‌های شبکهٔ عصبی است که Attention را با پردازش ویژگی‌ها، Residual Connection و Normalization ترکیب می‌کند.',
  'می‌خواهیم هم رابطهٔ موقعیت‌ها را یاد بگیریم، هم ویژگی هر موقعیت را تغییر بدهیم؛ یک بلوک این دو کار را کنار هم می‌گذارد.',
  'Encoder، Decoder و مدل‌های Decoder-only قرارداد دید متفاوتی دارند. Mini-GPT از بلوک‌های Pre-Norm و Causal Self-Attention استفاده می‌کند.',
  'یک بلوک ابتدا <code>y = x + attention(norm(x))</code> و سپس <code>out = y + ffn(norm(y))</code> را حساب می‌کند.',
  'چند بلوک مستقل پشت هم قرار می‌گیرند؛ شکل (B,T,C) حفظ می‌شود اما مقدارها و Parameterها تغییر می‌کنند.', 'self-attention layer-normalization residual-connection feed-forward-network', '40-block 41-stack 42-families'),
 'layer': Term('Layer','لایه',
  'بخشی از شبکه است که ورودی را طبق یک محاسبهٔ مشخص به خروجی تبدیل می‌کند.',
  'تقسیم شبکه به Layerها کمک می‌کند شکل ورودی، کار هر قطعه و Parameterهایش را جدا بررسی کنیم.',
  'هر Layer الزاماً Parameter ندارد. Linear دارای وزن و Bias است، ولی عملی مثل Dropout وزن قابل یادگیری ندارد.',
  '<code>nn.Linear(4, 8)</code> آخرین محور چهارتایی را به هشت ویژگی تبدیل می‌کند.',
  'Embedding، Projectionهای Attention و Layer نهایی واژگان اجزای متفاوت مدل‌اند.', 'tensor parameter transformer dropout', '18-module 37-ffn 43-lm-head'),
 'batch': Term('Batch','دسته',
  'گروهی از نمونه‌هاست که در یک مرحله با هم پردازش می‌شوند.',
  'به‌جای اجرای جداگانه برای تک‌تک متن‌ها، چند پنجره را کنار هم قرار می‌دهیم تا محاسبه مشترک انجام شود.',
  'محور B تعداد نمونه‌هاست، نه طول متن. اندازهٔ Batch روی هزینهٔ حافظه و برآورد Gradient اثر دارد.',
  'ورودی (4,16) یعنی چهار نمونه، هرکدام با شانزده Token؛ نه یک متن با ۶۴ موقعیت.',
  'Training پنجره‌ها را تصادفی با جایگذاری انتخاب می‌کند؛ DataLoader برای پیمایش کامل Validation استفاده می‌شود.', 'dataset epoch gradient tensor', '05-shape 12-sgd 20-loader'),
 'epoch': Term('Epoch','اپوک',
  'یک بار عبور از همهٔ نمونه‌های Dataset آموزشی، طبق قرارداد پیمایش داده است.',
  'Step و Epoch یک واحد نیستند: در هر Step یک به‌روزرسانی داریم، ولی یک Epoch معمولاً چند Step دارد.',
  'تعداد Stepهای یک Epoch به اندازهٔ Dataset، Batch و drop_last بستگی دارد. نمونه‌برداری با جایگذاری ممکن است این تعبیر را تغییر دهد.',
  'برای ۱۰۰ نمونه و Batch ده‌تایی، بدون حذف نمونه، یک Epoch ده Step دارد.',
  'گزارش اصلی Mini-GPT بر حسب Step است؛ تعداد Step را بی‌دلیل Epoch ننامید.', 'batch dataset training optimizer', '20-loader 47-loop 52-first-run'),
 'gradient': Term('Gradient','گرادیان',
  'مجموعهٔ مشتق‌های یک مقدار اسکالر، مانند Loss، نسبت به متغیرهای ورودی آن است.',
  'می‌پرسیم اگر هر Parameter کمی تغییر کند، Loss به کدام سمت و با چه حساسیتی تغییر می‌کند.',
  'برای θ چندبعدی، Gradient شامل ∂L/∂θ است. جهت منفی آن یک جهت کاهش محلی برای گام به‌اندازهٔ کافی کوچک است، نه تضمین حل جهانی.',
  'اگر <code>L = w*w</code> و w=3 باشد، مشتق نسبت به w برابر ۶ است.',
  'پس از backward، فیلد grad پارامترها برای Optimizer آماده است؛ zero_grad از انباشته‌شدن ناخواسته جلوگیری می‌کند.', 'loss parameter backpropagation optimizer', '11-derivative 12-chain 46-gradient-path'),
 'backpropagation': Term('Backpropagation','پس‌انتشار مشتق|پس‌انتشار|پس انتشار',
  'روشی برای محاسبهٔ مشتق‌ها در گراف محاسباتی با حرکت از خروجی به ورودی است.',
  'اثر یک Parameter ممکن است از چند تبدیل بگذرد؛ Chain Rule سهم این مسیرها را به هم وصل می‌کند.',
  'Backpropagation مشتق می‌سازد، اما به‌تنهایی Parameterها را تغییر نمی‌دهد. به‌روزرسانی کار Optimizer است.',
  'برای y=2w و L=y²، مشتق L نسبت به w برابر 8w است.',
  '<code>loss.backward()</code> مشتق مسیر Loss تا Embeddingها و بلوک‌ها را محاسبه می‌کند.', 'gradient autograd optimizer loss', '12-chain 17-autograd 46-gradient-path'),
 'optimizer': Term('Optimizer','بهینه‌ساز|بهینه ساز',
  'قاعده‌ای است که با استفاده از Gradient و گاهی وضعیت مراحل قبل، Parameterها را به‌روزرسانی می‌کند.',
  'داشتن جهت تغییر کافی نیست؛ باید بدانیم با چه اندازه و با چه حافظه‌ای از گذشته حرکت کنیم.',
  'SGD ساده از Gradient فعلی استفاده می‌کند. AdamW میانگین‌های متحرک و Weight Decay جداگانه دارد؛ وضعیتش بخشی از ادامهٔ دقیق Training است.',
  'در SGD ساده، <code>w_new = w - lr * grad</code>.',
  'Training loop از AdamW استفاده می‌کند و وضعیت آن را کنار وزن‌ها در Checkpoint می‌گذارد.', 'gradient learning-rate parameter checkpoint', '12-sgd 47-loop 51-resume'),
 'learning-rate': Term('Learning Rate','learning-rate|نرخ یادگیری',
  'ضریبی است که اندازهٔ تغییر Parameterها را در قاعدهٔ Optimizer کنترل می‌کند.',
  'گام بسیار کوچک ممکن است حرکت را کند کند؛ گام بسیار بزرگ می‌تواند Training را ناپایدار کند.',
  'اثر Learning Rate به Optimizer و مقیاس Gradient وابسته است. Schedule می‌تواند مقدار آن را در طول Stepها تغییر دهد.',
  'با Gradient برابر ۴ و Learning Rate برابر 0.01، SGD مقدار w را 0.04 کم می‌کند.',
  'schedule.py برنامهٔ Warmup و Cosine را محاسبه می‌کند؛ ادامهٔ اجرا باید همان شمارهٔ Step را بازیابی کند.', 'optimizer gradient training checkpoint', '49-rate 49b-schedule 51-resume'),
 'loss': Term('Loss','زیان',
  'عددی است که ناسازگاری پیش‌بینی مدل با هدف را طبق یک معیار مشخص اندازه می‌گیرد.',
  'برای مقایسهٔ تنظیم‌های مختلف، به سنجه‌ای عددی نیاز داریم. Loss کمتر روی دادهٔ Training، به‌تنهایی کیفیت متن تازه را ثابت نمی‌کند.',
  'در پیش‌بینی Token بعدی، Cross-Entropy روی Logits و ID هدف محاسبه می‌شود؛ شکل و روش میانگین‌گیری بخشی از تعریف سنجه‌اند.',
  'اگر احتمال Token هدف 0.5 باشد، Loss تک‌نمونه‌ای برابر <code>-log(0.5)</code>، حدود 0.693 است.',
  'Training از Cross-Entropy استفاده می‌کند؛ ارزیابی، مجموع Loss را با تعداد Tokenهای سنجیده‌شده وزن می‌دهد.', 'logits cross-entropy softmax gradient', '01-model 10-entropy 48-evaluate'),
 'logits': Term('Logits','لاجیت|لاجیت‌ها|لوجیت|لوجیت‌ها',
  'امتیازهای خام خروجی مدل برای گزینه‌های ممکن، پیش از تبدیل به احتمال‌اند.',
  'مدل ابتدا میزان ترجیح خود را با عددهای آزاد بیان می‌کند؛ این عددها هنوز احتمال نیستند و می‌توانند منفی باشند.',
  'Softmax روی محور Vocabulary، Logits را به توزیع احتمال تبدیل می‌کند. اضافه‌کردن یک ثابت به همهٔ امتیازهای یک سطر، توزیع را عوض نمی‌کند.',
  'Logits برابر [2,1,0] بعد از Softmax تقریباً [0.665,0.245,0.090] می‌شوند.',
  'خروجی model.py شکل (B,T,V) دارد؛ هر موقعیت برای V گزینه امتیاز می‌دهد.', 'softmax loss vocabulary sampling', '09-softmax 43-lm-head 55-temperature'),
 'softmax': Term('Softmax','سافت‌مکس|سافتمکس',
  'تابعی است که مجموعه‌ای از امتیازها را به عددهای نامنفی با مجموع یک تبدیل می‌کند.',
  'برای وزن‌دادن نسبی به گزینه‌ها، اختلاف امتیازها مهم است؛ Softmax امتیاز بزرگ‌تر را به وزن بزرگ‌تر تبدیل می‌کند.',
  'برای پایداری عددی، ابتدا بیشترین امتیاز از همه کم می‌شود و سپس نمایی‌گیری و تقسیم بر مجموع انجام می‌شود.',
  '<code>softmax([0,0]) = [0.5,0.5]</code>. محور انتخابی تعیین می‌کند کدام گزینه‌ها با هم رقابت کنند.',
  'در Attention روی Keyها و در خروجی مدل روی Vocabulary اجرا می‌شود؛ این دو محور را اشتباه نگیرید.', 'logits attention causal-mask temperature', '09-softmax 29-scores 31-values'),
 'parameter': Term('Parameter','پارامتر',
  'عدد یا Tensor ثبت‌شده‌ای در مدل است که می‌تواند در فرایند یادگیری تنظیم شود.',
  'معماری شکل تابع را تعیین می‌کند؛ Parameterها تنظیم‌های عددی آن تابع‌اند.',
  'در PyTorch ثبت‌شدن با nn.Parameter باعث دیده‌شدن توسط model.parameters می‌شود. Buffer قرارداد متفاوتی دارد و معمولاً قابل آموزش نیست.',
  'Linear با ورودی C و خروجی D، با Bias، تعداد C×D+D پارامتر دارد.',
  'Embeddingها و Projectionها Parameter دارند؛ Causal Mask یک جدول کمکی است، نه وزن قابل یادگیری.', 'tensor layer optimizer embedding', '01-model 18-module 44-parameters'),
 'dataset': Term('Dataset','مجموعه‌داده|مجموعه داده',
  'مجموعهٔ نمونه‌هایی است که طبق قراردادی مشخص به مدل یا ابزار ارزیابی داده می‌شود.',
  'وجود یک فایل متن کافی نیست؛ باید معلوم باشد نمونه چیست، هدف از کجا می‌آید و مرز Training و Validation کجاست.',
  'Dataset پروژه پنجره‌های متوالی می‌سازد؛ Target هر موقعیت یک Token جلوتر است. Vocabulary فقط از بخش Training ساخته می‌شود.',
  'از «abcd» می‌توان ورودی «abc» و هدف «bcd» ساخت.',
  'data.py تقسیم متن و هویت فایل را کنترل می‌کند؛ dataset.py نمونه‌های پنجره‌ای را می‌سازد.', 'batch token context-window validation-set', '01-learning 20-loader 24-data-contract'),
 'checkpoint': Term('Checkpoint','چک‌پوینت|چک پوینت',
  'فایل وضعیت ذخیره‌شدهٔ مدل و، برای ادامهٔ Training، اطلاعات لازم برای بازسازی همان آزمایش است.',
  'ذخیرهٔ وزن‌ها برای بعضی کاربردها کافی است، اما ادامهٔ همان مسیر به وضعیت Optimizer، تصادف و داده هم نیاز دارد.',
  'معماری، Vocabulary، Step، وضعیت RNG و Optimizer باید سازگار باشند. فایل ناشناس را صرفاً به‌دلیل پسوندش قابل اعتماد ندانید.',
  'ادامهٔ چهار Step با دو Step دیگر در آزمون کنترل‌شدهٔ CPU با اجرای شش Step مقایسه می‌شود.',
  'checkpoint.py قرارداد best.pt و last.pt را بررسی می‌کند؛ اولی بهترین ارزیابی و دومی آخرین وضعیت است.', 'parameter optimizer tokenizer training', '50-checkpoint 51-resume 52-first-run'),
 'fine-tuning': Term('Fine-Tuning','fine tuning|تنظیم تکمیلی|تنظیم دقیق|فاین‌تیونینگ',
  'ادامهٔ آموزش یک مدل از وزن‌های ازپیش‌آموخته، با داده یا هدفی متناسب با کاربرد تازه است.',
  'به‌جای شروع از عددهای تصادفی، از توانایی قبلی استفاده می‌کنیم؛ دادهٔ تازه می‌تواند رفتار مدل را تغییر دهد، نه اینکه بی‌خطایی را تضمین کند.',
  'همه یا بخشی از Parameterها به‌روزرسانی می‌شوند. SFT یک نوع Fine-Tuning با پاسخ مرجع است؛ LoRA روش محدودکردن Parameterهای قابل آموزش است.',
  'مدل پایه را روی جفت‌های دستور و پاسخ ادامه می‌دهیم و Loss را در موقعیت‌های پاسخ محاسبه می‌کنیم.',
  'این موضوع در انتهای کتاب معرفی می‌شود؛ Training اصلی Mini-GPT از ابتداست و سامانهٔ کامل SFT نیست.', 'sft lora parameter training', '65-sft 65b-lora 66-preference'),
 'inference': Term('Inference','اینفرنس',
  'استفاده از مدل آماده برای محاسبهٔ خروجی با Parameterهای ثابت است.',
  'یادگیری و استفاده دو کار متفاوت‌اند؛ ممکن است مدل بدون هیچ به‌روزرسانی وزن، بارها خروجی تازه بسازد.',
  'eval رفتار Layerهایی مثل Dropout را عوض می‌کند. no_grad یا inference_mode ثبت گراف Autograd را کنترل می‌کند؛ این دو تصمیم یکی نیستند.',
  'یک Prompt می‌دهیم، Logits آخرین موقعیت را می‌گیریم و Token بعدی را انتخاب می‌کنیم.',
  'generate.py مدل را از Checkpoint می‌خواند و با قاعدهٔ Sampling خروجی می‌سازد.', 'checkpoint sampling logits dropout', '01-model 54-generate 55-temperature'),
 'sampling': Term('Sampling','نمونه‌گیری|نمونه گیری|نمونه‌برداری',
  'انتخاب یک گزینه طبق یک توزیع احتمال است.',
  'بیشترین احتمال تنها انتخاب ممکن نیست؛ Sampling اجازه می‌دهد گزینه‌های دیگر هم به نسبت وزنشان انتخاب شوند.',
  'Temperature و Top-k/Top-p توزیع انتخاب را تغییر می‌دهند. Seed ثابت فقط در شرایط اجرایی سازگار به بازتولید کمک می‌کند.',
  'در توزیع [0.8,0.2] گزینهٔ دوم ناممکن نیست؛ در تکرارهای زیاد تقریباً یک‌پنجم انتخاب‌ها را می‌گیرد.',
  'sampling.py قانون انتخاب را از محاسبهٔ Logits جدا می‌کند؛ بازرس احتمال خام و احتمال انتخاب را جدا نشان می‌دهد.', 'softmax temperature top-k top-p inference', '03-counts 54-generate 56-topkp'),
 'context-window': Term('Context Window','context length|طول زمینه|پنجرهٔ زمینه|پنجرهٔ بافت',
  'بخشی از دنباله است که مدل برای محاسبهٔ فعلی اجازه و ظرفیت دیدن آن را دارد.',
  'مدل این پروژه حافظهٔ نامحدود از همهٔ متن ندارد؛ باید اندازهٔ ورودی و شیوهٔ بریدن متن را بدانیم.',
  'حداکثر طول با تنظیم مدل مشخص می‌شود. در Causal Attention، هر موقعیت حتی داخل این پنجره هم فقط گذشته و خودش را می‌بیند.',
  'با ظرفیت ۱۶، متن بلندتر هنگام تولید به پنجره‌ای محدود می‌شود؛ این به معنای فهم همهٔ متن قبلی نیست.',
  'تنظیم <code>context_length</code> هم با جدول موقعیت و هم با Shape امتیاز Attention ارتباط دارد.', 'token causal-mask positional-embedding kv-cache', '23-shift 26-positions 57-prompts'),
 'layer-normalization': Term('Layer Normalization','LayerNorm|layer normalization|نرمال‌سازی لایه|نرمال‌سازی ویژگی‌های هر موقعیت',
  'ویژگی‌های یک نمونه یا موقعیت را بر اساس میانگین و واریانس محورهای مشخص نرمال می‌کند.',
  'می‌خواهیم مقیاس عددهای ورودی زیرلایه کنترل شود، بدون آنکه موقعیت‌های متن با هم میانگین‌گیری شوند.',
  'در این مدل محور C نرمال می‌شود و سپس ضریب و جابه‌جایی قابل یادگیری اعمال می‌شود. ε از تقسیم بر صفر جلوگیری می‌کند.',
  'برای (B,T,C)، هر جفت (b,t) آمار C ویژگی خودش را دارد؛ آمار کل Batch استفاده نمی‌شود.',
  'بلوک‌ها Pre-Norm هستند: Layer Normalization پیش از Attention و FFN قرار می‌گیرد.', 'transformer residual-connection layer parameter', '39-layernorm 40-block 58-ablation'),
 'residual-connection': Term('Residual Connection','اتصال باقی‌مانده|مسیر جمع مستقیم|مسیر میان‌بر|residual connection',
  'ورودی یک قطعه را از مسیری مستقیم به خروجی تبدیل‌شدهٔ آن جمع می‌کند.',
  'زیرلایه لازم نیست همه‌چیز را از نو بسازد؛ می‌تواند اصلاحی روی ورودی یاد بگیرد.',
  'رابطهٔ y=x+F(x) به سازگاری Shape دو طرف نیاز دارد. مسیر مستقیم برای Gradient مفید است، اما مصونیت کامل از ناپایداری نیست.',
  'اگر F(x)=0 باشد، خروجی همان x می‌ماند.',
  'در هر Transformer block دو جمع مستقل داریم: پس از Attention و پس از FFN.', 'transformer gradient layer-normalization feed-forward-network', '38-residual 40-block 46-gradient-path'),
 'query': Term('Query','بردار پرسش',
  'یک Vector است که مشخص می‌کند یک موقعیت در محاسبهٔ Attention با چه معیاری منابع را مقایسه کند.',
  'می‌توان آن را درخواست تطبیق دانست، نه یک سؤال زبانی واقعی یا نوع خاصی از کلمه.',
  'Q با Projection قابل یادگیری ساخته می‌شود. ضرب QKᵀ برای هر Query امتیاز همهٔ Keyهای مجاز را می‌سازد.',
  'برای Q با شکل (T,D) و K با شکل (T,D)، امتیازها شکل (T,T) دارند.',
  'در هر Head، Query از ورودی نرمال‌شدهٔ همان بلوک ساخته می‌شود.', 'key value attention self-attention', '28-qkv 29-scores 35-split-heads'),
 'key': Term('Key','بردار کلید تطبیق',
  'یک Vector است که یک موقعیت برای سنجیده‌شدن در برابر Queryها فراهم می‌کند.',
  'تطبیق Query با Key وزن رابطه را تعیین می‌کند؛ محتوای منتقل‌شونده از Value می‌آید.',
  'ستون‌های QKᵀ به Keyها مربوط‌اند. جابه‌جایی Q و K نقش سطر و ستون را عوض می‌کند.',
  'اگر q=[1,0] و k=[2,3] باشد، ضرب داخلی آن‌ها ۲ است.',
  'Projection مستقل K برای هر Head ویژگی‌های مناسب تطبیق را می‌سازد.', 'query value attention kv-cache', '28-qkv 29-scores 64-cache'),
 'value': Term('Value','بردار محتوای منتقل‌شده',
  'بردار اطلاعاتی است که پس از تعیین وزن‌های Attention در ترکیب نهایی شرکت می‌کند.',
  'معیار انتخاب منبع با محتوای آن یکی نیست: Q و K وزن می‌سازند و V اطلاعات می‌دهد.',
  'خروجی O=AV است. با Q و K ثابت، تغییر V خروجی را عوض می‌کند، نه وزن‌های A را.',
  'ترکیب نیم‌به‌نیم [2,0] و [0,4] برابر [1,2] است.',
  'پس از ترکیب V در هر Head، خروجی Headها دوباره کنار هم قرار می‌گیرد.', 'query key attention multi-head-attention', '28-qkv 31-values 36-merge-heads'),
 'causal-mask': Term('Causal Mask','پوشش علّی|پوشش علی|ماسک علّی|causal masking',
  'قاعده‌ای است که دسترسی هر موقعیت به Tokenهای آینده را در Attention می‌بندد.',
  'وقتی Token بعدی هنوز تولید نشده، مدل نباید در Training پاسخ را از آن قرض بگیرد.',
  'برای سطر i فقط ستون‌های j≤i مجازند. امتیاز ممنوع پیش از Softmax برابر منفی بی‌نهایت می‌شود؛ صفرکردن امتیاز کافی نیست.',
  'سطر اول فقط ستون اول را می‌بیند و وزن همان ستون ۱ است.',
  'آزمون future با تغییر آینده بررسی می‌کند که خروجی گذشته در eval تغییر نکند.', 'attention softmax self-attention context-window', '33-mask 34-causal-test 40-block'),
 'multi-head-attention': Term('Multi-Head Attention','multihead attention|توجه چندسر|توجه چند سر',
  'چند محاسبهٔ Attention موازی با Projectionهای قابل یادگیری متفاوت است.',
  'می‌خواهیم رابطه‌ها در چند فضای ویژگی بررسی شوند؛ تضمینی نیست هر Head معنای انسانی مشخصی داشته باشد.',
  'C=H×D است. Shape از (B,T,C) به (B,H,T,D) می‌رود و پس از ترکیب دوباره به (B,T,C) برمی‌گردد.',
  'با C=32 و H=4، هر Head هشت ویژگی دارد؛ ضریب مقیاس √8 است، نه √32.',
  'attention.py هم تقسیم ویژگی و هم جابه‌جایی محور زمان/Head را انجام می‌دهد.', 'attention query key value tensor', '35-split-heads 36-merge-heads 40-block'),
 'feed-forward-network': Term('Feed-Forward Network','FFN|شبکهٔ پیش‌خور|شبکه پیش‌خور',
  'زیرشبکه‌ای است که ویژگی‌های هر موقعیت را با تبدیل‌های خطی و یک تابع غیرخطی پردازش می‌کند.',
  'Attention اطلاعات را میان موقعیت‌ها جابه‌جا می‌کند؛ FFN روی ویژگی‌های هر موقعیت کار می‌کند.',
  'در این پروژه مسیر C → 4C → C با GELU داریم. یک FFN با وزن‌های مشترک روی همهٔ موقعیت‌ها اجرا می‌شود.',
  'ورودی (2,7,16) در میانه به (2,7,64) می‌رود و در انتها به (2,7,16) برمی‌گردد.',
  'خروجی FFN باید برای Residual Connection با ورودی هم‌شکل باشد.', 'layer transformer residual-connection', '37-ffn 40-block 44-parameters'),
 'dropout': Term('Dropout','دراپ‌اوت',
  'عملی است که هنگام Training بعضی مؤلفه‌ها را به‌صورت تصادفی صفر می‌کند.',
  'هدف، کاهش وابستگی شدید به مسیرهای خاص است؛ این کار تضمین بهبود هر مدل یا داده‌ای نیست.',
  'در حالت train، مؤلفه‌های باقی‌مانده بر 1−p تقسیم می‌شوند. eval این تصادف را خاموش می‌کند.',
  'اگر p=0.2 باشد، هر مؤلفه با احتمال ۲۰٪ حذف می‌شود؛ نه اینکه الزاماً دقیقاً ۲۰٪ هر Tensor صفر شود.',
  'وزن Attention پس از Dropout الزاماً مجموع یک ندارد؛ بازرس وزن پیش از آن را نشان می‌دهد.', 'layer attention training inference', '34-causal-test 40-block 48-evaluate'),
 'positional-embedding': Term('Positional Embedding','نمایش برداری موقعیت',
  'بردار قابل یادگیری‌ای است که شمارهٔ موقعیت را به ویژگی‌های ورودی اضافه می‌کند.',
  'جدول Token Embedding برای یک ID، مستقل از محل وقوعش، همان بردار را برمی‌گرداند؛ Positional Embedding اطلاعات جایگاه را به این بردار اضافه می‌کند؛ موقعیت اطلاعات تازه‌ای دربارهٔ جای Token می‌دهد.',
  'برای هر t یک بردار Cتایی برداشته می‌شود و با Token Embedding جمع می‌شود؛ محور تازه‌ای ساخته نمی‌شود.',
  '<code>x[:, t, :] = token_embedding + position_embedding[t]</code>.',
  'اندازهٔ جدول موقعیت با <code>context_length</code> محدود است؛ این قرارداد با Context Window هماهنگ می‌شود.', 'embedding token context-window', '26-positions 40-block 57-prompts'),
 'autograd': Term('Autograd','مشتق‌گیری خودکار|مشتق گیری خودکار',
  'سامانه‌ای برای ثبت عملیات و محاسبهٔ خودکار مشتق‌هاست.',
  'به‌جای نوشتن مشتق هر ترکیب پیچیده با دست، قواعد مشتق عملیات در یک گراف دنبال می‌شوند.',
  'requires_grad، اتصال گراف و زمان اجرای backward مهم‌اند. جداکردن با detach یا تبدیل نامناسب به عدد عادی می‌تواند مسیر مشتق را قطع کند.',
  '<code>x = torch.tensor(3., requires_grad=True)</code> و سپس <code>(x*x).backward()</code>، grad برابر ۶ می‌دهد.',
  'Training loop از Autograd استفاده می‌کند، اما update را Optimizer انجام می‌دهد.', 'gradient backpropagation tensor optimizer', '13-torch 17-autograd 46-gradient-path'),
 'training': Term('Training','',
  'فرایند تنظیم Parameterها با استفاده از داده و یک هدف عددی مشخص است.',
  'تفاوت مدل آموخته و قانون دست‌نویس این است که بخشی از تنظیم تابع با مشاهده‌ها انتخاب می‌شود.',
  'در حلقهٔ معمول، Gradient قبلی پاک می‌شود، خروجی و Loss ساخته می‌شوند، backward انجام می‌شود و Optimizer گام می‌زند.',
  '<code>zero_grad → forward → loss → backward → step</code>.',
  'train.py ارزیابی، Schedule و Checkpoint را هم در همین چرخه مدیریت می‌کند.', 'dataset loss optimizer validation-set inference', '01-model 47-loop 52-first-run'),
 'validation-set': Term('Validation Set','دادهٔ اعتبارسنجی|دادهٔ ارزیابی میانی',
  'داده‌ای کنارگذاشته‌شده است که برای سنجش رفتار مدل خارج از نمونه‌های به‌روزرسانی استفاده می‌شود.',
  'مدلی که متن Training را حفظ کرده ممکن است روی متن تازه ضعیف باشد؛ یک بخش جدا این تفاوت را آشکارتر می‌کند.',
  'این داده در Gradient update شرکت نمی‌کند، اما انتخاب تنظیم‌ها بر اساس آن هم می‌تواند باعث وابستگی به همان مجموعه شود؛ Test Set نقش نهایی جدا دارد.',
  'متن پیوسته ابتدا تقسیم می‌شود و سپس از هر بخش پنجره می‌سازیم تا مرزها قاطی نشوند.',
  'بهترین Checkpoint با معیار Validation انتخاب می‌شود، نه صرفاً Loss یک Batch آموزشی.', 'dataset loss overfitting checkpoint', '04-splits 24-data-contract 48-evaluate'),
 'overfitting': Term('Overfitting','بیش‌برازش|بیش برازش',
  'وضعیتی است که مدل با جزئیات دادهٔ Training سازگار می‌شود، اما بهبود مشابهی روی دادهٔ کنارگذاشته‌شده ندارد.',
  'حفظ پاسخ‌های تمرین با یادگیری قاعده‌ای که روی سؤال تازه کار کند، یکسان نیست.',
  'فاصلهٔ روند Training و Validation یک شاهد است؛ یک Batch پرنویز یا یک عدد منفرد برای تشخیص کافی نیست.',
  'ممکن است Loss آموزش پایین بیاید ولی Loss ارزیابی بعد از مدتی بالا برود.',
  'آزمایش one-batch عمداً حفظ یک Batch را می‌سنجد؛ برای عیب‌یابی مفید است، نه اثبات تعمیم.', 'training validation-set loss dropout', '04-splits 53-curves 61-one-batch'),
 'temperature': Term('Temperature','دمای نمونه‌گیری|دمای انتخاب|دما|دمای',
  'ضریبی مثبت برای تغییر تیزی توزیع انتخاب از Logits است.',
  'وزن مدل را عوض نمی‌کنیم؛ فقط میزان ترجیح گزینه‌های پرامتیاز را هنگام انتخاب تغییر می‌دهیم.',
  'توزیع از Softmax(logits/τ) ساخته می‌شود. τ کوچک‌تر معمولاً توزیع را تیزتر و τ بزرگ‌تر آن را تخت‌تر می‌کند.',
  'برای [2,0]، تقسیم بر 0.5 اختلاف را دو برابر می‌کند و گزینهٔ اول وزن بیشتری می‌گیرد.',
  'sampling.py مقدار نامعتبر را رد می‌کند؛ Greedy مسیر مستقل انتخاب بیشترین امتیاز است.', 'logits softmax sampling top-p', '55-temperature 56-topkp 57-prompts'),
 'top-k': Term('Top-k','',
  'روشی برای محدودکردن انتخاب به k گزینهٔ دارای بالاترین امتیاز است.',
  'بعضی گزینه‌های بسیار ضعیف را پیش از Sampling کنار می‌گذاریم، اما تعداد ثابت گزینه‌ها همیشه جرم احتمال ثابتی ندارد.',
  'پس از نگه‌داشتن گزینه‌ها، توزیع روی مجموعهٔ باقی‌مانده دوباره نرمال می‌شود. قرارداد تساوی امتیازها باید مشخص باشد.',
  'با k=2 از سه گزینه، فقط دو امتیاز بالاتر نامزد انتخاب می‌مانند.',
  'قرارداد Top-k پروژه در sampling.py و آزمون‌های آن تعریف شده است.', 'sampling top-p logits temperature', '56-topkp 57-prompts'),
 'top-p': Term('Top-p','Nucleus sampling',
  'گزینه‌های مرتب‌شده را تا رسیدن مجموع احتمال به آستانهٔ p نگه می‌دارد.',
  'تعداد نامزدها با میزان اطمینان مدل تغییر می‌کند؛ با Top-k که تعداد ثابتی دارد یکی نیست.',
  'گزینه‌ای که مجموع را از آستانه عبور می‌دهد نیز نگه داشته می‌شود و سپس توزیع دوباره نرمال می‌شود.',
  'برای [0.6,0.3,0.1] و p=0.8، دو گزینهٔ اول باقی می‌مانند چون مجموعشان 0.9 است.',
  'بازرس احتمال خام مدل و احتمال پس از فیلتر را جدا گزارش می‌کند.', 'sampling top-k softmax temperature', '56-topkp 57-prompts'),
 'kv-cache': Term('KV Cache','حافظهٔ موقت کلید و مقدار',
  'ذخیرهٔ Key و Value موقعیت‌های قبلی برای استفادهٔ دوباره هنگام تولید است.',
  'وقتی فقط یک Token تازه اضافه شده، نباید محاسبهٔ بخش‌های قابل استفادهٔ گذشته را بی‌دلیل تکرار کنیم.',
  'K و V تازه به مقدارهای ذخیره‌شده افزوده می‌شوند؛ Query فعلی با کل زمینهٔ مجاز کار می‌کند. مدیریت موقعیت و محدودیت پنجره ضروری است.',
  'پس از چهار Token، تولید Token پنجم می‌تواند K/V چهار موقعیت قبل را دوباره مصرف کند.',
  'Mini-GPT پایه Cache ندارد؛ این درس یک توسعهٔ بعدی و محدودیت‌های آن را توضیح می‌دهد.', 'key value attention context-window inference', '54-generate 64-cache'),
 'sft': Term('SFT','Supervised fine-tuning|تنظیم تکمیلی با پاسخ مرجع|تنظیم دقیق نظارت‌شده',
  'Fine-Tuning نظارت‌شده با نمونه‌هایی است که پاسخ مطلوب را مشخص می‌کنند.',
  'مدل از مثال‌های دستور و پاسخ می‌آموزد چگونه پاسخ بدهد؛ کیفیت و تنوع این مثال‌ها اهمیت دارد.',
  'در قرارداد رایج، Loss فقط روی Tokenهای پاسخ محاسبه می‌شود. Mask هدف با Causal Mask مسیر Attention فرق دارد.',
  'برای یک دستور و پاسخ، جایگاه‌های دستور می‌توانند از محاسبهٔ Loss کنار گذاشته شوند.',
  'کتاب قرارداد مفهومی و Mask هدف را توضیح می‌دهد؛ اجرای پایه سامانهٔ کامل SFT نیست.', 'fine-tuning loss causal-mask token', '65-sft 66-preference'),
 'lora': Term('LoRA','Low-rank adaptation|سازگارسازی با اصلاح کم‌رتبه',
  'روشی برای یادگیری یک اصلاح کم‌رتبه روی ماتریس وزن ثابت است.',
  'به‌جای قابل آموزش‌کردن همهٔ وزن‌ها، دو ماتریس کوچک‌تر را تنظیم می‌کنیم تا هزینهٔ Parameterهای قابل آموزش کمتر شود.',
  'در بیان ساده W′=W+BA است؛ رتبهٔ اصلاح با اندازهٔ میانی محدود می‌شود. پیاده‌سازی‌های عملی ممکن است ضریب مقیاس جدا داشته باشند.',
  'برای W با اندازهٔ d×k و رتبهٔ r، دو ماتریس r×k و d×r فقط r(k+d) مقدار دارند.',
  'درس LoRA یک مثال آموزشی مستقل دارد؛ به معنی وجود Fine-Tuning کامل در Training loop پایه نیست.', 'fine-tuning parameter layer', '65b-lora 65-sft'),
 'rag': Term('RAG','Retrieval-augmented generation|تولید با کمک متن بازیابی‌شده',
  'روشی است که متن‌های مرتبط را بازیابی و به زمینهٔ تولید اضافه می‌کند.',
  'برای استفاده از سند تازه همیشه لازم نیست وزن‌ها را تغییر دهیم؛ می‌توانیم شاهد مرتبط را در ورودی بگذاریم.',
  'بازیابی، انتخاب سند، جاگیری در Context Window و تولید مراحل متفاوت‌اند. سند بازیابی‌شده تضمین حقیقت یا تبعیت درست مدل نیست.',
  'بندی از راهنمای محصول را کنار سؤال می‌گذاریم و از مدل پاسخ مستند می‌خواهیم.',
  'RAG در پایان کتاب به‌عنوان مسیر مستقل توسعه معرفی شده؛ Mini-GPT کوچک کیفیت مدل عمومی را ندارد.', 'context-window inference fine-tuning embedding', '67-rag 65-sft'),
 'cross-entropy': Term('Cross-Entropy','Cross entropy|آنتروپی متقاطع',
  'معیاری برای سنجش ناسازگاری توزیع پیش‌بینی با توزیع هدف است.',
  'اگر هدف یک Token مشخص باشد، احتمال کمتر برای آن باید جریمهٔ بیشتری بدهد.',
  'برای هدف تک‌کلاسه، Loss برابر منفی لگاریتم احتمال هدف است. تابع PyTorch معمولاً Logits خام را می‌گیرد و محاسبهٔ پایدار را داخل خود انجام می‌دهد.',
  'احتمال هدف 0.25، Loss حدود 1.386 می‌دهد؛ احتمال 0.5، Loss حدود 0.693.',
  'پیش از F.cross_entropy دوباره Softmax نزنید؛ Shape و ID هدف را درست تحویل دهید.', 'loss logits softmax token', '10-entropy 43-lm-head 48-evaluate'),
 'vocabulary': Term('Vocabulary','واژگان مدل|واژگان',
  'مجموعهٔ Tokenهای شناخته‌شده و نگاشت ثابت آن‌ها به ID است.',
  'شمارهٔ هر Token فقط در همین قرارداد معنا دارد؛ دو Vocabulary متفاوت ممکن است برای یک عدد دو معنی متفاوت داشته باشند.',
  'اندازهٔ V هم تعداد سطرهای Token Embedding و هم اندازهٔ محور آخر Logits را تعیین می‌کند.',
  'در یک Vocabulary، ID=3 ممکن است «م» باشد؛ این شماره خاصیت جهانی حرف «م» نیست.',
  'Vocabulary از متن Training ساخته و در Checkpoint نگهداری می‌شود.', 'token tokenizer embedding logits checkpoint', '02-token 21-tokenizer 24-data-contract'),
 'neural-network': Term('Neural Network','شبکهٔ عصبی|شبکه عصبی',
  'ترکیبی از تبدیل‌های پارامتری و توابع غیرخطی است که از داده تنظیم می‌شود.',
  'چند تبدیل ساده در کنار هم می‌توانند رابطه‌ای بسازند که یک تبدیل خطی تنها نمی‌تواند بیان کند.',
  'ترکیب چند Linear بدون تابع غیرخطی همچنان خطی/آفین می‌ماند؛ عمق به‌تنهایی این محدودیت را رفع نمی‌کند.',
  'مسئلهٔ XOR با یک مرز خطی جداشدنی نیست، ولی شبکهٔ کوچک غیرخطی می‌تواند آن را یاد بگیرد.',
  'پیش از Transformer، یک شبکهٔ کوچک می‌سازیم تا Parameter، Loss و Optimizer را در عمل ببینیم.', 'layer parameter loss transformer', '01-learning 19-network 40-block'),
}

SUPPLEMENTAL = set()
# Usage audit, 2026-09-22. These sources establish Persian usage, not the
# correctness of every technical claim on their pages. Explanatory glosses are
# descriptions, not claims that a Persian phrase is the community's sole name.
USAGE_SOURCES = {
    'university': 'https://dsp-lab.ir/wp-content/uploads/2025/11/ML4NLP-HW1-1404-1.pdf',
    'datayad': 'https://datayad.com/supervised-machine-learning/',
    'howsam-llm': 'https://howsam.org/downloads/implementing-chatgpt-from-scratch-with-pytorch/',
    'tehrandata': 'https://tehrandata.org/courses/llm/',
    'avalai': 'https://docs.avalai.org/fa/guides/fine-tuning',
    'gradient': 'https://howsam.org/gradient-descent/',
    'university-ann': 'https://dsp-lab.ir/wp-content/uploads/2023/03/ML4NLP-Lecture6-ANN-RNN.pdf',
    'university-gate': 'https://dsp-lab.ir/wp-content/uploads/2022/01/ML4NLP-Lecture5-ANN.pdf',
    'howsam-gate': 'https://howsam.org/lstm-neural-network/',
    'residual-course': 'https://aminmazi.ir/learn/gpt-course/',
    'residual-guide': 'https://bardia.ai/ai-course/',
}
# category: A established Persian, B English-first, C both, D replaced literal,
# E ambiguous usage, F corrected meaning, G context-sensitive technical wording.
USAGE_DECISIONS = {
    'supervised-learning': ('D', 'Supervised learning (یادگیری نظارت‌شده)', 'high',
        ('university','datayad'), 'Replace یادگیری با هدف مرجع; it was a description, not the conventional name.'),
    'self-supervised-learning': ('C', 'Self-supervised learning (یادگیری خودنظارتی)', 'high',
        ('university','howsam-llm'), 'Use the recognizable name; automatic targets are still targets.'),
    'fine-tuning': ('D', 'Fine-Tuning (تنظیم دقیق)', 'high',
        ('tehrandata','avalai'), 'Replace تنظیم تکمیلی as the introductory label; retain English in later prose.'),
    'sft': ('D', 'SFT (تنظیم دقیق نظارت‌شده)', 'high',
        ('university','avalai','tehrandata'), 'Keep SFT and explain supervised fine-tuning, not an invented task family.'),
    'gradient': ('C', 'Gradient (گرادیان)', 'high',
        ('gradient','university-ann'), 'Keep Gradient/گرادیان distinct from a scalar derivative.'),
    'embedding': ('C', 'Embedding with a functional Persian explanation', 'high',
        ('university-ann','tehrandata'), 'تعبیه and بردار تعبیه occur; the English label avoids implying every representation is an Embedding.'),
    'attention': ('C', 'Attention (توجه)', 'high',
        ('university-ann','tehrandata'), 'Keep the conventional gloss; distinguish numerical Attention from human attention.'),
    'encoder': ('C', 'Encoder (رمزگذار)', 'high',
        ('university-ann','tehrandata'), 'Both occur; distinguish a network from tokenizer encoding.'),
    'decoder': ('C', 'Decoder (رمزگشا)', 'high',
        ('university-ann','tehrandata'), 'Both occur; do not remove a conventional Persian gloss.'),
    'gate': ('C', 'Gate with explanatory دریچه', 'medium',
        ('university-gate','howsam-gate'), 'دروازه, گیت and explanatory دریچه coexist; retain the English identity.'),
    'residual-connection': ('C', 'Residual Connection (اتصال باقی‌مانده)', 'medium',
        ('residual-course','residual-guide'), 'The Persian gloss has independent authored usage; keep it.'),
    'context-window': ('E', 'Context Window with a functional Persian explanation', 'medium',
        (), 'پنجرهٔ زمینه and پنجرهٔ بافت vary; evidence did not justify claiming one dominant Persian term.'),
    'pre-norm': ('E', 'Pre-Norm', 'medium',
        (), 'Sparse independent Persian usage; explain normalization before each sublayer, not an invented formal translation.'),
}

# Count concept-level corrections, not individual string replacements or the
# 16 conventional Persian names restored by the prose renderer.
CORRECTED_CONCEPTS = {
    'supervised-learning', 'self-supervised-learning', 'fine-tuning', 'sft',
    'parameter-efficient-fine-tuning', 'inference', 'one-hot', 'gradient-clipping',
    'gradient', 'autograd', 'positional-embedding', 'buffer',
    'feed-forward-network', 'query', 'key', 'value', 'loss', 'softmax',
    'pre-norm', 'language-model-head',
}


def terminology_inventory():
    """Call after extend_from_lessons: inventory every actual glossary concept."""
    from .terminology import CONVENTIONAL_PERSIAN
    inventory = {}
    english_first = {'tensor','token','tokenizer','transformer','batch','epoch',
                     'logits','softmax','dropout','checkpoint','optimizer','backpropagation'}
    for slug, term in TERMS.items():
        category, canonical, confidence, sources, reason = USAGE_DECISIONS.get(slug,
            ('A' if slug in CONVENTIONAL_PERSIAN else 'F' if slug in CORRECTED_CONCEPTS
             else 'B' if slug in english_first else 'G', term.name,
             'high' if slug in CONVENTIONAL_PERSIAN else 'context-reviewed', (),
             'Preserve established Persian wording.' if slug in CONVENTIONAL_PERSIAN else
             'Retain the English identity and its context-specific explanation; no frequency claim.'))
        inventory[slug] = dict(english=term.name, aliases=term.aliases,
            usage=term.meaning, category=category, canonical=canonical,
            confidence=confidence, evidence=[USAGE_SOURCES[key] for key in sources],
            reason=reason, corrected=slug in CORRECTED_CONCEPTS)
    return inventory

EDITORIAL_ALIASES = {
    'supervised-learning':'یادگیری نظارت‌شده|یادگیری نظارت شده|یادگیری با ناظر',
    'self-supervised-learning':'یادگیری خودنظارتی|یادگیری خودنظارت‌شده|یادگیری خودناظر',
    'scalar':'اسکالر', 'vector':'بردار', 'matrix':'ماتریس',
    'dot-product':'ضرب داخلی', 'matrix-multiplication':'ضرب ماتریسی',
    'transpose':'ترانهاده', 'derivative':'مشتق', 'partial-derivative':'مشتق جزئی',
    'chain-rule':'قاعدهٔ زنجیره‌ای|قاعده زنجیره‌ای',
    'gradient-descent':'گرادیان‌کاهشی|گرادیان کاهشی',
    'computational-graph':'گراف محاسبه|گراف محاسباتی',
    'subword':'زیرواژه', 'corpus':'پیکره', 'bias':'بایاس',
    'head':'سر توجه', 'cross-attention':'توجه میان دو منبع',
    'learning-rate-schedule':'برنامهٔ تغییر نرخ یادگیری',
    'underfitting':'کم‌برازش|کم برازش', 'data-leakage':'نشت داده',
    'perplexity':'سرگشتگی', 'quantization':'کم‌بیت‌سازی|کوانتیزه‌سازی',
    'mixed-precision':'دقت عددی ترکیبی', 'distributed-training':'آموزش توزیع‌شده',
    'instruction-tuning':'تنظیم روی دستور', 'weight-decay':'کاهش تدریجی وزن',
}


def definition_sentence(paragraph, definition):
    """Choose a complete sentence without splitting tags or code expressions."""
    chunks, current, depth = [], '', 0
    for piece in re.split(r'(<[^>]+>)', paragraph):
        if piece.startswith('<'):
            if piece.startswith('</'):
                depth = max(0, depth-1)
            elif not piece.endswith('/>') and not re.match(r'<(?:br|img|hr)\b', piece):
                depth += 1
            current += piece
        elif depth:
            current += piece
        else:
            sentences = re.split(r'(?<=[.؟!])\s+', piece)
            for index, sentence in enumerate(sentences):
                current += sentence
                if index < len(sentences)-1:
                    chunks.append(current.strip())
                    current = ''
    chunks.append(current.strip())
    return next((chunk for chunk in chunks if f'<dfn>{definition}</dfn>' in chunk), paragraph)


def extend_from_lessons(lessons):
    """Keep the book's existing definitions/examples, with stable concept URLs.

The hand-authored entries above explain the core independently. Other existing
definitions reuse their actual teaching context, worked answer and project link,
not invented dictionary filler. Their provenance is shown on the resulting page.
"""
    known = {alias.casefold():slug for slug,t in TERMS.items()
             for alias in (t.name,*t.aliases.split('|')) if alias}
    collected = []
    by_lesson = {}
    for lesson in lessons:
        paragraphs = re.findall(r'<p>(.*?)</p>',lesson.body,re.S)
        for paragraph_index, paragraph in enumerate(paragraphs):
            for raw in re.findall(r'<dfn>(.*?)</dfn>',paragraph):
                match = re.match(r'[A-Za-z][A-Za-z0-9 ._/-]*',raw)
                if not match:
                    continue
                name = match[0].strip()
                slug = known.get(name.casefold())
                if slug is None:
                    slug = re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')
                    known[name.casefold()] = slug
                    aliases = []
                    alternate = re.search(r' یا ([A-Za-z][A-Za-z0-9 -]*)',raw)
                    if alternate and alternate[1].strip().casefold() not in {'exp','log','pad','bos','eos'}:
                        aliases.append(alternate[1].strip())
                    collected.append((slug,name,'|'.join(aliases),lesson,paragraphs,paragraph_index,raw))
                by_lesson.setdefault(lesson.id,[]).append(slug)
    for slug,name,aliases,lesson,paragraphs,index,definition in collected:
        context = paragraphs[index]
        meaning = definition_sentence(context, definition)
        # Keep the defining paragraph and its adjacent mathematical context.
        # A following paragraph alone can refer to assumptions/equations it omits.
        start = lesson.body.index('<p>'+context+'</p>')
        blocks = re.findall(r'<p>.*?</p>|<div\b[^>]*>.*?</div>|<table>.*?</table>',
                            lesson.body[start:], re.S)
        detail = ''.join(blocks[:3])
        # The exercise can depend on earlier values in its lesson. Make that
        # complete context available, rather than presenting an orphaned answer.
        detail += ('<details class="concept-source-context"><summary>زمینهٔ کامل درس برای دنبال‌کردن مثال</summary>'
                   + lesson.body + '</details>')
        related = list(dict.fromkeys(s for s in by_lesson[lesson.id] if s != slug))[:5]
        if not related:
            related = ['tensor','parameter']
        aliases = '|'.join(a for a in (aliases,EDITORIAL_ALIASES.get(slug,'')) if a)
        example = '<p>تمرینِ درس مبدأ: '+lesson.task+'</p><p>راهنمای پاسخ: '+lesson.answer+'</p>'
        TERMS[slug] = Term(name,aliases,meaning,lesson.objective,detail,example,
                          lesson.project,' '.join(related),lesson.id)
        SUPPLEMENTAL.add(slug)

    from .glossary_examples import FOCUSED_EXAMPLES
    for slug, fields in FOCUSED_EXAMPLES.items():
        if slug not in TERMS:
            raise ValueError("Unknown curated concept: "+slug)
        TERMS[slug] = replace(TERMS[slug], **fields)
