"""The controller bridge; later system lessons are maintained separately."""
from .schema import Lesson


CONTROLLER = Lesson("80-controller",14,"مدل درون سامانه","چه کسی تصمیم می‌گیرد یک گام دیگر اجرا شود؟",
"Context، شاهد، حافظه و ابزار را به یک حلقهٔ محدود وصل می‌کنیم؛ پایان اجرا را از درستی پاسخ جدا نگه می‌داریم.",
"""<p>دستیار راهنمای دوره یک پاسخ ابزار گرفته است: ۱۲۰۰ دقیقه. حالا چه کسی تصمیم می‌گیرد پاسخ نهایی نوشته شود، محاسبهٔ دیگری انجام شود یا برنامه متوقف شود؟ خودِ تابع <code>MiniGPT.forward</code> چنین حلقه‌ای ندارد. این تابع هنوز همان ورودی Tokenها را به Logits تبدیل می‌کند. تصمیمِ تکرار، اجرای ابزار و نگهداری اطلاعات را برنامه‌ای بیرون مدل مدیریت می‌کند.</p>
<p>پیش از نوشتن حلقه، یک بار آن را دستی طی کنید. سؤال و سندهای مرتبط را آماده کنید؛ پیشنهاد <code>multiply(25,48)</code> را بررسی کنید؛ ابزار نتیجهٔ ۱۲۰۰ می‌دهد؛ نتیجه را همراه سؤال در Context تازه بگذارید؛ سپس پاسخ نهایی را بگیرید و با معیار مسئله مقایسه کنید. هر پیکان یک ورودی و خروجی قابل ثبت دارد. اگر درخواست دوم دقیقاً همان محاسبهٔ اول باشد، لزوماً با «تلاش بیشتر» پیشرفتی نکرده‌ایم.</p>
<div class="flow">سؤال + شاهد + حافظه → ساخت Context → پیشنهاد → اعتبارسنجی → اجرای ابزار → مشاهدهٔ نتیجه → پیشنهاد بعدی یا پاسخ → وارسی → پایان</div>
<p><dfn>Controller (کد کنترل مراحل سامانه)</dfn> در پروژه همین گردش را با حالت‌های مشخص اجرا می‌کند. <code>retrieve</code> شناسهٔ شاهدهای پیدا‌شده را ثبت می‌کند؛ <code>observe</code> نشان می‌دهد کدام قطعه‌ها واقعاً در Context جا گرفته‌اند؛ <code>propose</code> خروجی Backend است؛ <code>validate</code> قرارداد ابزار را می‌سنجد؛ <code>execute</code> نتیجهٔ واقعی را ثبت می‌کند؛ <code>verify</code> معیار مستقل پاسخ را به کار می‌برد. لازم نیست یک Framework نصب کنیم تا معنای این حالت‌ها را بفهمیم.</p>
<p>این بار همهٔ قطعه‌ها جای مشخصی دارند. <code>keyword_search</code> از سندهای کوچک دوره شاهد می‌آورد. برای حافظه، برنامهٔ فراخوان باید <code>MemoryStore</code> جداگانهٔ کاربر را بدهد و کلیدهای مجاز برای همین کار را در <code>memory_keys</code> انتخاب کند. بدون این انتخاب، هیچ رکوردی وارد Context نمی‌شود؛ کلید ناموجود هم نادیده گرفته می‌شود. Controller خودش رضایت کاربر یا مرتبط‌بودن رکورد را حدس نمی‌زند و از پاسخ، حافظهٔ تازه نمی‌سازد. <code>build_context</code> کل متنِ دارای مرزبندی را با Tokenizer مربوط می‌شمارد و برای پاسخ جا کنار می‌گذارد. ابزارها فقط سه محاسبهٔ آفلاینِ درس قبل‌اند. هیچ‌یک از این کارها وزن‌های Transformer را عوض نمی‌کند.</p>
<p>در این API، اولویت Context با قرارداد اجرا و نتیجه‌های ابزارِ قبلاً اجراشده است؛ سپس شاهدها و رکوردهای حافظه می‌آیند. حذف هر قطعه ثبت می‌شود. اگر خود سؤال، قرارداد یا نتیجهٔ ضروری در بودجه جا نشود، وضعیت <code>context_limit</code> می‌گیریم؛ بخش مهمی را بی‌خبر از ابتدا نمی‌بُریم. در مدل کوچک با Context خیلی کوتاه، این توقف نتیجهٔ قابل انتظار است. بزرگ‌تر اعلام‌کردن عدد بودجه، ظرفیت واقعی Position Embedding را زیاد نمی‌کند.</p>
<p><dfn>Agent (سامانهٔ هدف‌محور با امکان انتخاب و تکرار اقدام)</dfn> در کاربردهای مختلف تعریف یکسانی ندارد. در این کتاب از یک الگوی محدود حرف می‌زنیم: پیشنهاد مرحلهٔ بعد می‌تواند از مدل بیاید، اما مجوز اجرا و سقف تکرار را Controller تعیین می‌کند. در <a href="https://arxiv.org/abs/2210.03629">ReAct</a> نیز پیوند پیشنهادهای مرحله‌ای با اقدام و مشاهده بررسی شده است؛ برنامهٔ آموزشی ما بازتولید آن پژوهش یا یک عامل مستقلِ عمومی نیست.</p>
<p>دو Backend را عمداً جدا کرده‌ایم. <code>ScriptedFixture</code> فهرستی از پاسخ‌های ازپیش‌نوشته‌شده دارد و نامش در گزارش می‌آید؛ با آن می‌سنجیم آیا اتصال قطعه‌ها درست کار می‌کند. <code>MiniGPTBackend</code> واقعاً با Tokenizer، Tensor ورودی و <code>model.generate</code> ادامهٔ متن می‌سازد. اگر Mini-GPT نتواند JSON معتبر بدهد، نتیجه <code>invalid_action</code> است؛ هیچ پاسخ آماده‌ای جای آن قرار نمی‌گیرد. آزمایش پایین یک مدل تازه با وزن تصادفی و Context کافی برای کل قرارداد می‌سازد و خروجی خامش را نشان می‌دهد؛ این فقط آزمایش اتصال است، نه شاهدِ دستورپذیری. Checkpoint کوچکِ آزمایش SFT ممکن است همین Prompt را هم جا ندهد؛ موفقیت آن در کار آموزشیِ خودش، سازگاری با این پروتکلِ طولانی‌تر را تضمین نمی‌کند.</p>
<p>حلقهٔ بی‌پایان نسازید. حداکثر تعداد پیشنهاد و فراخوانی ابزار را جدا تعیین کنید. در این نمونه، درخواست تکراری با همان Arguments توقف می‌آورد؛ JSON نامعتبر و ابزار نامجاز نیز اجرا نمی‌شوند. این سیاست محافظه‌کارانه برای ماشین‌حساب بدون عارضهٔ جانبی مناسب است. در سامانهٔ واقعیِ دارای فایل، ایمیل یا پرداخت، مجوز کاربر، دسترسی محدود، Timeout، ثبت رویداد و جلوگیری از اجرای تکراریِ عملیات هم لازم‌اند؛ ابزار آفلاین ما وجود آن مسائل را حذف نمی‌کند.</p>
<p>در پایان، <code>status='finished'</code> فقط یعنی پروتکل به پاسخ رسیده است. بدون Verifier، فیلد <code>verified</code> همچنان False می‌ماند. معتبر بودن شناسهٔ ارجاع هم تنها نشان می‌دهد آن شاهد واقعاً در Context بوده؛ ثابت نمی‌کند هر جملهٔ پاسخ از آن نتیجه می‌شود. برای حسابِ ۲۵×۴۸ می‌توان معیار دقیق داشت. برای توضیح دوره، باید درستی پاسخ و نسبتش با منبع را جدا ارزیابی کنیم؛ همین نیاز، موضوع درس بعد است.</p>""",
"یک Fixture با درخواست ضرب و پاسخ نهایی بسازید. ابتدا بدون Verifier و سپس با معیار ۱۲۰۰ اجرا کنید. بعد پاسخ نهایی را ۱۲۰ بگذارید؛ درخواست تکراری و سقف صفرِ ابزار را نیز جدا امتحان کنید.",
"پاسخ بدون Verifier می‌تواند finished ولی verified=False باشد. پاسخ ۱۲۰ باید با معیار مستقل verification_failed بدهد. درخواست تکراری repeated_action و سقف صفر tool_limit می‌دهد؛ ابزار دیگری در سکوت اجرا نمی‌شود.",
"اگر Fixture همهٔ آزمون‌ها را بگذراند ولی MiniGPTBackend نتواند قالب معتبر بسازد، دربارهٔ کدام بخش موفق شده‌ایم و کدام قابلیت هنوز ثابت نشده است؟",
"assistant.py قطعه‌های context.py، retrieval.py، memory.py، tools.py و model.generate را وصل می‌کند. trace سامانه با trace Tensorهای داخل Transformer متفاوت است.",
code='''from mini_gpt.assistant import ScriptedFixture, run_assistant
responses = [
    {"action":"tool","name":"multiply","arguments":{"a":25,"b":48}},
    {"action":"finish","answer":"1200","citations":[]},
]
backend = ScriptedFixture(responses)
result = run_assistant("25 * 48 minutes?",backend,
                       verify_answer=lambda answer,citations,tools: answer == "1200")
assert result.status == "finished" and result.verified
assert result.tool_results[0]["value"] == 1200
print(result.backend)
print([event["state"] for event in result.events])

# A separate real, UNTRAINED MiniGPT with room for the whole protocol.
import torch
from mini_gpt.assistant import MiniGPTBackend, PROTOCOL
from mini_gpt.context import ContextItem, build_context
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharacterTokenizer
torch.set_num_threads(1)
torch.manual_seed(41)
question = "25*48?"
pack = build_context(question,[ContextItem("controller:protocol",PROTOCOL,"instruction")],
                     max_tokens=4096,reserve_tokens=4,count_tokens=len)
tokenizer = CharacterTokenizer.from_text(pack.prompt)
model = MiniGPT(ModelConfig(tokenizer.vocab_size,len(pack.prompt)+8,8,2,1,0.))
real = MiniGPTBackend(model,tokenizer,max_new_tokens=4)
observed = run_assistant(question,real)
raw = next(event["text"] for event in observed.events if event["state"] == "propose")
assert real.calls == 1 and observed.status == "invalid_action"
assert not observed.tool_results
print(real.label, "UNTRAINED raw output:",repr(raw),observed.status,real.last_counts)
''', source="mini_gpt/assistant.py",
review="پیش از افزودن قابلیت تازه، چند سؤال ثابت و چند شکست معلوم برای همین سامانه ثبت کنید. نمایشی که یک بار خوب جواب می‌دهد هنوز معیارِ قابل مقایسه نیست.")

LESSONS = [CONTROLLER]
