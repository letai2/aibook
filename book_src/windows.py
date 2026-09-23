"""Reader-facing Windows instructions; full English guide ships in the model ZIP."""
WINDOWS = r'''<p class="objective">برای خواندن کتاب فقط مرورگر لازم است؛ اجرای مدل و ساخت کتاب دو کار جدا هستند.</p>
<table><tr><th>هدف</th><th>نیاز</th></tr><tr><td>مطالعهٔ کتاب</td><td>مرورگر به‌روز؛ بدون نصب Python، Node یا PyTorch</td></tr><tr><td>پیش‌نمایش localhost</td><td>Python 3.11 یا جدیدتر؛ بدون بستهٔ اضافی</td></tr><tr><td>نسخهٔ شمارشی v0</td><td>فقط Python</td></tr><tr><td>مدل عصبی</td><td>Python 3.11 و PyTorch؛ CPU کافی است</td></tr><tr><td>ساخت و بسته‌بندی برای نویسنده</td><td>Python و Node.js 24 LTS؛ بدون npm install</td></tr></table>
<h2>۱. فایل درست را استخراج کنید</h2>
<p>در Windows 10/11، با Extract All کل ZIP را باز کنید. نسخهٔ ایستای کتاب در ریشهٔ خود <code>index.html</code> دارد؛ در مخزن کامل، کتاب در <code>dist</code> است. بستهٔ [[downloads/mini-gpt-project.zip|پروژهٔ قابل دانلود]] شامل <code>mini_gpt</code>، <code>data</code>، آزمون‌های مدل و <code>docs/WINDOWS_SETUP.md</code> است؛ منبع کتاب، ابزار ساخت، دفترهای Jupyter و <code>run.py</code> هم در آن هستند؛ در اولین اجرا کتاب ساخته می‌شود. راهنمای انگلیسیِ کامل با جزئیات نصب و رفع خطا در همان فایل آمده است.</p>
<p>می‌توانید <code>index.html</code> را مستقیم باز کنید. قلم، تصویرها و محاسبه‌های مرورگر محلی‌اند. ذخیرهٔ پیشرفت در file:// به مرورگر بستگی دارد؛ برای پیوستگی میان صفحه‌ها، localhost و خروجی JSON توصیه می‌شود.</p>
<h2>۲. Python را فقط در صورت نیاز نصب کنید</h2>
<p>از <a href="https://www.python.org/downloads/windows/">منبع رسمی Python</a> و <a href="https://docs.python.org/3/using/windows.html">راهنمای Windows</a> استفاده کنید. با Python Install Manager می‌توانید runtime نسخهٔ 3.11 را نصب کنید: <code>py install 3.11</code>. نصب خود Windows یا Installer در این بازبینی اجرا نشده؛ فرمان‌های پروژه در محیط مجازی تازه روی Windows موجود آزموده می‌شوند. مسیر مدلِ آزموده‌شده ۶۴بیتی x86 است؛ ARM و GPU در این محیط آزموده نشده‌اند.</p>
<pre><code>python --version
python -c "import sys, struct; print(sys.executable); print(struct.calcsize('P')*8)"</code></pre>
<p>برای مسیر آزموده‌شده، نسخهٔ 3.11.x و عدد ۶۴ را انتظار داریم. اگر <code>python</code> نسخهٔ دیگری است یا Store را باز می‌کند، برای ساخت محیط از <code>py -3.11</code> استفاده کنید. پس از نصب، ترمینال را دوباره باز کنید. مسیر مفسر از عنوان پنجره مهم‌تر است.</p>
<h2>۳. وارد پوشهٔ پروژه شوید</h2>
<p>این مسیر نمونه را با محل واقعی خود عوض کنید. فاصله و کاراکترِ فارسی در نام پوشه مجاز است.</p>
<h3>PowerShell</h3><pre><code>Set-Location -LiteralPath 'C:\Books\MiniGPT'</code></pre>
<h3>Command Prompt (CMD)</h3><pre><code>cd /d "C:\Books\MiniGPT"</code></pre>
<h2>۴. سرور محلی اختیاری</h2>
<p>از مخزن کامل:</p><pre><code>python -m http.server 8000 --bind 127.0.0.1 --directory dist</code></pre>
<p>از پوشهٔ ZIP ایستای استخراج‌شده:</p><pre><code>python -m http.server 8000 --bind 127.0.0.1 --directory .</code></pre>
<p>نشانی <a href="http://127.0.0.1:8000/">http://127.0.0.1:8000/</a> را باز کنید. ترمینال باید باز بماند؛ توقف با Ctrl+C است. این سرور فقط روی رایانهٔ خودتان گوش می‌دهد، نه میزبان عمومی. تغییر پورت یا مرورگر، محل ذخیرهٔ پیشرفت را تغییر می‌دهد؛ پیش از جابه‌جایی خروجی بگیرید.</p>
<h2>۵. اجرای مستقل مدل در همان محیط پروژه</h2>
<p>از پوشهٔ دارای <code>mini_gpt</code> و <code>data</code> اجرا کنید. دستورهای مسیر صریح در هر دو ترمینال کار می‌کنند و به فعال‌سازی نیاز ندارند. به‌روزرسانی <code>pip</code> مهم است: نسخهٔ قدیمیِ همراه Python این محیط در خواندن اطلاعات یک وابستگی خطا داشت.</p>
<pre><code>python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__)"
.\.venv\Scripts\python.exe -B -m mini_gpt.smoke_test
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v</code></pre>
<p>نصب اینترنت می‌خواهد، اجرای بعدی محلی است. برای مدل، CUDA لازم نیست؛ نصب کامل دفترها وابستگی‌های نمودار، از جمله NumPy، را هم فراهم می‌کند. فقط فایل Checkpoint مورد اعتماد خودتان را بارگذاری کنید.</p>
<h2>۶. استفاده از فرمان‌های کوتاهِ درس‌ها</h2>
<p>دستورهای درس‌ها که با <code>python</code> شروع می‌شوند، فرض می‌کنند محیط فعال است. در PowerShell:</p><pre><code>.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"</code></pre>
<p>در CMD:</p><pre><code>.venv\Scripts\activate.bat
python -c "import sys; print(sys.executable)"</code></pre>
<p>اگر PowerShell فعال‌سازی را مسدود کرد، سیاست امنیتی سراسری را تغییر ندهید؛ به‌جای <code>python</code> از <code>.\.venv\Scripts\python.exe</code> استفاده کنید. <code>deactivate</code> محیط فعال را ترک می‌کند.</p>
<pre><code>python -m mini_gpt.train --output runs/windows-check --steps 6 --eval-every 2 --context-length 8 --embedding-dim 16 --num-heads 2 --num-layers 1 --batch-size 2 --device cpu --threads 1
python -m mini_gpt.evaluate --checkpoint runs/windows-check/last.pt --device cpu
python -m mini_gpt.generate --checkpoint runs/windows-check/last.pt --prompt "مدل " --tokens 12 --greedy --device cpu</code></pre>
<p>پوشهٔ خروجی باید تازه باشد. شش گام آزمون اتصال‌هاست، نه وعدهٔ کیفیت متن. برای بازرسی و ادامهٔ اجرا، [[47-loop|حلقهٔ آموزش]]، [[51-resume|آزمایش Resume]] و [[52-first-run|اولین اجرای قابل ثبت]] را دنبال کنید.</p>
<h2>۷. رفع خطا</h2>
<ul><li>خطای <code>import</code>: مسیر Python نصب و اجرا را یکی کنید.</li><li>فایل یا پوشهٔ <code>data</code> پیدا نشد: از ریشهٔ پروژه اجرا کنید، نه داخل <code>mini_gpt</code>.</li><li>پورت اشغال است: سرور قبلی خود را متوقف کنید یا پورت و URL را هر دو عوض کنید.</li><li>خروجی فارسی: CLIهای پروژه UTF-8 هستند. برای اسکریپت شخصی از <code>python -X utf8 lesson.py</code> استفاده کنید. برای JSON از گزینهٔ <code>--output</code> استفاده کنید؛ Windows PowerShell 5 ممکن است خروجی &gt; را با کدگذاری دیگری بنویسد.</li><li>CSS یا قلم دیده نمی‌شود: کل ZIP را استخراج کنید، نه فقط <code>index.html</code> را.</li><li>پوشهٔ اجرا غیرخالی است: نام تازه انتخاب کنید یا آموزش همان اجرا را از <code>last.pt</code> ادامه دهید؛ سابقه را برای رفع خطا پاک نکنید.</li></ul>
<h2>۸. ساخت و انتشار برای نویسنده</h2>
<p>فقط مخزن کامل: Python 3.11+ و <a href="https://nodejs.org/en/download">Node.js 24 LTS</a> لازم‌اند؛ بستهٔ npm نداریم. پس از نصب Node، ترمینال را بازگشایی و <code>node --version</code> را بررسی کنید.</p>
<pre><code>python -B -m tools.build_book
python -B -m tools.validate_book
python -B -m tools.prepare_release</code></pre>
<p>بستهٔ <code>release/book-site.zip</code> فقط فایل‌های عمومی را دارد. مخزن، venv، archive و اجراهای مدل را منتشر نکنید. آزمودن پروژه در یک محیط مجازی تازه روی Windows موجود، وابستگی به بسته‌های نصب قبلی را کمتر می‌کند؛ اما جای آزمون روی یک Windows تازه‌نصب‌شده را نمی‌گیرد. جزئیات شواهد و محدودیت‌ها در گزارش آموزشی مخزن ثبت می‌شود.</p><p>برای دفترهای تعاملی، [[notebooks.html|راهنمای Jupyter]] را دنبال کنید؛ همان محیط پروژه را به کار ببرید.</p>'''
