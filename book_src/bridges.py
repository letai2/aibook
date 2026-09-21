"""Prerequisite reasoning before library abstraction."""
from .schema import Lesson

VECTOR_OPERATIONS = Lesson(
"05a-vector-operations", 2, "ساختار عددها", "از عدد تکی تا جمع وزن‌دار Vectorها",
"نمادهای جمع را به حلقهٔ Python وصل کنیم و جمع، ضرب عضو‌به‌عضو و ترکیب چند Vector را از هم جدا کنیم.",
"""<p>دو Vector داریم: <code>u=[1,2]</code> و <code>v=[3,-1]</code>. اگر بخواهیم آن‌ها را جمع کنیم، باید عددهایشان را یکی‌یکی جمع کنیم یا همه را در یک فهرست بگذاریم؟ پاسخ ریاضی با رفتار <code>list</code> در Python یکی نیست. اول محاسبه را با دست انجام می‌دهیم؛ بعد می‌بینیم کدام کد همان کار را می‌کند.</p>
<p>برای خواندن محاسبه‌ها، چند نماد را روشن کنیم. نماد u₀ همان خانهٔ <code>u[0]</code> است. زیرنویس شمارهٔ خانه است، نه توان؛ 2³ یعنی ۲×۲×۲. مساویِ ریاضی برابری دو مقدار را می‌گوید؛ در دستور Python مانند <code>u = [1,2]</code> نام را به مقدار نسبت می‌دهیم.</p>
<p><dfn>Vector addition (جمع مؤلفه‌های متناظر)</dfn> عددهای هم‌جای دو Vector را جمع می‌کند: <code>u+v=[4,1]</code>. در <dfn>Scalar multiplication (ضرب همهٔ مؤلفه‌ها در یک عدد)</dfn>، <code>2u=[2,4]</code> و <code>-u=[-1,-2]</code> است. تفریق را می‌توان جمع با مقدار منفی دید. برای جمع و ضرب عضو‌به‌عضو، دو Vector هم‌اندازه می‌خواهیم؛ ضرب Scalar در Vector چنین شرطی ندارد. بعداً Broadcasting را با قرارداد جداگانه می‌آموزیم.</p>
<p><dfn>Element-wise multiplication (ضرب مؤلفه‌های متناظر)</dfn> نتیجهٔ <code>[3,-2]</code> می‌دهد؛ هنوز دو ویژگی داریم. جمع این دو حاصل، Scalar برابر ۱ است؛ این مقدمهٔ Dot product در درس بعد است. هر ضربی Matrix multiplication نیست.</p>
<div class="math">u + v = [4,1]<br>2u = [2,4]<br>u ⊙ v = [3,−2]<br>Σ(i=0…D−1) u[i] = u[0] + … + u[D−1]</div>
<p>Σ فرمان «جمع کن» است. D تعداد مؤلفه‌ها و i شمارهٔ مؤلفه است؛ با شروع از صفر، آخرین شماره D−1 است. برای u، حاصل ۱+۲=۳ است. نماد ⊙ در این درس ضرب عضو‌به‌عضو را نشان می‌دهد؛ @ را بعداً برای ضرب ماتریسی به کار می‌بریم.</p>
<p>در <dfn>Weighted sum (جمع وزن‌دار)</dfn> ابتدا هر Vector را در ضریبش ضرب و سپس مؤلفه‌های متناظر را جمع می‌کنیم: <code>0.25u+0.75v=[2.5,-0.25]</code>. مؤلفهٔ اول ۰٫۲۵×۱+۰٫۷۵×۳ و مؤلفهٔ دوم ۰٫۲۵×۲+۰٫۷۵×(−۱) است. خروجی یک Vector دو‌مؤلفه‌ای است، نه یک Scalar یا چهار ویژگی. چون ضریب‌ها نامنفی‌اند و جمعشان یک است، این مثال میانگین وزن‌دار هم هست؛ هر Weighted sum الزاماً میانگین نیست.</p>
<p>حالا به سؤال اول برگردیم. جمع دو list معمولی آن‌ها را به هم می‌چسباند و <code>2*u</code> فهرست را تکرار می‌کند؛ هیچ‌کدام محاسبهٔ عددیِ موردنظر ما نیست. فعلاً این محاسبه را با حلقه می‌نویسیم و بعداً آن را به Tensor می‌سپاریم. البته برابر‌بودن Shape هم کافی نیست: اگر ویژگی‌های یک Vector «جرم و زمان» و دیگری «طول و قیمت» باشند، باید اول معلوم کنیم جمعشان چه معنایی دارد.</p>""",
"جمع، دوبرابرکردن، ضرب عضو‌به‌عضو و ترکیب ۰٫۲۵/۰٫۷۵ را پیش از اجرا حساب کنید. اگر ضریب‌ها ۱ و صفر شوند، نتیجه و Shape چه می‌شود؟ جمع listهای معمولی چه می‌دهد؟",
"نتیجه‌ها [4,1]، [2,4]، [3,-2] و [2.5,-0.25] هستند. ضریب‌های ۱ و صفر، خود u با Shape برابر (2,) را می‌دهند. جمع listها [1,2,3,-1] است. جمع حاصل‌های ضرب عضو‌به‌عضو، Scalar برابر ۱ می‌شود.",
"ترکیب سه Vector چهارمؤلفه‌ای با سه ضریب، چند ویژگی دارد؟ کدام محور جمع می‌شود؟",
"جمع Token Embedding و Positional Embedding، ترکیب Valueها در Attention و Residual Connection از همین حساب استفاده خواهند کرد؛ نام‌های تازه را در درس‌های خودشان باز می‌کنیم.",
code="""u, v = [1., 2.], [3., -1.]

def same_length(a, b):
    if len(a) != len(b):
        raise ValueError("equal vector lengths required")

def add(a, b):
    same_length(a, b)
    return [x+y for x, y in zip(a, b)]

def scale(number, vector):
    return [number*x for x in vector]

def multiply_elements(a, b):
    same_length(a, b)
    return [x*y for x, y in zip(a, b)]

mixed = add(scale(0.25, u), scale(0.75, v))
assert add(u, v) == [4., 1.]
assert scale(2., u) == [2., 4.]
assert multiply_elements(u, v) == [3., -2.]
assert sum(multiply_elements(u, v)) == 1.
assert mixed == [2.5, -0.25]
assert add(scale(1., u), scale(0., v)) == u
assert u+v == [1., 2., 3., -1.]
print("weighted sum:", mixed)
print("ordinary list concatenation:", u+v)
""")

NEURON = Lesson(
"12b-neuron", 2, "از محاسبه تا شبکه", "یک Neuron را با دست آموزش دهیم",
"از دو ویژگی تا پیش‌بینی برویم؛ سپس Loss، Gradient و تغییر واقعی Parameterها را حساب کنیم.",
"""<p>تا اینجا جمع وزن‌دار، Loss و Gradient را جدا دیده‌ایم. حالا می‌خواهیم آن‌ها را در یک محاسبه دنبال کنیم: دو ویژگی وارد می‌شوند، یک پیش‌بینی می‌سازیم و از خطای آن می‌فهمیم وزن‌ها را چقدر تغییر دهیم. هنوز به PyTorch نیاز نداریم؛ عددها آن‌قدر کوچک‌اند که می‌شود یک گام کامل را با دست حساب کرد.</p>
<p><dfn>Neuron (یک واحد محاسباتی شبکه)</dfn> در این مثال ابتدا جمع وزن‌دار ورودی را می‌سازد؛ این نام به معنای شبیه‌سازی دقیق سلول زیستی نیست. <dfn>Weight (ضریب یک ویژگی ورودی)</dfn> اثر آن ویژگی را تنظیم می‌کند. اگر ورودی صفر باشد، حاصل ضرب‌ها هم صفر می‌شود. <dfn>Bias (مقدار جابه‌جایی مستقل از ورودی)</dfn> اجازه می‌دهد خروجیِ پیش از Activation به صفر محدود نماند. هر دو Parameter قابل یادگیری‌اند.</p>
<p>ورودی <code>x=[1,2]</code>، وزن <code>w=[0.5,-0.25]</code> و Bias برابر ۱ است. جمع وزن‌دار <code>z=0.5×1-0.25×2+1=1</code> می‌شود. z مقدار میانی است. <dfn>Activation function (تابع تبدیلِ پس از جمع وزن‌دار)</dfn> می‌تواند رابطه را غیرخطی کند. اینجا <dfn>ReLU (تابعی که مقدار منفی را صفر می‌کند)</dfn> را می‌گذاریم: <code>a=max(0,z)</code>؛ پس پیش‌بینی a=1 است.</p>
<div class="math">x → z=w·x+b → a=max(0,z)<br>target y=2 → L=(a−y)²=1</div>
<p>حرکت از ورودی به پیش‌بینی، <dfn>Forward pass (محاسبهٔ رو به جلوی مدل)</dfn> است. در Training، پیش‌بینی را با Target مقایسه می‌کنیم. Loss این مثال مربع فاصله است. هنوز وزن عوض نشده است: نخست حساسیت Loss به هر Parameter را حساب می‌کنیم.</p>
<p>از انتها شروع کنیم: <code>dL/da=2(a-y)=-2</code>. چون z مثبت است، ReLU در این نقطه همان z و Derivative آن ۱ است. پس <code>dL/dz=-2×1=-2</code>. تغییر w₀ به اندازهٔ کوچک δ، z را 1δ تغییر می‌دهد؛ برای w₁ این تغییر 2δ است. حساسیت نسبت به Bias نیز ۱ است. با Chain Rule، Gradient وزن‌ها <code>[-2,-4]</code> و Gradient مربوط به Bias برابر −۲ است.</p>
<div class="math">Learning Rate = 0.1<br>w_new = [0.5,−0.25] − 0.1[−2,−4] = [0.7,0.15]<br>b_new = 1 − 0.1(−2) = 1.2<br>a_new = 0.7×1 + 0.15×2 + 1.2 = 2.2<br>L_new = (2.2−2)² = 0.04</div>
<p>Loss از ۱ به ۰٫۰۴ رسید؛ پس این گام پیش‌بینی را بهتر کرد. از همین نتیجه نمی‌توانیم بگوییم هر Learning Rate مناسب است. دو کار جدا انجام دادیم: Backpropagation Gradient را محاسبه کرد و Gradient descent وزن‌ها را تغییر داد. اثر Target را هم جداگانه بررسی کنید: فقط Target را عوض کنید. پیش‌بینی فعلی همان می‌ماند، اما Loss و Gradient تغییر می‌کنند.</p>
<p>یک <dfn>Layer (مرحله‌ای از پردازش شبکه)</dfn> می‌تواند چند Neuron با ورودی مشترک و وزن‌های مستقل داشته باشد. Neuron دوم با <code>w=[1,0]</code> و Bias برابر −۲، z=−1 و خروجی صفر می‌دهد. پیش از آموزش، این Layer ورودی دو‌مؤلفه‌ای را به <code>[1,0]</code> می‌برد. وزن هر Neuron را یک سطر Matrix بگذاریم، جمع‌های وزن‌دار به Matrix multiplication تبدیل می‌شوند. خروجی یک Layer می‌تواند ورودی Layer بعد باشد: این همان ارتباط قطعه‌ها در Neural Network است.</p>
<p>در API ممکن است جمع وزن‌دار همراه Bias و Activation دو Layer جدا باشند؛ همهٔ Layerها Parameter ندارند. ReLU برای z منفی شیب صفر دارد و در صفر Derivative دوطرفه ندارد؛ مثال اصلی عمداً از صفر دور است. Gradient صفر ثابت نمی‌کند که پیش‌بینی درست است.</p>""",
"پیش‌بینی، Loss، Gradient و وزن تازه را پیش از اجرا حساب کنید. در آزمایشی جدا، همان ورودی و Weightهای اولیه را نگه دارید و Bias را −۱ بگذارید. z و شیب ReLU را دوباره حساب کنید؛ چرا این بار Loss غیرصفر است اما Gradient صفر می‌شود؟ آزمایش مثبتِ کد عمداً شرط z>0 دارد؛ آن شرط را با تغییر Bias نادیده نگیرید.",
"در حالت اصلی پیش‌بینی ۱، Loss برابر ۱، Gradientها [-2,-4] و −۲، وزن تازه [0.7,0.15] و Bias تازه ۱٫۲ است؛ پیش‌بینی تازه ۲٫۲ و Loss برابر ۰٫۰۴ می‌شود. با Bias اولیهٔ −۱، z=−1 و پیش‌بینی صفر است. Loss برابر ۴ است، ولی شیب ReLU در این نقطه صفر است و Gradientها صفر می‌شوند.",
"Input، Target و Parameter را نام ببرید. چرا Gradient صفر و Loss صفر دو گزارهٔ متفاوت‌اند؟",
"Projection و FFN همین جمع وزن‌دار و تبدیل ویژگی‌ها را دارند. GPT به‌جای خروجی Scalar، Logits واژگان و Cross Entropy دارد؛ زنجیرهٔ Forward → Loss → Backpropagation → Optimizer همان است.",
code="""import math

x, w = [1., 2.], [0.5, -0.25]
bias, target, rate = 1., 2., 0.1

def forward(inputs, weights, bias):
    if len(inputs) != len(weights):
        raise ValueError("input and weight sizes differ")
    z = sum(value*weight for value, weight in zip(inputs, weights)) + bias
    return z, max(0., z)

z, prediction = forward(x, w, bias)
loss = (prediction-target)**2
assert z > 0.  # ReLU derivative is 1 at this point.
dloss_dz = 2*(prediction-target)*1.
grad_w = [dloss_dz*value for value in x]
grad_bias = dloss_dz
new_w = [weight-rate*gradient for weight, gradient in zip(w, grad_w)]
new_bias = bias-rate*grad_bias
_, new_prediction = forward(x, new_w, new_bias)
new_loss = (new_prediction-target)**2
assert grad_w == [-2., -4.] and grad_bias == -2.
assert math.isclose(new_prediction, 2.2)
assert math.isclose(new_loss, 0.04)
assert new_loss < loss
print("before:", prediction, loss, "after:", new_prediction, new_loss)
layer = [forward(x, w, bias)[1], forward(x, [1., 0.], -2.)[1]]
assert layer == [1., 0.]
negative_z, negative_prediction = forward(x, w, -1.)
negative_gradient = 2*(negative_prediction-target)*0.
assert negative_z < 0. and negative_gradient == 0.
assert (negative_prediction-target)**2 == 4.
""",
review="پیش از شروع PyTorch، این یک گام را بدون کد بازسازی کنید. ابزار باید محاسبهٔ آشنا را خودکار کند، نه اینکه معنای آن را پنهان کند.")
