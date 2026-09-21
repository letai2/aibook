"""Self-contained numerical examples for foundational and ambiguous concepts."""
FOCUSED_EXAMPLES = {
    'vector-addition': {'example': 'برای u=[1,2] و v=[3,-1]، جمع مؤلفه‌های متناظر [4,1] است. خروجی همچنان دو مؤلفه دارد؛ جمع list در Python به‌جای این عمل، [1,2,3,-1] می‌سازد.'},
    'scalar-multiplication': {'example': 'ضرب Scalar برابر ۰٫۵ در Vector برابر [2,-4]، نتیجهٔ [1,-2] می‌دهد. این عمل تعداد مؤلفه‌ها را تغییر نمی‌دهد؛ شرط هم‌اندازه‌بودن دو Vector مطرح نیست.'},
    'element-wise-multiplication': {'example': 'برای u=[1,2] و v=[3,-1]، ضرب عضو‌به‌عضو [3,-2] و Dot product حاصل جمع آن، برابر ۱ است. نوع خروجی این دو عملیات متفاوت است.'},
    'weighted-sum': {
        'technical': 'برای u=[1,2] و v=[3,-1]، ترکیب 0.25u+0.75v هر مؤلفه را مستقل حساب می‌کند. مؤلفهٔ نخست 0.25×1+0.75×3=2.5 و دومی 0.25×2+0.75×(-1)=-0.25 است.',
        'example': 'خروجی [2.5,-0.25] یک Vector دو‌مؤلفه‌ای است. با ضریب‌های ۱ و صفر خود u را می‌گیریم. اگر ضریب‌ها نامنفی و مجموعشان یک باشند، این ترکیب میانگین وزن‌دار نیز هست.'},
    'scalar': {'example': 'Loss برابر ۲٫۳ یک Scalar است. یک Tensor با Shape برابر () صفر محور و یک مقدار دارد؛ این با Vector تک‌عضویِ Shape برابر (1,) متفاوت است.'},
    'vector': {'example': 'Vector برابر [1,2,3] سه مؤلفه و Shape برابر (3,) دارد. خانهٔ شمارهٔ ۱ مقدار ۲ است؛ شمار محور یک است، نه سه.', 'lessons': '05-shape 05a-vector-operations 06-dot'},
    'matrix': {'example': 'Matrix برابر [[1,2,3],[4,5,6]] دو سطر و سه ستون و Shape برابر (2,3) دارد. عنصر [1,2] مقدار ۶ است.'},
    'probability': {'example': 'اگر در یک مثال گسسته دو ادامهٔ ممکن داشته باشیم و سهمشان ۰٫۷۵ و ۰٫۲۵ باشد، هر دو نامنفی‌اند و جمعشان یک است. انتخاب با احتمال ۰٫۷۵ تضمین نمی‌کند که دقیقاً سه بار از چهار بار رخ دهد.'},
    'probability-distribution': {'example': 'برای Vocabulary فرضیِ [ا،ب،فاصله]، احتمال‌های [0.2,0.3,0.5] یک توزیع معتبرند؛ هر احتمال به Token مشخص خودش مربوط است و مجموع یک است.'},
    'recurrent-neural-network': {
        'technical': '<p>در هر گام یک ورودی x_t و حافظهٔ قبلی h_(t−1) داریم. وزن‌های W_x و W_h در همهٔ گام‌ها مشترک‌اند. برای x با C ویژگی و h با M ویژگی، Shape این وزن‌ها (M,C) و (M,M) است؛ Bias نیز M مؤلفه دارد.</p><div class="math">h_t = tanh(W_x x_t + W_h h_(t−1) + b)</div>',
        'example': 'در مثال Scalar، W_x=1، W_h=0.5، b=0، x_t=1 و حافظهٔ قبلی صفر می‌دهد h_t=tanh(1)≈0.7616. با همان ورودی و حافظهٔ قبلی ۱، مقدار تازه tanh(1.5)≈0.9051 می‌شود؛ گذشته روی نتیجه اثر دارد.'},
    'variance': {
        'technical': 'برای N عدد، میانگین را کم می‌کنیم، فاصله‌ها را مربع می‌کنیم و بر N تقسیم می‌کنیم. اگر مؤلفه‌های Q/K مستقل، با میانگین صفر و Variance یک باشند، Variance جمع D حاصل‌ضرب تقریباً D است. این فرض‌ها توجیه مقیاس آغازین Attention هستند، نه تضمین آمار همهٔ مدل‌های آموزش‌دیده.',
        'example': 'برای [1,2,3]، میانگین ۲ و فاصله‌ها [-1,0,1] است. Variance=(1+0+1)/3=2/3. برای [2,4,6] مقدار 8/3 است: دوبرابرکردن عددها، Variance را چهار برابر می‌کند.'},
    'standard-deviation': {
        'technical': 'ریشهٔ Variance است و واحدش همان واحد عددهای اصلی است. ضرب همهٔ عددها در ضریب a، Standard deviation را در |a| ضرب می‌کند. در Attention تقسیم بر √D جلوی رشد مقیاس امتیاز با تعداد ویژگی‌ها را در فرض‌های آغازین درس می‌گیرد.',
        'example': 'برای [1,2,3]، Variance برابر ۲/۳ و Standard deviation حدود ۰٫۸۱۶ است. همهٔ عددها اگر برابر باشند، هر دو صفر می‌شوند.'},
    'gelu': {'example': 'در تعریف دقیق GELU(x)=xΦ(x)، احتمال Φ(1)≈0.8413 است؛ پس GELU(1)≈0.8413. برای −۱، مقدار تقریباً −۰٫۱۵۸۷ و برای صفر، صفر است. بنابراین همهٔ خروجی‌های منفی حذف نمی‌شوند.'},
    'head': {
        'meaning': 'Head در این مدخل یک مسیر Attention با Projectionها و جدول وزن خودش است؛ با Language-model head خروجی فرق دارد.',
        'technical': 'با C=12 و H=3، هر Attention Head چهار ویژگی دارد. جدول وزن هر Head همهٔ جفت‌های مجاز موقعیت را می‌سنجد؛ Head یک تکهٔ جدا از متن نیست.',
        'example': 'برای B=2,T=5,C=12,H=3، شکل Q برابر (2,3,5,4) و شکل وزن Attention برابر (2,3,5,5) است. سه Head سه روش آموخته برای ترکیب اطلاعات دارند، نه سه Vocabulary جدا.',
        'related': 'multi-head-attention query key value language-model-head'},
    'language-model-head': {
        'technical': 'در پروژه یک Linear بدون Bias، آخرین محور C را به V امتیاز خام تبدیل می‌کند؛ شکل (B,T,C) به (B,T,V) می‌رود. این جزء جدول وزن Attention نیست.',
        'example': 'برای hidden=[2,1] و Weight برابر [[1,0],[0,1],[1,-1]]، Logits برابر [2,1,1] است. سه سطر Weight، سه Token ممکن را امتیاز می‌دهند. برای احتمال‌گیری Softmax و برای Training مستقیماً Cross-Entropy را روی Logits اجرا می‌کنیم.'},
    'buffer': {'example': 'Causal Mask بولیِ ۴×۴ در state_dict می‌تواند ۱۶ مقدار داشته باشد، اما صفر Parameter قابل یادگیری به شمار مدل اضافه می‌کند؛ Optimizer آن را به‌روزرسانی نمی‌کند.'},
    'persistent-buffer': {'example': 'اگر Mask را با register_buffer ثبت کنیم و persistent پیش‌فرض True باشد، انتقال دستگاه و state_dict آن را همراه می‌برند. ثبت آن به معنی داشتن Gradient یا قرارگرفتن در model.parameters نیست.'},
    'neural-network': {'lessons': '01-learning 12b-neuron 19-network 40-block'},
    'layer': {'lessons': '12b-neuron 18-module 37-ffn'},
}
