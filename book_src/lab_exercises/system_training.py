"""Learner tasks for actual SFT, measured evaluation and the final system.

Reference solutions stay in the author's separate answer/verification path.
"""

EXERCISES = {
    "65a-sft-lab": {
        "title": "از وزن پایه تا پاسخ هدف، با دو مدل مستقل",
        "goal": "حلقهٔ SFT را روی همان MiniGPT بنویسید و اثرش را بدون دست‌کاری Baseline بسنجید.",
        "prerequisite": "Loss mask درس قبل، `deepcopy`، Optimizer و تفاوت Training با عبارت Held-out.",
        "predict": "اگر مدل پایه و نسخهٔ تنظیم‌شده یک شیء باشند، بعد از آموزش چه چیزی از مقایسهٔ قبل و بعد از دست می‌رود؟",
        "setup": """from copy import deepcopy
import torch
from mini_gpt.instruction import (run_tiny_experiment, TRAIN_EXAMPLES, HELDOUT_EXAMPLES,
    encode_example, train_response_step, evaluate_instructions, InstructionExample)

torch.set_num_threads(1)
experiment = run_tiny_experiment(pretrain_steps=20, sft_steps=0)
base, tokenizer = experiment.base, experiment.tokenizer
print('base measurements:', experiment.reports['base_train'])
print('The targets are one character: ب=yes, ن=no. This is not a general assistant.')
""",
        "task": "تابع `fine_tune_copy(base, tokenizer, examples, steps, rate)` یک `deepcopy` از مدل بگیرد و با AdamW، نرخ `rate` و `weight_decay=0.0` آموزش دهد. نمونه‌ها غیرخالی‌اند؛ در گام `s` نمونهٔ `s % len(examples)` را با `encode_example` و `train_response_step` مصرف کنید. مدل تازه را برگردانید؛ حتی با `steps=0` باید مستقل باشد. دادهٔ Held-out وارد تابع نشود.",
        "starter": """def fine_tune_copy(base, tokenizer, examples, steps, rate):
    # TODO: همان نقطهٔ شروع، اما وزن و Optimizer مستقل
    return None
""",
        "check": """def test_exercise():
    before = {name:value.clone() for name,value in base.state_dict().items()}
    result = fine_tune_copy(base, tokenizer, TRAIN_EXAMPLES, 40, 0.003)
    if result is None:
        return False
    assert result is not base
    assert all(torch.equal(before[n],v) for n,v in base.state_dict().items())
    assert not torch.equal(result.language_model_head.weight,base.language_model_head.weight)
    initial = evaluate_instructions(base,tokenizer,TRAIN_EXAMPLES)['response_loss']
    measured = evaluate_instructions(result,tokenizer,TRAIN_EXAMPLES)
    assert measured['response_loss'] < initial
    zero = fine_tune_copy(base,tokenizer,TRAIN_EXAMPLES,0,0.003)
    assert zero is not base and zero.token_embedding.weight is not base.token_embedding.weight
    assert all(torch.equal(zero.state_dict()[n],v) for n,v in base.state_dict().items())
    # The supplied data must be used: compare two opposite labels for one prompt.
    a = (InstructionExample(TRAIN_EXAMPLES[0].instruction,'ب'),)
    b = (InstructionExample(TRAIN_EXAMPLES[0].instruction,'ن'),)
    first = fine_tune_copy(base,tokenizer,a,3,0.01)
    second = fine_tune_copy(base,tokenizer,b,3,0.01)
    assert not torch.equal(first.language_model_head.weight,second.language_model_head.weight)
    print('actual held-out wording:',evaluate_instructions(result,tokenizer,HELDOUT_EXAMPLES))
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: fine_tune_copy')
""",
        "solution": """def fine_tune_copy(base, tokenizer, examples, steps, rate):
    candidate = deepcopy(base)
    optimizer = torch.optim.AdamW(candidate.parameters(),lr=rate,weight_decay=0.0)
    encoded = [encode_example(tokenizer,e,candidate.config.context_length) for e in examples]
    for step in range(steps):
        train_response_step(candidate,optimizer,encoded[step % len(encoded)])
    return candidate
""",
        "vary": "فقط تعداد گام‌های SFT را عوض کنید. هر اجرا از همان Seed و همان تعداد گام Pretraining آغاز می‌شود. کیفیت Held-out را اندازه بگیرید؛ بهترشدن قطعی را شرط اجرا نگذارید.",
        "vary_code": """for steps in (20,80):
    trial = run_tiny_experiment(pretrain_steps=20,sft_steps=steps,seed=41)
    print('SFT steps:',steps,
          'train:',trial.reports['tuned_train']['response_loss'],
          'held-out:',trial.reports['tuned_heldout']['exact_match'])
""",
        "debug": "نسخهٔ خراب همان پرسش‌های آموزش را با نام Held-out گزارش می‌کند. `prompts_disjoint(training, heldout)` باید برای دو فهرست `InstructionExample` فقط وقتی True بدهد که هیچ متن دستور مشترکی ندارند. تغییر پاسخِ یک پرسش تکراری آن را مستقل نمی‌کند. این بررسی فقط تکرار دقیق متن را می‌بیند، نه همهٔ شکل‌های نشت معنایی.",
        "bug_code": """wrong_heldout = list(TRAIN_EXAMPLES)
print('claimed held-out question:',wrong_heldout[0].instruction)
print('already a training question:',TRAIN_EXAMPLES[0].instruction)
""",
        "fix": """def prompts_disjoint(training, heldout):
    # TODO: مقایسهٔ متن دستور، نه نام متغیر یا برچسب پاسخ
    return None
""",
        "fix_check": """def test_repair():
    result = prompts_disjoint(TRAIN_EXAMPLES,HELDOUT_EXAMPLES)
    if result is None:
        return False
    assert result is True
    assert prompts_disjoint(TRAIN_EXAMPLES,TRAIN_EXAMPLES) is False
    altered = [InstructionExample(TRAIN_EXAMPLES[0].instruction,'ن')]
    assert prompts_disjoint(TRAIN_EXAMPLES,altered) is False
    assert prompts_disjoint([],HELDOUT_EXAMPLES) is True
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: prompts_disjoint')
""",
        "fix_solution": """def prompts_disjoint(training, heldout):
    trained = {example.instruction for example in training}
    return trained.isdisjoint(example.instruction for example in heldout)
""",
        "connection": "Pretraining و SFT هر دو روی MiniGPT واقعی اجرا شده‌اند. `instruction.py` Loss پاسخ را بیرون `forward` می‌سازد؛ معماری تغییر نکرده و فایل قبلی بارگذاری یا بازنویسی نشده است. فقط چهار پاسخ کوتاه هدف‌اند.",
        "takeaway": "کدام شاهد نشان داد وزن‌ها واقعاً تغییر کردند، کدام نتیجه فقط حفظ نمونه‌ها را می‌سنجد و برای ادعای تعمیم چه داده‌ای هنوز لازم است؟",
    },
    "81-system-eval": {
        "title": "پاسخ درست به‌تنهایی کافی نیست",
        "goal": "نرخ موفقیت مشترک و میانگین معیارهای قابل اعمال را جدا حساب کنید؛ Regressionها را با نام نگه دارید.",
        "prerequisite": "خروجی `AssistantResult`، تفاوت Exact match با پشتیبانی معنایی، و سناریوی ثابت در برابر توان مدل.",
        "predict": "اگر یک مورد پاسخ درست ولی ابزار غلط داشته باشد، آیا `answer_exact` و `passed` باید یکسان باشند؟",
        "setup": """from copy import deepcopy
from dataclasses import replace
from mini_gpt.evaluation_system import (course_fixture_suite,evaluate_cases,score_result)

cases,run_case = course_fixture_suite()
report = evaluate_cases(cases,run_case)
rows = report['rows']
for row in rows:
    print(row)
print('These are ScriptedFixture integration checks, not MiniGPT accuracy.')
""",
        "task": "`aggregate_checks(rows)` برای سطرهای غیرخالی و یکتای گزارش، dict با `pass_rate` و `mean_recall` بدهد. اولی میانگین boolهای `passed` است. دومی فقط میانگین `retrieval_recall`هایی است که `None` نیستند؛ اگر هیچ مورد قابل اعمالی نداریم، `None` بدهید. مقدار صفر یک شکست واقعی است و نباید حذف شود.",
        "starter": """def aggregate_checks(rows):
    # TODO: None با صفر فرق دارد
    return None
""",
        "check": """def test_exercise():
    sample = [{'passed':True,'retrieval_recall':1.0},
              {'passed':False,'retrieval_recall':0.0},
              {'passed':True,'retrieval_recall':None}]
    result = aggregate_checks(sample)
    if result is None:
        return False
    assert result == {'pass_rate':2/3,'mean_recall':0.5}
    assert aggregate_checks([{'passed':False,'retrieval_recall':None}]) == {'pass_rate':0.0,'mean_recall':None}
    assert aggregate_checks(rows) == {'pass_rate':report['pass_rate'],'mean_recall':report['mean_retrieval_recall']}
    partial = [{'passed':False,'retrieval_recall':0.25},{'passed':False,'retrieval_recall':0.75}]
    assert aggregate_checks(partial) == {'pass_rate':0.0,'mean_recall':0.5}
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: aggregate_checks')
""",
        "solution": """def aggregate_checks(rows):
    applicable = [row['retrieval_recall'] for row in rows if row['retrieval_recall'] is not None]
    return {'pass_rate':sum(row['passed'] for row in rows)/len(rows),
            'mean_recall':sum(applicable)/len(applicable) if applicable else None}
""",
        "vary": "فقط ارجاع پاسخ مستند را حذف کنید؛ خود پاسخ، شاهد واردشده و نتیجهٔ وارسی قبلی ثابت‌اند. این دست‌کاریِ گزارش برای جداکردن معیارهاست، نه یک پاسخ تازهٔ مدل.",
        "vary_code": """case = cases[0]
original = run_case(case)
without_citation = replace(original,citations=())
print('original:',score_result(case,original))
print('only citation removed:',score_result(case,without_citation))
""",
        "debug": "مقایسهٔ موقعیت سطرها، با تغییر ترتیب گزارش، مورد غلط را Regression معرفی می‌کند. `regressed_cases(before, after)` نام مواردی را به‌ترتیب الفبایی بدهد که قبلاً `passed=True` و اکنون False دارند. دو فهرست همین شناسه‌های یکتا را دارند ولی ترتیبشان ممکن است فرق کند.",
        "bug_code": """before = [{'id':'evidence','passed':True},{'id':'tool','passed':False}]
after = [{'id':'tool','passed':False},{'id':'evidence','passed':True}]
wrong = [left['id'] for left,right in zip(before,after) if left['passed'] and not right['passed']]
print('false regression from row positions:',wrong)
""",
        "fix": """def regressed_cases(before, after):
    # TODO: هویت مورد، نه شمارهٔ سطر
    return None
""",
        "fix_check": """def test_repair():
    result = regressed_cases(before,after)
    if result is None:
        return False
    assert result == []
    earlier = [{'id':'b','passed':True},{'id':'a','passed':True},{'id':'c','passed':False}]
    later = [{'id':'c','passed':False},{'id':'a','passed':False},{'id':'b','passed':False}]
    assert regressed_cases(earlier,later) == ['a','b']
    assert regressed_cases([{'id':'x','passed':False}],[{'id':'x','passed':True}]) == []
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: regressed_cases')
""",
        "fix_solution": """def regressed_cases(before, after):
    latest = {row['id']:row['passed'] for row in after}
    return sorted(row['id'] for row in before if row['passed'] and not latest[row['id']])
""",
        "connection": "سطرها از اجرای واقعی کنترل‌گر، Retrieval و ابزار می‌آیند؛ فقط پیشنهادهای Backend از قبل نوشته شده‌اند. Exact match و شناسهٔ ارجاع، معیارهای قراردادی این سناریوها هستند و داور عمومی معنای پاسخ نیستند.",
        "takeaway": "چرا می‌توان با بهترشدن میانگین کل، یک Regression مهم داشت؟ کدام ادعا هنوز نیازمند خروجی واقعی مدل و بررسی انسانی منبع است؟",
    },
    "82-performance": {
        "title": "زمان ثابت نداریم؛ قرارداد اندازه‌گیری داریم",
        "goal": "تأخیر هر اجرا و نرخ کل تولید را با واحد و مخرج درست بسنجید؛ نمایش کم‌بیتی را با سرعت بیشتر یکی ندانید.",
        "prerequisite": "حلقهٔ واقعی `generate`، تعداد Token تازه، Batch و اندازهٔ Tensor.",
        "predict": "اگر دو دنباله هرکدام چهار Token تازه بسازند، صورت کسر Throughput چهار است یا هشت؟ Prompt هم در آن حساب می‌شود؟",
        "setup": """import math
from statistics import median
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
from mini_gpt.performance import benchmark_generation,quantize_symmetric

torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(12,32,16,2,1,0.0))
prompt = torch.ones(2,6,dtype=torch.long)
measured = benchmark_generation(model,prompt,new_tokens=4,repeats=3,warmup=1)
print(measured)
print('Actual CPU timing of an untrained model; not an answer-quality experiment.')
""",
        "task": "`workload_summary(seconds, batch, new_tokens)` برای زمان‌های مثبتِ چند اجرای هم‌اندازه، dict با `median_seconds`، `total_generated` و `pooled_tokens_per_second` بدهد. هر اجرا `batch*new_tokens` Token تازه می‌سازد؛ نرخ تجمیعی، کل این Tokenها تقسیم بر مجموع زمان‌هاست. آن را با میانهٔ نرخ اجراها یکی نگیرید.",
        "starter": """def workload_summary(seconds, batch, new_tokens):
    # TODO: واحد و تعداد اجرای تکراری را نگه دارید
    return None
""",
        "check": """def test_exercise():
    result = workload_summary([0.1,0.3],2,4)
    if result is None:
        return False
    assert math.isclose(result['median_seconds'],0.2)
    assert result['total_generated'] == 16
    assert math.isclose(result['pooled_tokens_per_second'],40.0)
    other = workload_summary([1.0,2.0,6.0],1,3)
    assert other == {'median_seconds':2.0,'total_generated':9,'pooled_tokens_per_second':1.0}
    assert workload_summary([2.0],3,2) == {'median_seconds':2.0,'total_generated':6,'pooled_tokens_per_second':3.0}
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: workload_summary')
""",
        "solution": """def workload_summary(seconds, batch, new_tokens):
    total = len(seconds)*batch*new_tokens
    return {'median_seconds':median(seconds),'total_generated':total,
            'pooled_tokens_per_second':total/sum(seconds)}
""",
        "vary": "فقط تعداد بیت قراردادی Quantization را از ۴ به ۸ ببرید؛ همان وزن‌ها را بازسازی کنید. هم خطا و هم اندازهٔ ذخیرهٔ واقعی را چاپ کنید؛ خروجی هر دو در این مثال `int8` است، نه بسته‌بندی واقعی چهاربیتی.",
        "vary_code": """values = model.language_model_head.weight.detach()
for bits in (4,8):
    integer,scale,reconstructed = quantize_symmetric(values,bits)
    print('bits:',bits,'max error:',(reconstructed-values).abs().max().item(),
          'actual integer bytes:',integer.numel()*integer.element_size(),
          'scale stored separately:',scale)
""",
        "debug": "نسخهٔ خراب طول کل خروجی را تعداد Token تولیدشده می‌نامد و Prompt را دوباره می‌شمارد. `generated_count(prompt_lengths, output_lengths)` مجموع اختلاف طول‌های متناظر را بدهد. دو فهرست هم‌اندازه‌اند و خروجی هر دنباله کوتاه‌تر از Prompt نیست؛ صفر Token تازه هم معتبر است.",
        "bug_code": """prompt_lengths = [6,6]
output_lengths = [10,10]
print('wrong generated-token count:',sum(output_lengths))
print('input lengths:',prompt_lengths,'output lengths:',output_lengths)
""",
        "fix": """def generated_count(prompt_lengths, output_lengths):
    # TODO: فقط ادامهٔ تولیدشده
    return None
""",
        "fix_check": """def test_repair():
    result = generated_count(prompt_lengths,output_lengths)
    if result is None:
        return False
    assert result == 8
    assert generated_count([3,8],[5,9]) == 3
    assert generated_count([4],[4]) == 0
    assert generated_count([],[]) == 0
    assert generated_count([6,6],[len(row) for row in measured['output_ids']]) == 8
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: generated_count')
""",
        "fix_solution": """def generated_count(prompt_lengths, output_lengths):
    return sum(output-prompt for prompt,output in zip(prompt_lengths,output_lengths))
""",
        "connection": "زمان از `generate` واقعی و بدون KV cache آمده است. آموزش‌ندیده‌بودن مدل، شمارش محاسبه را ساختگی نمی‌کند، اما اجازهٔ نتیجه‌گیری دربارهٔ کیفیت نمی‌دهد. هیچ آزمونی زمان ثابت یا سریع‌ترشدن قطعی را مطالبه نمی‌کند.",
        "takeaway": "کدام بخش هزینه در این زمان نیست؟ چرا مدل با پارامترهای کمتر یا عددهای کم‌بیت‌تر الزاماً در هر محیط تأخیر کمتری ندارد؟",
    },
    "83-deployment": {
        "title": "بستهٔ یکسان، ورودی محدود، اجرای محلی",
        "goal": "قرارداد بودجهٔ درخواست و سازگاری Vocabulary را پیش از تولید بررسی کنید.",
        "prerequisite": "Context Window، ذخیرهٔ وزن و فرق Checkpoint آموزش با بستهٔ Inference.",
        "predict": "اگر دو Vocabulary هم‌اندازه‌اند ولی ترتیبشان فرق دارد، آیا برابری Shape جدول‌ها کافی است؟",
        "setup": """import tempfile
from pathlib import Path
import torch
from mini_gpt.instruction import (run_tiny_experiment,format_prompt,TRAIN_EXAMPLES,
    save_instruction_bundle,load_instruction_bundle)
from mini_gpt.performance import InferenceSession,RequestLimits

torch.set_num_threads(1)
experiment = run_tiny_experiment(pretrain_steps=2,sft_steps=4)
prompt = format_prompt(TRAIN_EXAMPLES[0].instruction)
before = InferenceSession(experiment.tuned,experiment.tokenizer).request(prompt)
with tempfile.TemporaryDirectory(prefix='aibook-deploy-') as directory:
    path = Path(directory)/'مدل تمرین.pt'
    save_instruction_bundle(path,experiment.tuned,experiment.tokenizer)
    restored,restored_tokenizer = load_instruction_bundle(path)
after = InferenceSession(restored,restored_tokenizer,RequestLimits(128,8)).request(prompt)
assert before == after
print('same local request before/after loading:',after)
print('The temporary bundle is already removed. This is not a public service.')
""",
        "task": "`request_fits(prompt_tokens, new_tokens, context_limit, output_limit)` یک bool بدهد: Prompt حداقل یک Token داشته باشد، خروجی بین ۱ و `output_limit` باشد و مجموعشان از `context_limit` بیشتر نشود. ورودی‌های تمرین عدد صحیح‌اند؛ کاراکتر و Token را با هم مخلوط نکنید.",
        "starter": """def request_fits(prompt_tokens, new_tokens, context_limit, output_limit):
    # TODO: خروجی از قبل جا رزرو می‌کند
    return None
""",
        "check": """def test_exercise():
    result = request_fits(56,8,64,8)
    if result is None:
        return False
    assert result is True
    assert request_fits(57,8,64,8) is False
    assert request_fits(1,9,64,8) is False
    assert request_fits(0,1,64,8) is False
    assert request_fits(4,0,64,8) is False
    assert request_fits(4,-1,64,8) is False
    assert request_fits(1,1,2,1) is True
    length = len(restored_tokenizer.encode(prompt))
    assert request_fits(length,1,restored.config.context_length,8) is True
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: request_fits')
""",
        "solution": """def request_fits(prompt_tokens, new_tokens, context_limit, output_limit):
    return prompt_tokens >= 1 and 1 <= new_tokens <= output_limit and prompt_tokens+new_tokens <= context_limit
""",
        "vary": "فقط بودجهٔ خروجی در همان درخواست را بیشتر کنید. برای ردشدن درخواست، فقط `ValueError` مورد انتظار را بگیرید؛ هر خروجی متناظر با همین مدل کم‌آموزش است و ادعای کیفیت ندارد.",
        "vary_code": """session = InferenceSession(restored,restored_tokenizer,RequestLimits(128,8))
for count in (1,4,9):
    try:
        print(count,session.request(prompt,new_tokens=count))
    except ValueError as error:
        print(count,'expected request rejection:',error)
""",
        "debug": "نسخهٔ خراب Vocabularyها را به `set` تبدیل می‌کند؛ با این کار جابه‌جایی IDها پنهان می‌شود. `compatible_vocabulary(left_tokens,right_tokens)` فقط برای فهرست‌های هم‌ترتیب True بدهد. شباهت مجموعهٔ کاراکترها کافی نیست.",
        "bug_code": """left_tokens = ['<|unk|>','ا','ب']
right_tokens = ['<|unk|>','ب','ا']
print('wrong unordered compatibility:',set(left_tokens)==set(right_tokens))
print('ID 1 now means:',left_tokens[1],right_tokens[1])
""",
        "fix": """def compatible_vocabulary(left_tokens, right_tokens):
    # TODO: ترتیب، بخشی از قرارداد وزن است
    return None
""",
        "fix_check": """def test_repair():
    result = compatible_vocabulary(left_tokens,right_tokens)
    if result is None:
        return False
    assert result is False
    assert compatible_vocabulary(left_tokens,list(left_tokens)) is True
    assert compatible_vocabulary(left_tokens,left_tokens+['پ']) is False
    assert compatible_vocabulary(experiment.tokenizer.id_to_token,restored_tokenizer.id_to_token) is True
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: compatible_vocabulary')
""",
        "fix_solution": """def compatible_vocabulary(left_tokens, right_tokens):
    return left_tokens == right_tokens
""",
        "connection": "ذخیره، بارگذاری و تولید واقعاً اجرا شدند. بستهٔ این درس Optimizer و RNG لازم برای Resume را ندارد. لایهٔ درخواست فقط محلی، همگام و محدود به تعداد Token است؛ Timeout، احراز هویت یا ایمنی سرویس عمومی را پیاده نمی‌کند.",
        "takeaway": "آزمون Round-trip چه چیزی را ثابت کرد و برای اینکه یک سرویس عمومی قابل اتکا باشد چه قراردادهای دیگری هنوز لازم‌اند؟",
    },
    "84-capstone": {
        "title": "یک تحویل با دو نوع شاهد",
        "goal": "مسیر مدل واقعی و مسیر Fixture را کنار هم اجرا کنید و نتیجهٔ ابزار اجراشده را از پیشنهاد متن جدا نگه دارید.",
        "prerequisite": "کنترل‌گر محدود، ارزیابی جزءبه‌جزء، حافظهٔ بیرونی صریح و مرز توان Mini-GPT کوچک.",
        "predict": "اگر مدل فقط متنی شبیه درخواست ابزار تولید کند ولی کنترل‌گر آن را اجرا نکند، آیا عدد آن متن نتیجهٔ ابزار است؟",
        "setup": """import torch
from mini_gpt.assistant import ScriptedFixture,MiniGPTBackend,run_assistant
from mini_gpt.instruction import run_system_experiment
from mini_gpt.evaluation_system import course_fixture_suite,evaluate_cases
from mini_gpt.memory import MemoryStore,MemoryRecord
from mini_gpt.retrieval import COURSE_DOCUMENTS,chunk_document

torch.set_num_threads(1)
experiment = run_system_experiment()
chunks = [c for d in COURSE_DOCUMENTS for c in chunk_document(d,chunk_words=24,overlap_words=4)]
backend = MiniGPTBackend(experiment.tuned,experiment.tokenizer,max_new_tokens=8)
actual = run_assistant('Checkpoint',backend,chunks=chunks)
print('Training/capacity limits:',experiment.metadata)
print('ACTUAL tiny model path:',actual.status,actual.backend,'generate calls:',backend.calls)
print('Actual proposals:',[e['text'] for e in actual.events if e['state']=='propose'])
cases,runner = course_fixture_suite()
print('SCRIPTED integration fixtures:',evaluate_cases(cases,runner))
memory = MemoryStore()
memory.upsert(MemoryRecord('style','توضیح کوتاه','explicit exercise preference; no personal data'))
print('Explicit memory records:',memory.records())
""",
        "task": "`run_checked_case(question, proposals, expected_answer, chunks, memory, memory_keys)` یک `ScriptedFixture` تازه از `proposals` بسازد و آن را به `run_assistant` بدهد. سندها، Store و فقط کلیدهای مجازِ `memory_keys` را عبور دهید. تابع وارسی فقط برابری دقیق متن با `expected_answer` را بررسی کند؛ خروجی همان `AssistantResult` باشد. این داور محدود، صحت عمومی یا پشتیبانی معنایی همهٔ ادعاها را نمی‌سنجد.",
        "starter": """def run_checked_case(question, proposals, expected_answer, chunks, memory, memory_keys):
    # TODO: اتصال اجزا و وارسی صریح، بدون جایگزینی پاسخ
    return None
""",
        "check": """def test_exercise():
    proposals = [
        {'action':'tool','name':'add','arguments':{'a':2,'b':3}},
        {'action':'finish','answer':'5','citations':[]}]
    result = run_checked_case('دو به علاوه سه؟',proposals,'5',[],memory,['style'])
    if result is None:
        return False
    assert result.status == 'finished' and result.verified
    assert result.answer == '5' and result.tool_results[0]['value'] == 5
    assert 'memory:style' in result.context_ids
    assert 'ScriptedFixture' in result.backend
    wrong = [{'action':'finish','answer':'6','citations':[]}]
    rejected = run_checked_case('دو به علاوه سه؟',wrong,'5',[],memory,[])
    assert rejected.status == 'verification_failed' and rejected.answer == '6'
    assert 'memory:style' not in rejected.context_ids
    invalid = [{'action':'tool','name':'shell','arguments':{'a':2,'b':3}}]
    assert run_checked_case('اجرا',invalid,'5',[],memory,[]).status == 'invalid_action'
    again = run_checked_case('دو به علاوه سه؟',proposals,'5',[],memory,[])
    assert again.status == 'finished'  # Fresh fixture cursor on every call.
    assert len(memory.records()) == 1
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: run_checked_case')
""",
        "solution": """def run_checked_case(question, proposals, expected_answer, chunks, memory, memory_keys):
    backend = ScriptedFixture(proposals)
    def verify(answer, citations, tool_results):
        return answer == expected_answer
    return run_assistant(question,backend,chunks=chunks,memory=memory,memory_keys=memory_keys,verify_answer=verify)
""",
        "vary": "دو Store مستقل از رکوردهای اولیه بسازید و فقط در یکی رکورد را حذف کنید؛ حافظهٔ اصلی تمرین دست‌نخورده بماند. در دو اجرای Fixture با پاسخ ثابت، شناسه‌های Context را مقایسه کنید؛ ثابت‌ماندن پاسخ ازپیش‌نوشته‌شده شاهد بی‌اثر بودن حافظه بر یک مدل آموخته نیست.",
        "vary_code": """fixed = [{'action':'finish','answer':'نمونه','citations':[]}]
kept_store = MemoryStore(memory.records())
removed_store = MemoryStore(memory.records())
with_memory = run_assistant('Checkpoint',ScriptedFixture(fixed),chunks=chunks,memory=kept_store,memory_keys=['style'])
removed_store.forget('style')
without_memory = run_assistant('Checkpoint',ScriptedFixture(fixed),chunks=chunks,memory=removed_store,memory_keys=['style'])
print('before:',with_memory.context_ids)
print('after:',without_memory.context_ids)
assert 'memory:style' in with_memory.context_ids and 'memory:style' not in without_memory.context_ids
assert len(memory.records()) == 1
""",
        "debug": "پیشنهادِ مدل و خروجیِ ابزار دو رویداد متفاوت‌اند. `executed_values(events)` فقط مقدارهای `event['result']['value']` رویدادهایی با `state='execute'` را به‌ترتیب برگرداند. متن proposal حتی اگر عدد داشته باشد، شاهد اجرا نیست.",
        "bug_code": """events = [
    {'state':'propose','text':'The tool returned 999'},
    {'state':'execute','result':{'name':'add','value':5}},
    {'state':'verify','passed':True}]
print('unexecuted claim:',events[0]['text'])
print('actual execution record:',events[1])
""",
        "fix": """def executed_values(events):
    # TODO: فقط رویداد اجرای واقعی
    return None
""",
        "fix_check": """def test_repair():
    result = executed_values(events)
    if result is None:
        return False
    assert result == [5]
    assert executed_values([{'state':'propose','text':'5'}]) == []
    assert executed_values([]) == []
    multiple = [{'state':'execute','result':{'value':0}},
                {'state':'execute','result':{'value':-2}}]
    assert executed_values(multiple) == [0,-2]
    real = runner(cases[1])
    assert executed_values(real.events) == [5]
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: executed_values')
""",
        "fix_solution": """def executed_values(events):
    return [event['result']['value'] for event in events if event['state']=='execute']
""",
        "connection": "سامانه از ماژول‌های واقعی پروژه ساخته شد. در آزمایش مستقلِ Context بزرگ، تولید MiniGPT واقعاً فراخوانی می‌شود ولی آموزش کوتاه، قالب JSON یا موقعیت‌های دور را آموزش نداده است؛ خروجی نامعتبر را پنهان نمی‌کنیم. مسیر Fixture فقط اتصال اجزا را می‌سنجد. حافظه فقط با کلیدهای مجاز وارد زمینه شد و هیچ فایل کاربر نوشته نشد.",
        "takeaway": "در تحویل خود، چهار چیز را جدا بنویسید: رفتار اندازه‌گیری‌شدهٔ مدل، تست اتصال اجزا، هزینهٔ اجرا و محدودیت‌های باقی‌مانده. کدام شاهد می‌تواند ادعای شما دربارهٔ هرکدام را رد کند؟",
    },
}
