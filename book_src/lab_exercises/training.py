"""Distinct learner exercises for the training, generation and advanced lessons.

Solutions are source-side material for the separate answer/validation path;
the learner notebook builder must never insert them into starter notebooks.
"""

EXERCISES = {
    "47-loop": {
        "title": "وزن واقعاً تغییر کرد؟",
        "goal": "یک گام آموزش را بنویسید و تغییر وزن را از صرف محاسبهٔ مشتق جدا کنید.",
        "prerequisite": "forward، backward و Optimizer؛ ورودی و Target با شکل (B,T).",
        "predict": "اگر backward اجرا شود ولی step اجرا نشود، کدام عددها تغییر می‌کنند؟",
        "setup": """import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
def make_model():
    torch.manual_seed(7)
    return MiniGPT(ModelConfig(12, 8, 16, 2, 1, 0.0))
x = torch.tensor([[1,2,3,4]])
y = torch.tensor([[2,3,4,5]])
model = make_model()
print('input / targets:', x.tolist(), y.tolist())
print('initial loss:', model(x,y)[1].item())
""",
        "task": "تابع update_step(model, optimizer, x, y) را بنویسید: حالت آموزش، پاک‌کردن مشتق قبلی، محاسبهٔ Loss، backward و step. مقدار Loss پیش از تغییر وزن را به‌صورت float برگردانید.",
        "starter": """def update_step(model, optimizer, x, y):
    # TODO: یک گام آموزش
    return None
""",
        "check": """def test_exercise():
    candidate = make_model()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.01)
    before = candidate.token_embedding.weight.detach().clone()
    result = update_step(candidate, optimizer, x, y)
    if result is None:
        return False
    assert isinstance(result, float) and result > 0
    assert not torch.equal(before, candidate.token_embedding.weight)
    assert candidate.training
    reference = make_model()
    opt = torch.optim.AdamW(reference.parameters(), lr=0.01)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        reference(x,y)[1].backward()
        opt.step()
    update_step(candidate, optimizer, x, y)
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(actual, expected)
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: update_step')
""",
        "solution": """def update_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(x,y)[1]
    loss.backward()
    optimizer.step()
    return loss.item()
""",
        "vary": "فقط نرخ را عوض کنید. وزن آغازین، ورودی و Gradient یکسان بمانند؛ آیا اندازهٔ تغییر هم یکسان است؟",
        "vary_code": """for rate in (0.001, 0.01):
    trial = make_model()
    optimizer = torch.optim.SGD(trial.parameters(), lr=rate)
    before = trial.token_embedding.weight.detach().clone()
    trial(x,y)[1].backward()
    optimizer.step()
    print(rate, (trial.token_embedding.weight.detach()-before).norm().item())
""",
        "debug": "دو بار مشتق یک عبارت را گرفته‌ایم، بی‌آنکه مشتق قبلی پاک شود. gradient_once(parameter) را تعمیر کنید: مشتق (parameter-3)**2 را مستقل از فراخوانی‌های قبلی برگرداند و وزن را تغییر ندهد.",
        "bug_code": """w = torch.nn.Parameter(torch.tensor(1.0))
for _ in range(2):
    ((w-3)**2).backward()
print('accumulated gradient:', w.grad.item(), 'single gradient:', -4.0)
""",
        "fix": """def gradient_once(parameter):
    # TODO: مشتق قبلی نباید جمع شود
    return None
""",
        "fix_check": """def test_repair():
    w = torch.nn.Parameter(torch.tensor(1.0))
    result = gradient_once(w)
    if result is None:
        return False
    assert float(result) == -4.0
    assert float(gradient_once(w)) == -4.0
    assert w.item() == 1.0
    assert float(gradient_once(torch.nn.Parameter(torch.tensor(4.0)))) == 2.0
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: gradient_once')
""",
        "fix_solution": """def gradient_once(parameter):
    parameter.grad = None
    ((parameter-3)**2).backward()
    return parameter.grad.detach().clone()
""",
        "connection": "مدل، MiniGPT واقعی است؛ حلقهٔ کوتاه این دفتر همان عملیات مرکزی mini_gpt/train.py را بدون گزارش و ذخیره اجرا می‌کند.",
        "takeaway": "کدام آزمون نشان داد که دو گام مستقل نوشته‌اید، نه دو backward با مشتق انباشته؟",
    },
    "48-evaluate": {
        "title": "دستهٔ آخر نباید امتیاز اضافی بگیرد",
        "goal": "میانگین را بر تعداد Targetها وزن دهید و آن را با evaluate واقعی مقایسه کنید.",
        "prerequisite": "Loss میانگین هر Batch و حالت eval/no_grad.",
        "predict": "میانگین خطاهای ۱ و ۳، با تعداد هدف‌های ۸ و ۲، باید به کدام مقدار نزدیک‌تر باشد؟",
        "setup": """import math
import torch
from torch.utils.data import DataLoader
from mini_gpt.dataset import NextTokenDataset
from mini_gpt.evaluate import evaluate
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(12,4,8,2,1,0.2))
data = NextTokenDataset([1,2,3,4,5,6,7,8],3)
print('windows:', len(data), 'real evaluation:', evaluate(model, DataLoader(data,batch_size=2), 'cpu'))
""",
        "task": "weighted_mean(losses, counts) میانگین Loss هر دسته و تعداد Targetهایش را می‌گیرد و میانگین کل را برمی‌گرداند. ورودی تمرین فهرست‌های هم‌اندازه و ناتهی با counts مثبت است.",
        "starter": """def weighted_mean(losses, counts):
    # TODO: هر Target سهم برابر دارد
    return None
""",
        "check": """def test_exercise():
    result = weighted_mean([1.0,3.0], [8,2])
    if result is None:
        return False
    assert math.isclose(result,1.4)
    assert math.isclose(weighted_mean([1.0,3.0],[2,2]),2.0)
    assert math.isclose(weighted_mean([2.5],[7]),2.5)
    losses, counts = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in DataLoader(data,batch_size=2):
            losses.append(model(inputs,targets)[1].item())
            counts.append(targets.numel())
    assert math.isclose(weighted_mean(losses,counts), evaluate(model,DataLoader(data,batch_size=2),'cpu'), abs_tol=1e-7)
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: weighted_mean')
""",
        "solution": """def weighted_mean(losses, counts):
    return sum(loss*count for loss,count in zip(losses,counts))/sum(counts)
""",
        "vary": "فقط batch_size ارزیابی را عوض کنید. چرا Loss کل باید تقریباً ثابت بماند؟",
        "vary_code": """before = {name:value.clone() for name,value in model.state_dict().items()}
for size in (1,2,3):
    print(size, evaluate(model,DataLoader(data,batch_size=size),'cpu'))
assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())
""",
        "debug": "میانگین Perplexity دسته‌ها با exp میانگین Loss یکی نیست. perplexity_from_batches(losses, counts) را با میانگینِ وزن‌دار Loss تعمیر کنید.",
        "bug_code": """wrong = (8*math.exp(1.0)+2*math.exp(3.0))/10
print('wrong mean of perplexities:', wrong)
print('exp of mean loss:', math.exp(1.4))
""",
        "fix": """def perplexity_from_batches(losses, counts):
    # TODO: ابتدا Loss کل، سپس exp
    return None
""",
        "fix_check": """def test_repair():
    result = perplexity_from_batches([1.0,3.0],[8,2])
    if result is None:
        return False
    assert math.isclose(result, math.exp(1.4))
    assert math.isclose(perplexity_from_batches([0.0,0.0],[1,4]),1.0)
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: perplexity_from_batches')
""",
        "fix_solution": """def perplexity_from_batches(losses, counts):
    mean_loss = sum(loss*count for loss,count in zip(losses,counts))/sum(counts)
    return math.exp(mean_loss)
""",
        "connection": "evaluate پروژه دقیقاً از targets.numel() برای وزن‌دهی استفاده می‌کند. پنجره‌ها هم‌پوشان‌اند؛ این عدد را با سنجش تک‌گذری یک متن دیگر یکی ندانید.",
        "takeaway": "برابری نتیجه با چند batch_size کدام خطای پیاده‌سازی را آشکار می‌کند و چه چیزی دربارهٔ کیفیت زبان نمی‌گوید؟",
    },
    "49-rate": {
        "title": "جهت مشتق را نگه دارید، اندازه‌اش را محدود کنید",
        "goal": "محدودسازی Norm را از بریدن جداگانهٔ مؤلفه‌ها و از اندازهٔ نهایی update جدا کنید.",
        "prerequisite": "بردار، Norm و ترتیب backward تا optimizer.step.",
        "predict": "بردار [3,4] با سقف Norm برابر ۱، به [1,1] تبدیل می‌شود یا برداری هم‌جهت با خودش؟",
        "setup": """import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(12,4,8,2,1,0.0))
x,y = torch.tensor([[1,2,3]]),torch.tensor([[2,3,4]])
model(x,y)[1].backward()
norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
print('norm before clipping:', norm.item())
""",
        "task": "clip_vector(gradient, limit) برای بردار float و limit مثبت، یک بردار تازه با همان جهت و Norm حداکثر limit برگرداند. اگر Norm کوچک است یا بردار صفر است، مقدارها تغییر نکنند.",
        "starter": """def clip_vector(gradient, limit):
    # TODO: یک ضریب مشترک برای کل بردار
    return None
""",
        "check": """def test_exercise():
    source = torch.tensor([3.0,4.0])
    result = clip_vector(source,1.0)
    if result is None:
        return False
    torch.testing.assert_close(result,torch.tensor([0.6,0.8]))
    torch.testing.assert_close(source,torch.tensor([3.0,4.0]))
    torch.testing.assert_close(clip_vector(torch.tensor([0.1,0.2]),1.0),torch.tensor([0.1,0.2]))
    torch.testing.assert_close(clip_vector(torch.zeros(3),1.0),torch.zeros(3))
    torch.testing.assert_close(clip_vector(torch.tensor([-6.0,8.0]),2.0),torch.tensor([-1.2,1.6]))
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: clip_vector')
""",
        "solution": """def clip_vector(gradient, limit):
    norm = gradient.norm().item()
    factor = min(1.0, limit/norm) if norm else 1.0
    return gradient*factor
""",
        "vary": "Gradient ثابت است؛ فقط Learning rate را عوض کنید. این مثال SGD است، نه فرمول کامل AdamW.",
        "vary_code": """gradient = torch.tensor([0.6,0.8])
for rate in (0.01,0.1,1.0):
    update = -rate*gradient
    print(rate, 'update norm:', update.norm().item())
""",
        "debug": "اگر ابتدا step را انجام دهیم، محدودکردن مشتق دیگر آن تغییر وزن را اصلاح نمی‌کند. clipped_sgd(value, gradient, rate, limit) را برای عددهای scalar بنویسید: اول gradient را به بازهٔ مجاز ببرید و سپس وزن تازه را برگردانید.",
        "bug_code": """value, gradient, rate, limit = 2.0, 10.0, 0.1, 1.0
wrong_value = value-rate*gradient
gradient = max(-limit,min(limit,gradient))
print('late clipping leaves weight at:',wrong_value)
""",
        "fix": """def clipped_sgd(value, gradient, rate, limit):
    # TODO: ترتیب محدودسازی و update
    return None
""",
        "fix_check": """def test_repair():
    result = clipped_sgd(2.0,10.0,0.1,1.0)
    if result is None:
        return False
    assert abs(result-1.9)<1e-8
    assert abs(clipped_sgd(2.0,-10.0,0.1,1.0)-2.1)<1e-8
    assert abs(clipped_sgd(2.0,0.5,0.1,1.0)-1.95)<1e-8
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: clipped_sgd')
""",
        "fix_solution": """def clipped_sgd(value, gradient, rate, limit):
    return value-rate*max(-limit,min(limit,gradient))
""",
        "connection": "train.py مقدار بازگشتی clip_grad_norm_ را پیش از step ثبت می‌کند. این مقدار Norm پیش از محدودسازی است، نه اندازهٔ تغییر وزن AdamW.",
        "takeaway": "اگر Norm محدود شد ولی Loss همچنان نامتناهی بود، چرا بزرگ‌ترکردن سقف راه‌حل قابل اتکایی نیست؟",
    },
    "49b-schedule": {
        "title": "ادامهٔ آموزش، ادامهٔ همان ساعت است",
        "goal": "یک نرخ کسینوسی با افق ثابت بنویسید و خطای شروع دوبارهٔ شمارنده را پیدا کنید.",
        "prerequisite": "نرخ پایه، شمارهٔ گام از ۱، Warmup و کسینوس درس.",
        "predict": "با پایان کاهش در گام ۶، آیا توقف در گام ۳ باید نرخ گام ۴ را عوض کند؟",
        "setup": """import math
from mini_gpt.schedule import ScheduleConfig
schedule = ScheduleConfig('cosine',2,6,0.1)
print('reference step 1:',schedule.learning_rate(0.001,1))
print('reference step 6:',schedule.learning_rate(0.001,6))
""",
        "task": "cosine_rate(step, peak, warmup, end, ratio) را برای step>=1 و end>warmup>=0 بنویسید. Warmup تا خود گام warmup است؛ پس از end نرخ روی peak*ratio می‌ماند. حالت warmup=0 نیز معتبر است.",
        "starter": """def cosine_rate(step, peak, warmup, end, ratio):
    # TODO: نرخ همان شمارهٔ گام، بدون بازتنظیم افق
    return None
""",
        "check": """def test_exercise():
    result = cosine_rate(1,0.001,2,6,0.1)
    if result is None:
        return False
    for warmup,end in ((2,6),(0,6),(1,4)):
        oracle = ScheduleConfig('cosine',warmup,end,0.1)
        for step in (1,2,4,6,9):
            assert math.isclose(cosine_rate(step,0.001,warmup,end,0.1),oracle.learning_rate(0.001,step),abs_tol=1e-12)
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: cosine_rate')
""",
        "solution": """def cosine_rate(step, peak, warmup, end, ratio):
    if step<=warmup:
        return peak*step/warmup
    u = min(1.0,(step-warmup)/(end-warmup))
    return peak*(ratio+(1-ratio)*(1+math.cos(math.pi*u))/2)
""",
        "vary": "فقط طول Warmup را از ۲ به ۴ تغییر دهید؛ اوج و پایان کاهش ثابت‌اند.",
        "vary_code": """for warmup in (2,4):
    trial = ScheduleConfig('cosine',warmup,8,0.1)
    print(warmup,[round(trial.learning_rate(0.001,s),6) for s in range(1,10)])
""",
        "debug": "پس از سه گام، نرخ‌های ادامه را اشتباهاً از شمارهٔ ۱ گرفته‌ایم. resumed_rates(schedule, peak, completed, count) باید نرخ count گام بعد از completed را برگرداند.",
        "bug_code": """wrong = [schedule.learning_rate(0.001,s) for s in range(1,4)]
expected_steps = [4,5,6]
print('restarted rates:',wrong,'but next step numbers are:',expected_steps)
""",
        "fix": """def resumed_rates(schedule, peak, completed, count):
    # TODO: completed گام انجام شده است
    return None
""",
        "fix_check": """def test_repair():
    result = resumed_rates(schedule,0.001,3,3)
    if result is None:
        return False
    assert result == [schedule.learning_rate(0.001,s) for s in (4,5,6)]
    assert resumed_rates(schedule,0.001,0,1)==[0.0005]
    assert resumed_rates(schedule,0.001,6,0)==[]
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: resumed_rates')
""",
        "fix_solution": """def resumed_rates(schedule, peak, completed, count):
    return [schedule.learning_rate(peak,s) for s in range(completed+1,completed+count+1)]
""",
        "connection": "ScheduleConfig همان کلاس مصرف‌شده در train.py است. پایان کاهش از Checkpoint بازیابی می‌شود و به --steps تازه وابسته نیست.",
        "takeaway": "برابری فرمول نرخ چرا به‌تنهایی برتری کیفیت آموزش را ثابت نمی‌کند؟",
    },
    "50-checkpoint": {
        "title": "یک فایل سالم، یک معنی ثابت",
        "goal": "بازیابی مدل را با وزن‌ها و ترتیب واژگان بیازمایید، نه فقط با بازشدن فایل.",
        "prerequisite": "state_dict، Tokenizer و کپی مستقل Tensor.",
        "predict": "اگر دو نویسه در واژگان جابه‌جا شوند ولی شکل همهٔ وزن‌ها درست باشد، کدام آزمون باید شکست بخورد؟",
        "setup": """import copy
import tempfile
from pathlib import Path
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharacterTokenizer
from mini_gpt.checkpoint import save_checkpoint,load_checkpoint
torch.set_num_threads(1)
torch.manual_seed(7)
tokenizer = CharacterTokenizer.from_text('abc')
model = MiniGPT(ModelConfig(tokenizer.vocab_size,4,8,2,1,0.0))
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
x,y = torch.tensor([[1,2,3]]),torch.tensor([[2,3,1]])
loss = model(x,y)[1]
loss.backward()
optimizer.step()
with tempfile.TemporaryDirectory(prefix='aibook-checkpoint-') as directory:
    path = Path(directory)/'example.pt'
    save_checkpoint(path,model,tokenizer,optimizer,1,{},torch.Generator().manual_seed(8),loss.item())
    restored,restored_tokenizer,payload = load_checkpoint(path)
print('restored step:',payload['step'],'optimizer has history:',bool(payload['optimizer']['state']))
""",
        "task": "same_artifact(left, right, left_tokens, right_tokens) دو state_dict و دو فهرست واژگان را مقایسه کند. فقط اگر کلیدها، مقدار Tensorها و ترتیب واژگان یکسان‌اند True بدهد.",
        "starter": """def same_artifact(left, right, left_tokens, right_tokens):
    # TODO: شکل درست به‌تنهایی کافی نیست
    return None
""",
        "check": """def test_exercise():
    a,b = model.state_dict(),restored.state_dict()
    tokens = tokenizer.id_to_token
    result = same_artifact(a,b,tokens,restored_tokenizer.id_to_token)
    if result is None:
        return False
    assert result is True
    swapped = list(tokens)
    swapped[1],swapped[2] = swapped[2],swapped[1]
    assert same_artifact(a,b,tokens,swapped) is False
    changed = {key:value.clone() for key,value in b.items()}
    changed['token_embedding.weight'][1,0] += 1
    assert same_artifact(a,changed,tokens,tokens) is False
    assert same_artifact(a,{},tokens,tokens) is False
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: same_artifact')
""",
        "solution": """def same_artifact(left, right, left_tokens, right_tokens):
    return (left_tokens==right_tokens and left.keys()==right.keys()
            and all(torch.equal(left[key],right[key]) for key in left))
""",
        "vary": "فقط یک وزنِ مدل بازیابی‌شده را عوض کنید و اختلاف Logits ورودی ثابت را ببینید؛ مدل اصلی دست‌نخورده بماند.",
        "vary_code": """trial = copy.deepcopy(restored).eval()
model.eval()
with torch.no_grad():
    baseline = model(x)[0]
    trial.language_model_head.weight[1,0] += 0.5
    print('maximum logit difference:',(trial(x)[0]-baseline).abs().max().item())
""",
        "debug": "state_dict به‌تنهایی عکس مستقل وزن‌های در حال تغییر نیست. snapshot(model) را بنویسید تا همهٔ Tensorها detach و clone شوند.",
        "bug_code": """trial = copy.deepcopy(model)
wrong_snapshot = trial.state_dict()
old_value = wrong_snapshot['token_embedding.weight'][1,0].item()
with torch.no_grad():
    trial.token_embedding.weight[1,0] += 1
print('snapshot changed too:',wrong_snapshot['token_embedding.weight'][1,0].item()!=old_value)
""",
        "fix": """def snapshot(model):
    # TODO: نگاشت مستقل از تغییرهای بعدی مدل
    return None
""",
        "fix_check": """def test_repair():
    trial = copy.deepcopy(model)
    result = snapshot(trial)
    if result is None:
        return False
    old = result['token_embedding.weight'].clone()
    with torch.no_grad():
        trial.token_embedding.weight.add_(1)
    assert torch.equal(old,result['token_embedding.weight'])
    assert result.keys()==trial.state_dict().keys()
    assert all(not value.requires_grad for value in result.values())
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: snapshot')
""",
        "fix_solution": """def snapshot(model):
    return {name:value.detach().clone() for name,value in model.state_dict().items()}
""",
        "connection": "save_checkpoint/load_checkpoint واقعی استفاده شدند. metadata خالی این مثال برای نمایش بازیابی است؛ برای Resume کامل، فایل را با train.py بسازید. فایل موقت در پایان setup پاک شده است.",
        "takeaway": "چرا هم برابری Tensorها و هم برابری واژگان لازم بود؟",
    },
    "51-resume": {
        "title": "شش گام پیوسته یا چهار گام و ادامه؟",
        "goal": "دو اجرای واقعی را مقایسه کنید و وضعیت تصادفی انتخاب Batch را بازیابی کنید.",
        "prerequisite": "Checkpoint، AdamW و مولد عدد تصادفی.",
        "predict": "اگر وزن‌ها برابر باشند ولی وضعیت مولد Batch فرق کند، آیا ادامهٔ آموزش الزاماً برابر است؟",
        "setup": """import tempfile
from pathlib import Path
import torch
from mini_gpt.train import build_parser,train
from mini_gpt.checkpoint import load_checkpoint
torch.set_num_threads(1)
parser = build_parser()
with tempfile.TemporaryDirectory(prefix='aibook-resume-') as directory:
    root = Path(directory)
    corpus = root/'corpus.txt'
    corpus.write_text('abcde '*20,encoding='utf-8')
    common = ['--text',str(corpus),'--context-length','4','--embedding-dim','8',
              '--num-heads','2','--num-layers','1','--batch-size','2','--dropout','0.1',
              '--threads','1','--device','cpu','--eval-every','2','--seed','17']
    full_path = train(parser.parse_args(common+['--steps','6','--output',str(root/'full')]))
    split_path = train(parser.parse_args(common+['--steps','4','--output',str(root/'split')]))
    resumed_path = train(parser.parse_args(['--text',str(corpus),'--resume',str(split_path),
        '--output',str(root/'split'),'--steps','6','--eval-every','2','--threads','1','--device','cpu']))
    _,_,full_payload = load_checkpoint(full_path)
    _,_,resumed_payload = load_checkpoint(resumed_path)
print('total steps:',full_payload['step'],resumed_payload['step'])
""",
        "task": "different_tensors(left, right) برای دو state_dict با کلیدهای برابر، نام Tensorهای نابرابر را به‌ترتیب الفبایی برگرداند. از برابری کامل استفاده کنید، نه Loss گرد‌شده.",
        "starter": """def different_tensors(left, right):
    # TODO: فهرست نام Tensorهای متفاوت
    return None
""",
        "check": """def test_exercise():
    result = different_tensors(full_payload['model'],resumed_payload['model'])
    if result is None:
        return False
    assert result==[],result
    assert different_tensors({'b':torch.tensor([1]),'a':torch.tensor([2])},
                             {'b':torch.tensor([2]),'a':torch.tensor([3])})==['a','b']
    assert different_tensors({'x':torch.tensor([1.0])},{'x':torch.tensor([1.0001])})==['x']
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: different_tensors')
""",
        "solution": """def different_tensors(left, right):
    return sorted(name for name in left if not torch.equal(left[name],right[name]))
""",
        "vary": "فقط وضعیت مولد انتخاب Batch را عوض کنید؛ محدوده و تعداد نمونه‌ها ثابت بمانند.",
        "vary_code": """for seed in (17,18):
    generator = torch.Generator().manual_seed(seed)
    print(seed,torch.randint(100,(8,),generator=generator).tolist())
""",
        "debug": "ساخت دوبارهٔ Generator با Seed اولیه، وضعیت میانهٔ اجرا را برنمی‌گرداند. restore_batch_generator(payload) یک Generator تازه با rng_batches ذخیره‌شده بسازد.",
        "bug_code": """wrong = torch.Generator().manual_seed(18)
saved = torch.Generator()
saved.set_state(resumed_payload['rng_batches'])
print('from initial seed:',torch.randint(100,(8,),generator=wrong).tolist())
print('from saved state:',torch.randint(100,(8,),generator=saved).tolist())
""",
        "fix": """def restore_batch_generator(payload):
    # TODO: از وضعیت ذخیره‌شده، نه Seed آغازین
    return None
""",
        "fix_check": """def test_repair():
    generator = restore_batch_generator(resumed_payload)
    if generator is None:
        return False
    expected = torch.Generator()
    expected.set_state(resumed_payload['rng_batches'])
    assert torch.equal(torch.randint(100,(12,),generator=generator),torch.randint(100,(12,),generator=expected))
    another = restore_batch_generator(resumed_payload)
    assert torch.equal(another.get_state(),resumed_payload['rng_batches'])
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: restore_batch_generator')
""",
        "fix_solution": """def restore_batch_generator(payload):
    generator = torch.Generator()
    generator.set_state(payload['rng_batches'])
    return generator
""",
        "connection": "هر سه اجرا از train واقعی‌اند؛ فایل‌ها در پوشه‌های موقت مستقل ساخته و سپس پاک شدند. برابری بیت‌به‌بیت این آزمون مربوط به CPU و محیط ثابت همین اجراست، نه وعده‌ای برای هر GPU یا نسخه.",
        "takeaway": "علاوه بر rng_batches، کدام وضعیت‌های Checkpoint برای ادامهٔ دقیق لازم‌اند؟",
    },
    "52-first-run": {
        "title": "از اجرای واقعی تا گزارش قابل بررسی",
        "goal": "گزارش CSV و بازرسی مدل یک اجرای کوتاه را بخوانید و ناسازگاری سابقه را تشخیص دهید.",
        "prerequisite": "ساختار Checkpoint، CSV و مفهوم Validation.",
        "predict": "اگر آخرین گام بهترین Validation را نداشته باشد، در خلاصهٔ اجرا کدام دو شماره باید جدا بمانند؟",
        "setup": """import csv
import math
import tempfile
from pathlib import Path
import torch
from mini_gpt.train import build_parser,train
from mini_gpt.checkpoint import load_checkpoint
from mini_gpt.inspect import inspect_model
torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='aibook-run-') as directory:
    root = Path(directory)
    corpus = root/'corpus.txt'
    corpus.write_text('مدل با داده آموزش می‌بیند. '*12,encoding='utf-8')
    path = train(build_parser().parse_args(['--text',str(corpus),'--output',str(root/'run'),
        '--steps','6','--eval-every','2','--context-length','8','--embedding-dim','8',
        '--num-heads','2','--num-layers','1','--batch-size','2','--threads','1','--device','cpu']))
    with (root/'run'/'metrics.csv').open(encoding='utf-8',newline='') as stream:
        rows = list(csv.DictReader(stream))
    model,tokenizer,payload = load_checkpoint(path)
report = inspect_model(model,tokenizer,'مدل ',layer=0,head=0,max_tokens=8,generate_tokens=0,greedy=True)
print('actual trace shapes:',report['shapes'])
print('logged steps:',[row['step'] for row in rows])
""",
        "task": "summarize_metrics(rows) فهرست ناتهی سطرهای CSV را بگیرد و dict با last_step، best_step و best_validation برگرداند. step را int و validation_loss را float کنید؛ در تساوی بهترین خطا، نخستین سطر را انتخاب کنید.",
        "starter": """def summarize_metrics(rows):
    # TODO: آخرین گام و بهترین ارزیابی یکی نیستند
    return None
""",
        "check": """def test_exercise():
    sample = [{'step':'1','validation_loss':'2.0'},{'step':'4','validation_loss':'1.0'},
              {'step':'6','validation_loss':'1.5'}]
    result = summarize_metrics(sample)
    if result is None:
        return False
    assert result=={'last_step':6,'best_step':4,'best_validation':1.0}
    assert summarize_metrics([sample[0]])=={'last_step':1,'best_step':1,'best_validation':2.0}
    actual = summarize_metrics(rows)
    assert actual['last_step']==payload['step']==6
    assert actual['best_validation']==min(float(row['validation_loss']) for row in rows)
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: summarize_metrics')
""",
        "solution": """def summarize_metrics(rows):
    best = min(rows,key=lambda row:float(row['validation_loss']))
    return {'last_step':int(rows[-1]['step']),'best_step':int(best['step']),
            'best_validation':float(best['validation_loss'])}
""",
        "vary": "فقط شمارهٔ Head در همان Checkpoint را عوض کنید؛ ورودی و Layer ثابت‌اند.",
        "vary_code": """for head in (0,1):
    item = inspect_model(model,tokenizer,'مدل ',layer=0,head=head,max_tokens=8,generate_tokens=0,greedy=True)
    print('head',head,'last attention row:',item['attention']['weights'][-1])
""",
        "debug": "چسباندن گزارش دو اجرا ممکن است گام‌های تکراری یا عقب‌رفته بسازد. valid_metrics(rows) باید برای گام‌های مثبت و اکیداً صعودی و validation_loss متناهی True بدهد؛ گزارش خالی نامعتبر است.",
        "bug_code": """mixed = rows+rows[:1]
print('mixed history:',[row['step'] for row in mixed])
print('all rows individually have numbers, but the history restarts')
""",
        "fix": """def valid_metrics(rows):
    # TODO: هم عددها و هم ترتیب سابقه را بررسی کنید
    return None
""",
        "fix_check": """def test_repair():
    result = valid_metrics(rows)
    if result is None:
        return False
    assert result is True
    assert valid_metrics(rows+rows[:1]) is False
    assert valid_metrics([]) is False
    assert valid_metrics([{'step':'1','validation_loss':'nan'}]) is False
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: valid_metrics')
""",
        "fix_solution": """def valid_metrics(rows):
    if not rows:
        return False
    steps = [int(row['step']) for row in rows]
    return (steps[0]>0 and all(a<b for a,b in zip(steps,steps[1:]))
            and all(math.isfinite(float(row['validation_loss'])) for row in rows))
""",
        "connection": "آموزش، CSV، Checkpoint و inspect همگی از پیاده‌سازی واقعی‌اند. train_batch_loss قبل از update و validation_loss پس از آن محاسبه می‌شود؛ این دو ستون ارزیابی همسانِ دو Split نیستند.",
        "takeaway": "کدام اطلاعات لازم است تا کسی دیگر همین نمودار و همین نقشهٔ Attention را بازسازی کند؟",
    },
    "53-curves": {
        "title": "منحنی را بخوانید، نتیجه را محدود کنید",
        "goal": "توقف بر اساس Validation را روی یک منحنی معلوم پیاده کنید و Test را از انتخاب تنظیمات بیرون نگه دارید.",
        "prerequisite": "تفاوت Training، Validation و Test؛ عدد کمتر همیشه ادعای بزرگ‌تر نمی‌دهد.",
        "predict": "اگر Training بهتر شود ولی Validation بعد از دو ارزیابی بدتر شود، آیا باید آخرین وزن را بهترین بدانیم؟",
        "setup": """import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
from mini_gpt.dataset import NextTokenDataset
from mini_gpt.evaluate import evaluate
torch.set_num_threads(1)
torch.manual_seed(7)
data = NextTokenDataset([1,2,3,4,1,2,3,4],3)
model = MiniGPT(ModelConfig(5,4,8,2,1,0.0))
print('real untrained-model loss:',evaluate(model,DataLoader(data,batch_size=2),'cpu'))
# Synthetic curves: these are not measurements from MiniGPT.
train_curve = [2.4,2.0,1.7,1.4,1.1,0.8]
valid_curve = [2.5,2.2,2.0,2.05,2.1,2.2]
plt.plot(range(1,7),train_curve,'o-',label='Synthetic train')
plt.plot(range(1,7),valid_curve,'o-',label='Synthetic validation')
plt.xlabel('Evaluation number'); plt.ylabel('Loss'); plt.legend(); plt.show()
""",
        "task": "early_stop(values, patience) نخستین شمارهٔ ارزیابی، از ۱، را برگرداند که patience ارزیابی پیاپی بهبودِ اکید نداشته‌ایم. اگر چنین گامی نیست None بدهید. با هر بهبود شمارنده صفر می‌شود؛ تساوی بهبود نیست.",
        "starter": """def early_stop(values, patience):
    # TODO: بهترین مقدار و تعداد ارزیابی‌های بدون بهبود
    return None
""",
        "check": """def test_exercise():
    result = early_stop(valid_curve,2)
    if result is None:
        return False
    assert result==5
    assert early_stop([3,2,1],2) is None
    assert early_stop([2,2,2],2)==3
    assert early_stop([3,4,2,3,4],2)==5
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: early_stop')
""",
        "solution": """def early_stop(values, patience):
    best, stale = float('inf'),0
    for number,value in enumerate(values,1):
        if value<best:
            best,stale = value,0
        else:
            stale += 1
        if stale>=patience:
            return number
    return None
""",
        "vary": "فقط Seed آغازین مدل را عوض کنید؛ داده و معماری ثابت‌اند. این‌ها Loss پیش از آموزش‌اند، نه مقایسهٔ کیفیت نهایی.",
        "vary_code": """for seed in (7,8,9):
    torch.manual_seed(seed)
    trial = MiniGPT(ModelConfig(5,4,8,2,1,0.0))
    print(seed,evaluate(trial,DataLoader(data,batch_size=2),'cpu'))
""",
        "debug": "انتخاب Checkpoint با نگاه‌کردن به Test، استقلال آن را از بین می‌برد. select_by_validation(values) فقط فهرست Validation را بگیرد و اندیس صفرمبنای نخستین کمینه را بدهد.",
        "bug_code": """validation = [1.4,1.1,1.3]
test = [1.0,1.2,1.15]
wrong_index = min(range(len(test)),key=lambda i:test[i])
print('selected using test:',wrong_index,'validation at that point:',validation[wrong_index])
""",
        "fix": """def select_by_validation(values):
    # TODO: Test وارد انتخاب نمی‌شود
    return None
""",
        "fix_check": """def test_repair():
    result = select_by_validation([1.4,1.1,1.3])
    if result is None:
        return False
    assert result==1
    assert select_by_validation([2.0,2.0])==0
    assert select_by_validation([0.4])==0
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: select_by_validation')
""",
        "fix_solution": """def select_by_validation(values):
    return min(range(len(values)),key=lambda i:values[i])
""",
        "connection": "پروژه best.pt را با Validation نگه می‌دارد، اما Early stopping خودکار ندارد. تابع این دفتر یک تمرین مستقل است و رفتار train.py را تغییر نمی‌دهد.",
        "takeaway": "چرا یک منحنی با شکل بیش‌برازش، بدون شناخت داده و شیوهٔ سنجش، تشخیص قطعی علت نیست؟",
    },
    "54-generate": {
        "title": "حلقهٔ تولید را خودتان ببندید",
        "goal": "از Logits آخرین موقعیت یک ادامهٔ حریصانه بسازید و با generate واقعی مقایسه کنید.",
        "prerequisite": "شکل (B,T,V)، argmax و بریدن Context.",
        "predict": "وقتی Prompt از ظرفیت Context بلندتر است، طول خروجی باید از Prompt کوتاه‌تر شود یا فقط ورودی forward محدود می‌شود؟",
        "setup": """import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(12,4,8,2,1,0.0)).eval()
prompt = torch.tensor([[1,2,3]])
print('actual generation:',model.generate(prompt,3,greedy=True).tolist())
print('This untrained model demonstrates mechanics, not language quality.')
""",
        "task": "greedy_extend(model, ids, count) را بنویسید. مدلِ ورودی در eval است؛ در no_grad، هر بار آخرین context_length شناسه را بدهید، از آخرین موقعیت argmax بگیرید و به متن کامل اضافه کنید. ids اصلی را تغییر ندهید.",
        "starter": """def greedy_extend(model, ids, count):
    # TODO: هر بار فقط یک Token تازه
    return None
""",
        "check": """def test_exercise():
    result = greedy_extend(model,prompt,3)
    if result is None:
        return False
    assert torch.equal(result,model.generate(prompt,3,greedy=True))
    assert torch.equal(prompt,torch.tensor([[1,2,3]]))
    for ids,count in ((prompt,0),(torch.tensor([[1,2,3,4,5,6]]),2),
                      (torch.tensor([[1,2],[3,4]]),4)):
        assert torch.equal(greedy_extend(model,ids,count),model.generate(ids,count,greedy=True))
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: greedy_extend')
""",
        "solution": """def greedy_extend(model, ids, count):
    result = ids.clone()
    with torch.no_grad():
        for _ in range(count):
            logits,_ = model(result[:,-model.config.context_length:])
            next_id = logits[:,-1,:].argmax(-1,keepdim=True)
            result = torch.cat((result,next_id),dim=1)
    return result
""",
        "vary": "فقط تعداد Tokenهای تازه را تغییر دهید و تفاوت طول Context و متن برگشتی را ثبت کنید.",
        "vary_code": """for count in (0,2,6):
    result = model.generate(prompt,count,greedy=True)
    print(count,'full shape:',tuple(result.shape),'context limit:',model.config.context_length)
""",
        "debug": "argmax روی محور زمان، شناسهٔ Token نمی‌دهد. last_token_scores(logits) باید برای ورودی (B,T,V)، امتیازهای آخرین موقعیت با شکل (B,V) را برگرداند.",
        "bug_code": """with torch.no_grad():
    scores = model(prompt)[0]
wrong = scores.argmax(dim=1)
print('wrong shape:',tuple(wrong.shape),'these values index time, not vocabulary')
""",
        "fix": """def last_token_scores(logits):
    # TODO: آخرین موقعیت، همهٔ واژگان
    return None
""",
        "fix_check": """def test_repair():
    values = torch.arange(30).reshape(2,3,5)
    result = last_token_scores(values)
    if result is None:
        return False
    assert result.shape==(2,5)
    assert torch.equal(result,torch.tensor([[10,11,12,13,14],[25,26,27,28,29]]))
    assert last_token_scores(torch.zeros(1,1,7)).shape==(1,7)
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: last_token_scores')
""",
        "fix_solution": """def last_token_scores(logits):
    return logits[:,-1,:]
""",
        "connection": "تابع شما با MiniGPT.generate واقعی مقایسه می‌شود. نسخهٔ پروژه علاوه بر حلقه، حالت train/eval قبلی را برمی‌گرداند و گزینه‌های Sampling را اعتبارسنجی می‌کند.",
        "takeaway": "چرا محاسبهٔ امتیاز همهٔ موقعیت‌ها در forward به معنی اضافه‌کردن همهٔ آن پیش‌بینی‌ها به متن نیست؟",
    },
    "55-temperature": {
        "title": "دما، بدون تغییر وزن",
        "goal": "توزیع انتخاب را تغییر دهید و ثابت‌ماندن رتبه‌ها و وزن‌های مدل را بررسی کنید.",
        "prerequisite": "Softmax و تفاوت Logits با احتمال.",
        "predict": "اگر همهٔ Logits را در یک عدد مثبت تقسیم کنیم، آیا شناسهٔ بزرگ‌ترین امتیاز عوض می‌شود؟",
        "setup": """import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
from mini_gpt.sampling import sampling_distribution
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(8,4,8,2,1,0.0)).eval()
prompt = torch.tensor([[1,2,3]])
with torch.no_grad():
    actual_logits = model(prompt)[0][:,-1,:]
print('real model probabilities:',sampling_distribution(actual_logits).tolist())
""",
        "task": "temperature_probs(logits, temperature) برای Logits دوبعدی متناهی و دمای مثبت، Softmax امتیازهای تقسیم‌شده بر دما را در محور آخر برگرداند. برای دمای صفر یا منفی ValueError بدهید.",
        "starter": """def temperature_probs(logits, temperature):
    # TODO: تغییر احتمال‌ها، نه وزن‌ها
    return None
""",
        "check": """def test_exercise():
    logits = torch.tensor([[2.0,1.0,0.0],[0.0,0.0,0.0]])
    result = temperature_probs(logits,0.5)
    if result is None:
        return False
    torch.testing.assert_close(result,sampling_distribution(logits,temperature=0.5))
    torch.testing.assert_close(result.sum(-1),torch.ones(2))
    torch.testing.assert_close(temperature_probs(logits+9,2.0),temperature_probs(logits,2.0))
    torch.testing.assert_close(temperature_probs(actual_logits,1.3),sampling_distribution(actual_logits,temperature=1.3))
    for invalid in (0.0,-1.0):
        try:
            temperature_probs(logits,invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('reject nonpositive temperature')
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: temperature_probs')
""",
        "solution": """def temperature_probs(logits, temperature):
    if temperature<=0:
        raise ValueError('temperature must be positive')
    return (logits/temperature).softmax(-1)
""",
        "vary": "فقط دما را عوض کنید؛ مدل، Prompt و Seed ثابت‌اند. متن مدل آموزش‌ندیده معیار کیفیت نیست؛ تغییر احتمال و ادامه را ببینید.",
        "vary_code": """before = {name:value.clone() for name,value in model.state_dict().items()}
for temperature in (0.5,1.0,2.0):
    torch.manual_seed(19)
    print(temperature,model.generate(prompt,6,temperature=temperature).tolist())
assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())
""",
        "debug": "دمای صفر روش پیاده‌سازی Greedy نیست. greedy_ids(logits) را بدون تقسیم بنویسید؛ خروجی باید (B,1) باشد و در تساوی نخستین شناسه را انتخاب کند.",
        "bug_code": """try:
    model.generate(prompt,1,temperature=0)
except ValueError as error:
    print('expected failure:',error)
else:
    raise AssertionError('zero temperature must be rejected')
""",
        "fix": """def greedy_ids(logits):
    # TODO: انتخاب مستقیم بیشترین امتیاز
    return None
""",
        "fix_check": """def test_repair():
    result = greedy_ids(torch.tensor([[2.0,1.0],[0.0,3.0]]))
    if result is None:
        return False
    assert result.tolist()==[[0],[1]]
    assert greedy_ids(torch.zeros(1,4)).tolist()==[[0]]
    assert result.dtype==torch.long
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: greedy_ids')
""",
        "fix_solution": """def greedy_ids(logits):
    return logits.argmax(-1,keepdim=True)
""",
        "connection": "sampling_distribution و generate همان توابع پروژه‌اند. تغییر دما مرحلهٔ تولید را عوض می‌کند و هیچ آموزش تازه‌ای انجام نمی‌دهد.",
        "takeaway": "چرا تغییر Seed یا دما نمی‌تواند اطلاعاتی را که Tokenizer حذف کرده برگرداند؟",
    },
    "56-topkp": {
        "title": "گزینهٔ عبورکننده از آستانه را گم نکنید",
        "goal": "ماسک Top-p را بنویسید و رفتار Top-k در امتیازهای مساوی را تعمیر کنید.",
        "prerequisite": "مرتب‌سازی، مجموع تجمعی و بازگرداندن اندیس‌ها.",
        "predict": "برای [0.6,0.25,0.1,0.05] و p=0.8 چند گزینه لازم است؟",
        "setup": """import torch
from mini_gpt.sampling import filter_logits
torch.set_num_threads(1)
probabilities = torch.tensor([0.6,0.25,0.1,0.05])
print('project candidates:',torch.isfinite(filter_logits(probabilities.log()[None],top_p=0.8)).tolist())
""",
        "task": "nucleus_mask(probabilities, p) برای یک بردار احتمال مثبت با مجموع یک و 0<p<=1، ماسک bool هم‌اندازه در ترتیب اصلی بدهد. کوچک‌ترین پیشوند مرتب با مجموع حداقل p حفظ شود؛ در تساوی احتمال، ترتیب اصلی حفظ شود.",
        "starter": """def nucleus_mask(probabilities, p):
    # TODO: مرتب‌سازی، عبور از آستانه، بازگشت به ترتیب اصلی
    return None
""",
        "check": """def test_exercise():
    result = nucleus_mask(probabilities,0.8)
    if result is None:
        return False
    assert result.dtype==torch.bool and result.tolist()==[True,True,False,False]
    assert nucleus_mask(probabilities,0.01).tolist()==[True,False,False,False]
    assert nucleus_mask(torch.tensor([0.5,0.5]),0.5).tolist()==[True,False]
    assert nucleus_mask(torch.tensor([0.1,0.6,0.05,0.25]),0.8).tolist()==[False,True,False,True]
    assert nucleus_mask(probabilities,1.0).all()
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: nucleus_mask')
""",
        "solution": """def nucleus_mask(probabilities, p):
    order = torch.argsort(probabilities,descending=True,stable=True)
    result = torch.zeros_like(probabilities,dtype=torch.bool)
    total = 0.0
    for index in order.tolist():
        result[index] = True
        total += probabilities[index].item()
        if total>=p:
            break
    return result
""",
        "vary": "فقط p را تغییر دهید و تعداد نامزدها را با پیاده‌سازی واقعی اندازه بگیرید؛ این تعداد ثابت نیست.",
        "vary_code": """for p in (0.1,0.5,0.8,0.95,1.0):
    mask = torch.isfinite(filter_logits(probabilities.log()[None],top_p=p))
    print(p,mask.tolist(),int(mask.sum()))
""",
        "debug": "حذف امتیازهای کمتر از kاُم، هنگام تساوی بیش از k گزینه نگه می‌دارد. exact_topk_mask(logits,k) برای بردار یک‌بعدی و 1<=k<=V ماسکی با دقیقاً k خانهٔ True بدهد؛ هویت گزینه‌های مساوی در این تمرین مهم نیست.",
        "bug_code": """ties = torch.tensor([2.0,2.0,2.0,0.0])
wrong = ties>=ties.topk(2).values[-1]
print('wanted 2, kept:',int(wrong.sum()),wrong.tolist())
""",
        "fix": """def exact_topk_mask(logits, k):
    # TODO: از اندیس گزینه‌ها استفاده کنید
    return None
""",
        "fix_check": """def test_repair():
    result = exact_topk_mask(torch.tensor([2.0,2.0,2.0,0.0]),2)
    if result is None:
        return False
    assert result.dtype==torch.bool and result.sum().item()==2
    assert not result[-1]
    assert exact_topk_mask(torch.tensor([0.0,3.0,1.0]),1).tolist()==[False,True,False]
    assert exact_topk_mask(torch.zeros(4),4).all()
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: exact_topk_mask')
""",
        "fix_solution": """def exact_topk_mask(logits, k):
    result = torch.zeros_like(logits,dtype=torch.bool)
    result[logits.topk(k).indices] = True
    return result
""",
        "connection": "filter_logits در sampling.py با همین دو خطای مرزی روبه‌روست. در ترکیب دو روش، ابتدا Top-k اعمال و سپس Top-p روی توزیع باقی‌مانده حساب می‌شود.",
        "takeaway": "چرا ماسک درست باید هم احتمال تجمعی را رعایت کند و هم ترتیب اصلی واژگان را نگه دارد؟",
    },
    "57-prompts": {
        "title": "مدل واقعاً کدام متن را دیده است؟",
        "goal": "شناسه‌های ناشناخته و Context بریده‌شده را پیش از قضاوت دربارهٔ خروجی بررسی کنید.",
        "prerequisite": "CharacterTokenizer، شناسهٔ ناشناخته و Context Window.",
        "predict": "اگر دو Prompt بلند فقط در ابتدای حذف‌شده فرق کنند، آخرین ورودی مدل در تولید Greedy چه تفاوتی دارد؟",
        "setup": """import torch
from mini_gpt.tokenizer import CharacterTokenizer
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
tokenizer = CharacterTokenizer.from_text('مدل زبان ')
model = MiniGPT(ModelConfig(tokenizer.vocab_size,5,8,2,1,0.0)).eval()
text = 'مدل X زبان مدل'
ids = torch.tensor([tokenizer.encode(text)])
print('all IDs:',ids.tolist(),'effective IDs:',ids[:,-5:].tolist())
""",
        "task": "effective_ids(tokenizer, text, limit) دو list برگرداند: شناسه‌های کامل متن و حداکثر limit شناسهٔ آخر. متن ناتهی و limit مثبت است؛ واژگان را تغییر ندهید.",
        "starter": """def effective_ids(tokenizer, text, limit):
    # TODO: متن کامل و ورودی واقعی forward
    return None
""",
        "check": """def test_exercise():
    before = list(tokenizer.id_to_token)
    result = effective_ids(tokenizer,text,5)
    if result is None:
        return False
    full,context = result
    assert full==tokenizer.encode(text) and context==full[-5:]
    assert 0 in full
    assert effective_ids(tokenizer,'مدل',20)==(tokenizer.encode('مدل'),tokenizer.encode('مدل'))
    assert effective_ids(tokenizer,'مدل',1)[1]==tokenizer.encode('ل')
    assert tokenizer.id_to_token==before
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: effective_ids')
""",
        "solution": """def effective_ids(tokenizer, text, limit):
    full = tokenizer.encode(text)
    return full,full[-limit:]
""",
        "vary": "فقط پیشوندی را عوض کنید که بیرون Context است؛ پنج شناسهٔ آخر در هر دو Prompt ثابت بمانند.",
        "vary_code": """a = torch.tensor([tokenizer.encode('مدل زبان مدل')])
b = torch.tensor([tokenizer.encode('زبان زبان مدل')])
assert torch.equal(a[:,-5:],b[:,-5:])
print('next IDs:',model.generate(a,1,greedy=True)[0,-1].item(),model.generate(b,1,greedy=True)[0,-1].item())
""",
        "debug": "ساختن شناسهٔ تازه پس از آموزش، جدول Embedding را بزرگ نمی‌کند. encode_with_unknowns(tokenizer,text) باید همان encode معتبر را همراه تعداد صفرها برگرداند؛ برای متن خالی ValueError بدهد.",
        "bug_code": """try:
    model(torch.tensor([[tokenizer.vocab_size]]))
except ValueError as error:
    print('expected invalid new ID:',error)
else:
    raise AssertionError('out-of-vocabulary ID must be rejected')
""",
        "fix": """def encode_with_unknowns(tokenizer, text):
    # TODO: اطلاعات ناشناخته را گزارش کنید، نه اینکه واژگان را عوض کنید
    return None
""",
        "fix_check": """def test_repair():
    result = encode_with_unknowns(tokenizer,'مدل X')
    if result is None:
        return False
    assert result==(tokenizer.encode('مدل X'),1)
    assert encode_with_unknowns(tokenizer,'مدل')[1]==0
    assert encode_with_unknowns(tokenizer,'XY')[1]==2
    try:
        encode_with_unknowns(tokenizer,'')
    except ValueError:
        pass
    else:
        raise AssertionError('empty prompt must be rejected')
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: encode_with_unknowns')
""",
        "fix_solution": """def encode_with_unknowns(tokenizer, text):
    ids = tokenizer.encode(text)
    if not ids:
        raise ValueError('empty prompt')
    return ids,ids.count(0)
""",
        "connection": "این همان Tokenizer و generate پروژه است. متن برگشتی کامل می‌ماند ولی forward فقط پنجرهٔ آخر را می‌بیند؛ موقعیت‌ها در آن پنجره از صفر شروع می‌شوند.",
        "takeaway": "وقتی Prefix از Context خارج شده، آیا تغییر دما راهی برای بازیابی آن است؟",
    },
    "58-ablation": {
        "title": "آینده را عوض کنید، گذشته را بسنجید",
        "goal": "یک آزمون علّیت بنویسید که تنها یک کلید محاسبه را تغییر می‌دهد.",
        "prerequisite": "Causal mask و مدل در حالت eval.",
        "predict": "با ثابت‌ماندن سه Token نخست، تغییر دو Token آخر باید کدام Logits را ثابت نگه دارد؟",
        "setup": """import copy
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(12,8,16,2,2,0.0)).eval()
a,b = torch.tensor([[1,2,3,4,5]]),torch.tensor([[1,2,3,9,10]])
with torch.no_grad():
    trace = {}
    model(a,trace=trace)
print('future attention mass:',trace['layers'][0]['attention']['weights'].triu(1).sum().item())
""",
        "task": "past_difference(model, a, b, prefix, causal) در eval/no_grad، بیشترین قدرمطلق اختلاف Logits نخستین prefix موقعیت را برگرداند. مقدار causal را به هر دو forward بدهید و حالت قبلی مدل را برگردانید.",
        "starter": """def past_difference(model, a, b, prefix, causal):
    # TODO: ورودی‌ها فقط در آینده متفاوت‌اند
    return None
""",
        "check": """def test_exercise():
    model.train()
    result = past_difference(model,a,b,3,True)
    if result is None:
        return False
    assert result<1e-7
    assert model.training
    assert past_difference(model,a,b,3,False)>1e-6
    assert past_difference(model,a,a,3,False)==0
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: past_difference')
""",
        "solution": """def past_difference(model, a, b, prefix, causal):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            left = model(a,causal=causal)[0][:,:prefix]
            right = model(b,causal=causal)[0][:,:prefix]
            return (left-right).abs().max().item()
    finally:
        model.train(was_training)
""",
        "vary": "فقط use_positions را عوض کنید؛ در هر حالت دو جایگشت با Token آخر یکسان را مقایسه کنید. صفرشدن تفاوت را پیش‌فرض نگیرید.",
        "vary_code": """model.eval()
with torch.no_grad():
    for enabled in (False,True):
        left = model(torch.tensor([[1,2,3,4]]),use_positions=enabled)[0][:,-1]
        right = model(torch.tensor([[3,2,1,4]]),use_positions=enabled)[0][:,-1]
        print('positions:',enabled,'last difference:',(left-right).abs().max().item())
""",
        "debug": "دو مدل تصادفی مستقل، مقایسهٔ کنترل‌شده نیستند. paired_models(model) دو deepcopy مستقل از یک وضعیت یکسان برگرداند؛ تغییر یک نسخه نباید روی دیگری یا مدل پایه اثر بگذارد.",
        "bug_code": """left = MiniGPT(model.config)
right = MiniGPT(model.config)
print('independent initial weights equal:',torch.equal(left.token_embedding.weight,right.token_embedding.weight))
""",
        "fix": """def paired_models(model):
    # TODO: وزن آغازین برابر، حافظهٔ مستقل
    return None
""",
        "fix_check": """def test_repair():
    result = paired_models(model)
    if result is None:
        return False
    left,right = result
    assert left is not right and left is not model and right is not model
    assert all(torch.equal(left.state_dict()[name],value) for name,value in right.state_dict().items())
    with torch.no_grad():
        left.token_embedding.weight.add_(1)
    assert torch.equal(right.token_embedding.weight,model.token_embedding.weight)
    assert not torch.equal(left.token_embedding.weight,right.token_embedding.weight)
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: paired_models')
""",
        "fix_solution": """def paired_models(model):
    return copy.deepcopy(model),copy.deepcopy(model)
""",
        "connection": "causal و use_positions کلیدهای آزمایشی forward واقعی‌اند؛ train معمولی هر دو را فعال می‌گذارد. تغییر خروجی مدل تصادفی، برتری کیفیت پس از آموزش را ثابت نمی‌کند.",
        "takeaway": "آزمایش علّیت کدام ادعای ساختاری را می‌سنجد که یک متن تولیدی روان نمی‌تواند ثابت کند؟",
    },
    "59-capacity": {
        "title": "کدام اندازه را بزرگ کرده‌اید؟",
        "goal": "تعداد وزن‌ها و اندازهٔ جدول Attention را جدا بشمارید.",
        "prerequisite": "C، L، H، T و فرمول شمارش پارامترها.",
        "predict": "با C ثابت، دوبرابرکردن H کدام هزینه را عوض می‌کند و کدام شمارش را ثابت می‌گذارد؟",
        "setup": """import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
config = ModelConfig(12,8,16,2,1,0.0)
model = MiniGPT(config)
print('parameter tensors:',len(list(model.parameters())))
print('embedding shape:',tuple(model.token_embedding.weight.shape))
""",
        "task": "count_parameters(config) تعداد scalarهای قابل آموزش یک MiniGPT با این تنظیمات را برگرداند. تعداد شیءهای Parameter را با تعداد عددهای داخلشان اشتباه نگیرید.",
        "starter": """def count_parameters(config):
    # TODO: شمارش عددها در مدل واقعی
    return None
""",
        "check": """def test_exercise():
    result = count_parameters(config)
    if result is None:
        return False
    assert result==3824
    assert count_parameters(ModelConfig(12,8,16,2,2,0.0))==7104
    assert count_parameters(ModelConfig(12,8,32,2,1,0.0))==13792
    assert count_parameters(ModelConfig(12,8,16,4,1,0.0))==3824
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: count_parameters')
""",
        "solution": """def count_parameters(config):
    return sum(parameter.numel() for parameter in MiniGPT(config).parameters() if parameter.requires_grad)
""",
        "vary": "فقط C را عوض کنید؛ V، T، H و L ثابت بمانند. رشد وزن‌ها را با رشد عرض مقایسه کنید.",
        "vary_code": """for width in (8,16,32):
    trial = MiniGPT(ModelConfig(12,8,width,2,1,0.0))
    print(width,sum(parameter.numel() for parameter in trial.parameters()))
""",
        "debug": "شمردن جدول Attention با B*T*C تعداد نمایش‌ها را می‌دهد، نه امتیازهای هر Head. attention_elements(batch, heads, length) تعداد خانه‌های جدول (B,H,T,T) را حساب کند.",
        "bug_code": """B,H,T,C = 2,4,8,16
print('wrong attention count:',B*T*C)
print('actual score tensor shape:',(B,H,T,T))
""",
        "fix": """def attention_elements(batch, heads, length):
    # TODO: دو محور زمان داریم
    return None
""",
        "fix_check": """def test_repair():
    result = attention_elements(2,4,8)
    if result is None:
        return False
    assert result==512
    assert attention_elements(2,4,16)==4*result
    assert attention_elements(1,1,3)==9
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: attention_elements')
""",
        "fix_solution": """def attention_elements(batch, heads, length):
    return batch*heads*length*length
""",
        "connection": "شمارش وزن از MiniGPT واقعی است. جدول کامل Attention نیز شکل همان محاسبهٔ سادهٔ پروژه را دارد؛ تعداد پارامتر به‌تنهایی تخمین هزینهٔ اجرا نیست.",
        "takeaway": "چرا مقایسهٔ دو مدل با تعداد step برابر، مقایسه با زمان یا حافظهٔ برابر نیست؟",
    },
    "60-bug-clinic": {
        "title": "شکل درست، محاسبهٔ غلط",
        "goal": "محور کلاس در Cross-Entropy و ترتیب واقعی خانه‌ها در تقسیم Headها را آزمایش کنید.",
        "prerequisite": "reshape، transpose و معنای محورهای (B,T,V) و (B,T,C).",
        "predict": "وقتی T و V برابر باشند، چرا یک خطای محور ممکن است بدون پیام خطا اجرا شود؟",
        "setup": """import torch
from torch.nn import functional as F
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(5,4,8,2,1,0.0))
x,y = torch.tensor([[1,2,3],[2,3,4]]),torch.tensor([[2,3,4],[3,4,1]])
logits,real_loss = model(x,y)
print('actual logits:',tuple(logits.shape),'actual targets:',tuple(y.shape))
try:
    F.cross_entropy(logits,y)
except RuntimeError as error:
    print('expected wrong-axis failure:',error)
""",
        "task": "last_axis_loss(logits, targets) برای Logits با شکل (B,T,V) و Target با شکل (B,T)، میانگین Cross-Entropy روی B*T هدف را برگرداند. تست‌ها فقط Shape را بررسی نمی‌کنند.",
        "starter": """def last_axis_loss(logits, targets):
    # TODO: محور Vocabulary باید محور کلاس باشد
    return None
""",
        "check": """def test_exercise():
    result = last_axis_loss(logits,y)
    if result is None:
        return False
    torch.testing.assert_close(result,real_loss)
    for shape in ((2,3,3),(1,2,7)):
        scores = torch.arange(shape[0]*shape[1]*shape[2],dtype=torch.float32).reshape(shape)/7
        targets = torch.zeros(shape[:2],dtype=torch.long)
        expected = -scores.log_softmax(-1).gather(-1,targets[...,None]).mean()
        torch.testing.assert_close(last_axis_loss(scores,targets),expected)
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: last_axis_loss')
""",
        "solution": """def last_axis_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
""",
        "vary": "فقط برابر یا نابرابر بودن T و V را عوض کنید؛ اجرای بدون استثنا را با برابری مقدار اشتباه نگیرید.",
        "vary_code": """scores = torch.arange(18,dtype=torch.float32).reshape(2,3,3)/4
targets = torch.tensor([[0,1,2],[2,0,1]])
wrong = F.cross_entropy(scores,targets)
correct = -scores.log_softmax(-1).gather(-1,targets[...,None]).mean()
print('same shape accepted; wrong:',wrong.item(),'correct:',correct.item())
""",
        "debug": "reshape مستقیم به (B,H,T,D)، ترتیب خانه‌ها را مثل transpose جابه‌جا نمی‌کند. split_heads(x,heads) ورودی (B,T,C) را به (B,H,T,C/H) با حفظ معنی خانه‌ها تبدیل کند.",
        "bug_code": """values = torch.arange(2*3*8).reshape(2,3,8)
wrong = values.reshape(2,2,3,4)
expected_slice = values[0,:,4:8]
print('wrong head 1:',wrong[0,1].tolist())
print('source features 4:8:',expected_slice.tolist())
""",
        "fix": """def split_heads(x, heads):
    # TODO: اول محور ویژگی را بشکنید، سپس زمان و Head را جابه‌جا کنید
    return None
""",
        "fix_check": """def test_repair():
    result = split_heads(values,2)
    if result is None:
        return False
    assert result.shape==(2,2,3,4)
    assert torch.equal(result[0,1],values[0,:,4:8])
    other = torch.arange(3*7*20).reshape(3,7,20)
    divided = split_heads(other,4)
    assert divided.shape==(3,4,7,5)
    assert divided[2,3,6,4]==other[2,6,19]
    assert torch.equal(divided.transpose(1,2).reshape(3,7,20),other)
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: split_heads')
""",
        "fix_solution": """def split_heads(x, heads):
    batch,length,channels = x.shape
    return x.reshape(batch,length,heads,channels//heads).transpose(1,2)
""",
        "connection": "دو قرارداد مستقیماً در MiniGPT.forward و CausalSelfAttention.forward استفاده می‌شوند. آزمون با اندازه‌های نامساوی کمک می‌کند خطا پشت برابری تصادفی محورهای مختلف پنهان نماند.",
        "takeaway": "کدام آزمونِ مقدار توانست خطایی را ببیند که آزمون Shape به‌تنهایی از دست می‌داد؟",
    },
    "61-one-batch": {
        "title": "اجازه داریم همین یک نمونه را حفظ کنیم",
        "goal": "یک آزمون کوچک یادگیری بسازید و نبودِ step را از نبودِ Gradient جدا کنید.",
        "prerequisite": "حلقهٔ آموزش، بررسی تغییر وزن و محدودیت آزمون تک‌نمونه‌ای.",
        "predict": "اگر Dropout صفر و وزن ثابت باشد، آیا تکرار forward روی همان نمونه Loss تازه‌ای می‌دهد؟",
        "setup": """import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
def fresh_model():
    torch.manual_seed(7)
    return MiniGPT(ModelConfig(12,8,16,2,2,0.0))
x,y = torch.tensor([[1,2,3,4,5,6]]),torch.tensor([[2,3,4,5,6,7]])
model = fresh_model()
print('same input losses:',model(x,y)[1].item(),model(x,y)[1].item())
""",
        "task": "overfit_history(model,x,y,steps) یک AdamW با lr=0.01 بسازد و یک Batch ثابت را steps بار آموزش دهد. فهرست Loss پیش از آموزش و پس از هر update را برگرداند؛ طول فهرست steps+1 است.",
        "starter": """def overfit_history(model, x, y, steps):
    # TODO: همان داده در هر گام، با update واقعی
    return None
""",
        "check": """def test_exercise():
    trial = fresh_model()
    result = overfit_history(trial,x,y,80)
    if result is None:
        return False
    assert len(result)==81
    assert result[-1]<result[0]/10
    assert all(torch.isfinite(torch.tensor(value)) for value in result)
    other = fresh_model()
    before = other.token_embedding.weight.detach().clone()
    zero = overfit_history(other,x,y,0)
    assert len(zero)==1 and torch.equal(before,other.token_embedding.weight)
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: overfit_history')
""",
        "solution": """def overfit_history(model, x, y, steps):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.01)
    history = [model(x,y)[1].item()]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(x,y)[1].backward()
        optimizer.step()
        with torch.no_grad():
            history.append(model(x,y)[1].item())
    return history
""",
        "vary": "فقط فعال‌بودن Dropout را عوض کنید، بدون هیچ optimizer.step. نوسان Loss را با یادگیری اشتباه نگیرید.",
        "vary_code": """for dropout in (0.0,0.5):
    torch.manual_seed(7)
    trial = MiniGPT(ModelConfig(12,8,16,2,2,dropout)).train()
    before = trial.token_embedding.weight.detach().clone()
    print(dropout,[trial(x,y)[1].item() for _ in range(4)])
    assert torch.equal(before,trial.token_embedding.weight)
""",
        "debug": "در نمونهٔ scalar، backward وزن را حرکت نمی‌دهد. scalar_step(parameter, optimizer) یک گام برای کم‌کردن (parameter-3)**2 اجرا کند و مقدار تازهٔ وزن را بدهد.",
        "bug_code": """w = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.SGD([w],lr=0.1)
((w-3)**2).backward()
print('gradient exists:',w.grad.item(),'but weight:',w.item())
""",
        "fix": """def scalar_step(parameter, optimizer):
    # TODO: محاسبهٔ مشتق و اعمال آن دو کار جدا هستند
    return None
""",
        "fix_check": """def test_repair():
    w = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([w],lr=0.1)
    result = scalar_step(w,optimizer)
    if result is None:
        return False
    assert abs(result-0.6)<1e-6
    assert abs(scalar_step(w,optimizer)-1.08)<1e-6
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: scalar_step')
""",
        "fix_solution": """def scalar_step(parameter, optimizer):
    optimizer.zero_grad(set_to_none=True)
    ((parameter-3)**2).backward()
    optimizer.step()
    return parameter.item()
""",
        "connection": "مدل و تنظیم مرجع با آزمایش overfit پروژه سازگارند. این موفقیت فقط سلامت یادگیری همان نمونه را نشان می‌دهد؛ نشت داده یا اشکال Validation را رد نمی‌کند.",
        "takeaway": "چرا موفقیت این آزمون اجازه نمی‌دهد بگوییم مدل روی متن تازه خوب کار می‌کند؟",
    },
    "62-journal": {
        "title": "گزارشی که از مدل جدا نمی‌شود",
        "goal": "یک نتیجهٔ واقعی را همراه تنظیمات مستقل و عدد قابل ذخیره ثبت کنید.",
        "prerequisite": "dict، JSON، Seed و تفاوت Tensor با عدد Python.",
        "predict": "اگر گزارش همان dict تنظیمات را نگه دارد، تغییر تنظیمات آزمایش بعدی با سابقهٔ قبلی چه می‌کند؟",
        "setup": """import copy
import json
import math
from dataclasses import asdict
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
config = ModelConfig(12,4,8,2,1,0.0)
model = MiniGPT(config)
x,y = torch.tensor([[1,2,3]]),torch.tensor([[2,3,4]])
measured_loss = model(x,y)[1]
settings = asdict(config)
print('actual untrained loss:',measured_loss.item())
""",
        "task": "make_record(settings, seed, loss) یک dict با کلیدهای settings، seed و loss بسازد. settings باید deepcopy شود و loss یک float باشد. این گزارش وزن‌ها یا سند کامل آزمایش نیست؛ فقط ثبت یک مشاهده است.",
        "starter": """def make_record(settings, seed, loss):
    # TODO: سابقه نباید با تنظیمات آزمایش بعدی تغییر کند
    return None
""",
        "check": """def test_exercise():
    source = {'model':{'width':8},'rate':0.001}
    result = make_record(source,7,measured_loss)
    if result is None:
        return False
    assert result['seed']==7 and isinstance(result['loss'],float)
    assert result['loss']==measured_loss.item()
    source['model']['width']=16
    assert result['settings']['model']['width']==8
    assert json.loads(json.dumps(result))==result
    assert make_record({},8,2.0)=={'settings':{},'seed':8,'loss':2.0}
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: make_record')
""",
        "solution": """def make_record(settings, seed, loss):
    value = loss.detach().item() if isinstance(loss,torch.Tensor) else float(loss)
    return {'settings':copy.deepcopy(settings),'seed':seed,'loss':float(value)}
""",
        "vary": "فقط Seed را تغییر دهید و هر نتیجه را همراه همان Seed چاپ کنید. ادعا نکنید مدل با Loss آغازین کمتر، پس از آموزش هم بهتر خواهد بود.",
        "vary_code": """for seed in (7,8):
    torch.manual_seed(seed)
    trial = MiniGPT(config)
    print(json.dumps({'seed':seed,'initial_loss':trial(x,y)[1].item()}))
""",
        "debug": "Tensor خام در JSON ذخیره نمی‌شود؛ NaN هم مشاهدهٔ معتبر نیست. metric_number(value) برای Tensor scalar یا عدد Python یک float متناهی برگرداند؛ برای مقدار نامتناهی ValueError بدهد.",
        "bug_code": """try:
    json.dumps({'loss':measured_loss})
except TypeError as error:
    print('expected serialization failure:',error)
else:
    raise AssertionError('a raw tensor is not JSON data')
""",
        "fix": """def metric_number(value):
    # TODO: فقط عدد متناهی و مستقل از Graph
    return None
""",
        "fix_check": """def test_repair():
    result = metric_number(measured_loss)
    if result is None:
        return False
    assert isinstance(result,float) and result==measured_loss.item()
    assert metric_number(2)==2.0
    for value in (float('nan'),float('inf')):
        try:
            metric_number(value)
        except ValueError:
            pass
        else:
            raise AssertionError('nonfinite measurement')
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: metric_number')
""",
        "fix_solution": """def metric_number(value):
    result = float(value.detach().item() if isinstance(value,torch.Tensor) else value)
    if not math.isfinite(result):
        raise ValueError('measurement must be finite')
    return result
""",
        "connection": "مشاهده از MiniGPT واقعی است. در دفتر آزمایش و learning-log.md باید علاوه بر این عدد، سؤال، داده، فرمان، زمان و محدودیت نتیجه را ثبت کنید؛ این dict جای آن گزارش کامل نیست.",
        "takeaway": "برای بازسازی این آزمایش روی رایانه‌ای دیگر، چه اطلاعاتی هنوز در record شما کم است؟",
    },
    "62b-lifecycle": {
        "title": "ورودی تازه یا وزن تازه؟",
        "goal": "تغییر Context را از آموزش وزن جدا کنید؛ نام یک محصول از این دو مشاهده نتیجه نمی‌شود.",
        "prerequisite": "forward، generate و update؛ تفاوت هدف آموزش و معماری.",
        "predict": "افزودن دستور به ورودی، فراخوانی model.train() و optimizer.step() کدام‌یک می‌توانند وزن را تغییر دهند؟",
        "setup": """import copy
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(8,6,8,2,1,0.0))
old_ids,new_ids = torch.tensor([[1,2,3]]),torch.tensor([[4,1,2,3]])
before = {name:value.detach().clone() for name,value in model.state_dict().items()}
model.generate(new_ids,2,greedy=True)
after_inference = {name:value.detach().clone() for name,value in model.state_dict().items()}
print('input lengths:',old_ids.shape[1],new_ids.shape[1])
print('Generating with a different input is not SFT or preference training.')
""",
        "task": "transition_report(before, after, old_ids, new_ids) یک dict با دو bool بدهد: weights_changed برای تغییر هر Tensor وضعیت مدل و input_changed برای تغییر شناسه‌های ورودی. دو state_dict کلیدهای برابر دارند.",
        "starter": """def transition_report(before, after, old_ids, new_ids):
    # TODO: دو نوع تغییر را مستقل بسنجید
    return None
""",
        "check": """def test_exercise():
    result = transition_report(before,after_inference,old_ids,new_ids)
    if result is None:
        return False
    assert result=={'weights_changed':False,'input_changed':True}
    assert transition_report(before,before,old_ids,old_ids)=={'weights_changed':False,'input_changed':False}
    changed = {name:value.clone() for name,value in before.items()}
    changed['language_model_head.weight'][0,0] += 1
    assert transition_report(before,changed,old_ids,old_ids)=={'weights_changed':True,'input_changed':False}
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: transition_report')
""",
        "solution": """def transition_report(before, after, old_ids, new_ids):
    return {'weights_changed':any(not torch.equal(before[name],after[name]) for name in before),
            'input_changed':not torch.equal(old_ids,new_ids)}
""",
        "vary": "فقط ورودی را عوض کنید؛ شکل متفاوت یا Logits متفاوت به معنی آموزش وزن نیست.",
        "vary_code": """model.eval()
with torch.no_grad():
    for ids in (old_ids,new_ids):
        print(ids.tolist(),model(ids)[0][0,-1].tolist())
assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())
""",
        "debug": "model.train() فقط حالت لایه‌ها را تنظیم می‌کند. update_from_loss(model, loss, rate) با SGD یک update واقعی انجام دهد و bool تغییر وزن را برگرداند. loss از forward همین مدل آمده است.",
        "bug_code": """trial = copy.deepcopy(model)
old = trial.token_embedding.weight.detach().clone()
trial.train()
print('train() alone changed weights:',not torch.equal(old,trial.token_embedding.weight))
""",
        "fix": """def update_from_loss(model, loss, rate):
    # TODO: حالت آموزش جای Gradient و update را نمی‌گیرد
    return None
""",
        "fix_check": """def test_repair():
    trial = copy.deepcopy(model)
    loss = trial(old_ids,torch.tensor([[2,3,4]]))[1]
    result = update_from_loss(trial,loss,0.1)
    if result is None:
        return False
    assert result is True
    unchanged = copy.deepcopy(model)
    zero_loss = unchanged(old_ids)[0].sum()*0
    assert update_from_loss(unchanged,zero_loss,0.1) is False
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: update_from_loss')
""",
        "fix_solution": """def update_from_loss(model, loss, rate):
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(),lr=rate)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return any(not torch.equal(old,new) for old,new in zip(before,model.parameters()))
""",
        "connection": "MiniGPT واقعی را مشاهده کردید، نه یک دستیار عمومی. Pretraining، SFT و آموزش مبتنی بر ترجیح می‌توانند وزن را تغییر دهند ولی داده و هدف متفاوت دارند. RAG هنگام بازیابی و تولید، Context را عوض می‌کند. این دفتر آن فرایندها را کامل پیاده نکرده است.",
        "takeaway": "چرا دو پرچم این گزارش برای تشخیص اینکه یک مدل واقعاً دستورپذیر، درست‌گو یا هم‌راستا شده کافی نیستند؟",
    },
    "63-scale": {
        "title": "چهار بایت به‌ازای وزن، کل حافظه نیست",
        "goal": "تخمین حافظهٔ وزن را از تخمین سادهٔ آموزش و مصرف واقعی Tensorها جدا کنید.",
        "prerequisite": "numel، dtype، Gradient و دو وضعیت Adam.",
        "predict": "یک میلیون وزن float32 تقریباً چهار میلیون بایت است؛ چرا آموزش به بیشتر از همین مقدار نیاز دارد؟",
        "setup": """import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(12,4,8,2,1,0.0))
optimizer = torch.optim.AdamW(model.parameters())
model(torch.tensor([[1,2,3]]),torch.tensor([[2,3,4]]))[1].backward()
optimizer.step()
parameter_count = sum(p.numel() for p in model.parameters())
optimizer_bytes = sum(value.numel()*value.element_size() for state in optimizer.state.values()
                      for value in state.values() if isinstance(value,torch.Tensor))
print('parameters:',parameter_count,'actual optimizer tensor bytes:',optimizer_bytes)
print('Activation memory and allocator overhead are not measured here.')
""",
        "task": "memory_estimate(parameters, bytes_per_number, training) یک تعداد بایت صحیح بدهد. برای وزن‌ها یک مجموعه و برای تخمین سادهٔ آموزش چهار مجموعهٔ هم‌اندازه در نظر بگیرید: وزن، Gradient و دو وضعیت Adam. این مدل تخمینی فرض می‌کند dtype هر چهار یکسان است.",
        "starter": """def memory_estimate(parameters, bytes_per_number, training):
    # TODO: تخمین خام، نه حافظهٔ اوج واقعی
    return None
""",
        "check": """def test_exercise():
    result = memory_estimate(1_000_000,4,False)
    if result is None:
        return False
    assert result==4_000_000
    assert memory_estimate(1_000_000,4,True)==16_000_000
    assert memory_estimate(100_000_000,4,False)==400_000_000
    assert memory_estimate(10,2,True)==80
    assert memory_estimate(0,4,True)==0
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: memory_estimate')
""",
        "solution": """def memory_estimate(parameters, bytes_per_number, training):
    return parameters*bytes_per_number*(4 if training else 1)
""",
        "vary": "فقط تعداد بایت هر وزن را در تخمین تغییر دهید. این محاسبه مدل را Quantize نمی‌کند و سرعت اجرا را هم اندازه نمی‌گیرد.",
        "vary_code": """for bytes_per_weight in (4,2,1):
    print(bytes_per_weight,parameter_count*bytes_per_weight,'estimated raw weight bytes')
""",
        "debug": "len(model.parameters()) تعداد ظرف‌هاست، نه تعداد وزن‌ها. model_weight_bytes(model) مجموع numel*element_size همهٔ Parameterها را بدهد؛ Bufferها و Activationها جزو این تابع نیستند.",
        "bug_code": """wrong = len(list(model.parameters()))*4
print('wrong bytes from parameter-object count:',wrong)
print('first weight shape:',tuple(next(model.parameters()).shape))
""",
        "fix": """def model_weight_bytes(model):
    # TODO: اندازهٔ هر Tensor و dtype واقعی آن
    return None
""",
        "fix_check": """def test_repair():
    result = model_weight_bytes(model)
    if result is None:
        return False
    assert result==parameter_count*4
    layer = torch.nn.Linear(3,2,bias=True).double()
    assert model_weight_bytes(layer)==(3*2+2)*8
    assert model_weight_bytes(torch.nn.Identity())==0
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: model_weight_bytes')
""",
        "fix_solution": """def model_weight_bytes(model):
    return sum(parameter.numel()*parameter.element_size() for parameter in model.parameters())
""",
        "connection": "وزن‌ها و وضعیت Optimizer از مدل واقعی‌اند؛ برآورد چهارمجموعه‌ای فقط یک مدل ساده است. Mixed precision، آموزش توزیع‌شده و FlashAttention در این دفتر اجرا نمی‌شوند.",
        "takeaway": "کدام هزینه‌ها هنوز در تخمین شما نیستند و چرا کم‌شدن بایت هر وزن الزاماً سرعت را بیشتر نمی‌کند؟",
    },
    "64-cache": {
        "title": "وقتی گذشته واقعاً ثابت مانده است",
        "goal": "خروجی آخرین Query را با K/V نگه‌داشته‌شده بازسازی کنید و مرز اعتبار Cache را ببینید.",
        "prerequisite": "Q/K/V با شکل (B,H,T,D)، حالت eval و موقعیت‌های ثابت.",
        "predict": "اگر پنجره از چپ بریده و موقعیت‌ها از صفر شماره‌گذاری شوند، آیا K/V قبلی هنوز همان محاسبه را نشان می‌دهند؟",
        "setup": """import math
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(7)
model = MiniGPT(ModelConfig(12,4,8,2,1,0.0)).eval()
prefix = torch.tensor([[1,2,3]])
extended = torch.tensor([[1,2,3,4]])
with torch.no_grad():
    old_trace,new_trace = {},{}
    model(prefix,trace=old_trace)
    model(extended,trace=new_trace)
old_attention = old_trace['layers'][0]['attention']
new_attention = new_trace['layers'][0]['attention']
torch.testing.assert_close(old_attention['k'],new_attention['k'][:,:,:3])
print('old/new key shapes:',tuple(old_attention['k'].shape),tuple(new_attention['k'].shape))
""",
        "task": "cached_last_attention(q_new,k_old,v_old,k_new,v_new) سه Tensor برگرداند: خروجی Query تازه، K کامل و V کامل. همه شکل (B,H,T,D) دارند و q_new/k_new/v_new فقط یک موقعیت دارند؛ الحاق روی محور زمان و مقیاس 1/sqrt(D) است. آخرین Query اجازهٔ دیدن همهٔ این موقعیت‌ها را دارد.",
        "starter": """def cached_last_attention(q_new, k_old, v_old, k_new, v_new):
    # TODO: فقط محاسبهٔ آخرین Query
    return None
""",
        "check": """def test_exercise():
    args = (new_attention['q'][:,:,-1:],old_attention['k'],old_attention['v'],
            new_attention['k'][:,:,-1:],new_attention['v'][:,:,-1:])
    result = cached_last_attention(*args)
    if result is None:
        return False
    output,keys,values = result
    torch.testing.assert_close(keys,new_attention['k'])
    torch.testing.assert_close(values,new_attention['v'])
    torch.testing.assert_close(output,new_attention['weighted_values'][:,:,-1:])
    query = torch.ones(1,1,1,2)
    empty = torch.empty(1,1,0,2)
    value = torch.tensor([[[[3.0,5.0]]]])
    one = cached_last_attention(query,empty,empty,query,value)
    torch.testing.assert_close(one[0],value)
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: cached_last_attention')
""",
        "solution": """def cached_last_attention(q_new, k_old, v_old, k_new, v_new):
    keys = torch.cat((k_old,k_new),dim=-2)
    values = torch.cat((v_old,v_new),dim=-2)
    scores = q_new@keys.transpose(-2,-1)/math.sqrt(q_new.shape[-1])
    return scores.softmax(-1)@values,keys,values
""",
        "vary": "فقط طول Prefix را از ۱ تا ۳ افزایش دهید؛ K/V ذخیره‌شده را بشمارید. این شمارش، اندازهٔ Cache است نه حافظهٔ کل مدل.",
        "vary_code": """with torch.no_grad():
    for length in (1,2,3):
        trace = {}
        model(extended[:,:length],trace=trace)
        attention = trace['layers'][0]['attention']
        print(length,'K+V elements:',attention['k'].numel()+attention['v'].numel())
""",
        "debug": "حذف قدیمی‌ترین K/V کافی نیست، چون شمارهٔ موقعیت بقیه هم عوض شده است. cache_reusable(old_ids,new_ids,limit) برای فهرست‌های ID و فرض وزن و حالت ثابت، فقط وقتی True بدهد که دقیقاً یک Token به Prefix بدون تغییر اضافه شده و طول از limit نگذشته باشد.",
        "bug_code": """with torch.no_grad():
    shifted = {}
    model(torch.tensor([[2,3,4,5]]),trace=shifted)
stale_keys = new_attention['k'][:,:,1:]
recomputed_keys = shifted['layers'][0]['attention']['k'][:,:,:3]
print('stale/recomputed key difference:',(stale_keys-recomputed_keys).abs().max().item())
""",
        "fix": """def cache_reusable(old_ids, new_ids, limit):
    # TODO: اعتبار Prefix و موقعیت‌ها، پیش از استفادهٔ دوباره
    return None
""",
        "fix_check": """def test_repair():
    result = cache_reusable([1,2],[1,2,3],4)
    if result is None:
        return False
    assert result is True
    assert cache_reusable([1,2],[1,4,3],4) is False
    assert cache_reusable([1,2,3,4],[1,2,3,4,5],4) is False
    assert cache_reusable([1,2],[1,2],4) is False
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: cache_reusable')
""",
        "fix_solution": """def cache_reusable(old_ids, new_ids, limit):
    return len(new_ids)==len(old_ids)+1 and len(new_ids)<=limit and new_ids[:-1]==old_ids
""",
        "connection": "Q/K/V از trace واقعی MiniGPT آمده‌اند؛ تابع شما فقط هستهٔ Attention آخرین موقعیت را بازسازی می‌کند، نه Cache کامل همهٔ لایه‌ها. generate پروژه همچنان کل پنجره را دوباره محاسبه می‌کند.",
        "takeaway": "برای تبدیل این تمرین به Cache کامل مدل، کدام وضعیت‌ها را باید برای هر Layer نگه دارید؟",
    },
    "65-sft": {
        "title": "درخواست را ببینید، پاسخ را نمره بدهید",
        "goal": "Loss پاسخ را بیرون API فعلی MiniGPT بسازید و مرز آن را با Causal mask جدا کنید.",
        "prerequisite": "شیفت یک‌خانه‌ای Target و Cross-Entropy با reduction='none'.",
        "predict": "در [q0,q1,sep,a0,a1]، آیا خروجی موقعیت sep باید در Loss پاسخ حساب شود؟",
        "setup": """import torch
from torch.nn import functional as F
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(31)
model = MiniGPT(ModelConfig(8,4,8,2,1,0.0))
x,targets = torch.tensor([[1,2,3,4]]),torch.tensor([[2,3,4,5]])
selected = torch.tensor([[False,False,True,True]])
logits,_ = model(x)
print('logits shape:',tuple(logits.shape))
print('target ID / counted as response:',list(zip(targets[0].tolist(),selected[0].tolist())))
print('all targets:',targets.numel(),'response targets:',selected.sum().item())
""",
        "task": "response_loss(logits,targets,mask) فقط Cross-Entropy هدف‌هایی با mask=True را میانگین بگیرد. Logits شکل (B,T,V)، دو ورودی دیگر شکل (B,T) دارند. Mask خالی از True باید ValueError بدهد؛ از Loss آمادهٔ model(x,targets) استفاده نکنید چون قبلاً میانگین گرفته شده است.",
        "starter": """def response_loss(logits, targets, mask):
    # TODO: اول Loss هر Target، سپس انتخاب و میانگین
    return None
""",
        "check": """def test_exercise():
    result = response_loss(logits,targets,selected)
    if result is None:
        return False
    expected_response = F.cross_entropy(logits[0,2:,:],targets[0,2:])
    torch.testing.assert_close(result,expected_response)
    changed = targets.clone()
    changed[0,:2] = torch.tensor([6,7])
    torch.testing.assert_close(result,response_loss(logits,changed,selected))
    expected_all = F.cross_entropy(logits[0],targets[0])
    torch.testing.assert_close(response_loss(logits,targets,torch.ones_like(selected)),expected_all)
    try:
        response_loss(logits,targets,torch.zeros_like(selected))
    except ValueError:
        pass
    else:
        raise AssertionError('no selected targets')
    model.zero_grad(set_to_none=True)
    fresh_logits,_ = model(x)
    response_loss(fresh_logits,targets,selected).backward()
    assert model.token_embedding.weight.grad[1].abs().sum()>0
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: response_loss')
""",
        "solution": """def response_loss(logits, targets, mask):
    if not mask.any():
        raise ValueError('select at least one target')
    losses = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='none')
    return losses[mask.reshape(-1)].mean()
""",
        "vary": "فقط تعداد موقعیت‌های درخواست را در یک ماسک نمونه بیشتر کنید؛ دو موقعیت پاسخ ثابت بمانند. تعداد کل موقعیت‌ها و تعداد Trueها چه تغییری می‌کنند؟ برای میانگین Loss پاسخ، کدام تعداد باید مخرج باشد؟ این آزمایش فقط شمارش موقعیت‌هاست، نه اجرای مدل با درخواست بلندتر.",
        "vary_code": """response_mask = torch.tensor([[False,False,True,True]])
for extra_prompt_positions in (0,3,8):
    prefix = torch.zeros((1,extra_prompt_positions),dtype=torch.bool)
    longer_mask = torch.cat((prefix,response_mask),dim=1)
    print('extra prompt positions:',extra_prompt_positions,
          'all positions:',longer_mask.numel(),
          'response positions:',torch.count_nonzero(longer_mask).item())
""",
        "debug": "ماسکِ بخش ورودی، نخستین Token پاسخ را از Loss جا می‌اندازد. target_response_mask(roles) برای نقش‌های کل دنباله، ماسک Targetهای شیفت‌یافته را بدهد؛ فقط نقش 'answer' انتخاب می‌شود.",
        "bug_code": """roles = ['prompt','prompt','separator','answer','answer']
wrong = [role=='answer' for role in roles[:-1]]
print('input-based mask:',wrong)
print('input/target roles:',list(zip(roles[:-1],roles[1:])))
""",
        "fix": """def target_response_mask(roles):
    # TODO: نقش هر Target، نه نقش ورودی همان موقعیت
    return None
""",
        "fix_check": """def test_repair():
    result = target_response_mask(roles)
    if result is None:
        return False
    assert result==[False,False,True,True]
    assert target_response_mask(['separator','answer'])==[True]
    assert target_response_mask(['prompt','separator'])==[False]
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: target_response_mask')
""",
        "fix_solution": """def target_response_mask(roles):
    return [role=='answer' for role in roles[1:]]
""",
        "connection": "forward واقعی برای Logits استفاده شد؛ Response mask به‌صورت آموزشی بیرون آن محاسبه شد. API فعلی Target منفی را رد می‌کند و SFT آماده ندارد. حذف Loss درخواست، مسیر Gradient از پاسخ به نمایش درخواست را حذف نمی‌کند.",
        "takeaway": "چرا حذف یک موقعیت از میانگین Loss با منع Attention به آن موقعیت فرق دارد؟",
    },
    "65b-lora": {
        "title": "اصلاح کوچک روی وزن ثابت",
        "goal": "شاخهٔ LoRA را بنویسید و ببینید کدام Parameter در گام اول Gradient می‌گیرد.",
        "prerequisite": "ضرب ماتریسی، Linear، Parameter ثابت و قابل آموزش.",
        "predict": "اگر A و B هر دو صفر باشند، آیا این شاخه از صفر حرکت می‌کند؟",
        "setup": """import torch
from torch import nn
from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
torch.set_num_threads(1)
torch.manual_seed(12)
project_model = MiniGPT(ModelConfig(8,4,12,3,1,0.0))
W = project_model.language_model_head.weight.detach().clone()
A = nn.Parameter(torch.randn(2,12)*0.01)
B = nn.Parameter(torch.zeros(8,2))
x = torch.randn(3,12)
print('base weight shape:',tuple(W.shape),'adapter scalars:',A.numel()+B.numel())
print('base output shape:',tuple((x@W.T).shape))
""",
        "task": "lora_forward(x,W,A,B,alpha) خروجی xWᵀ + (alpha/r)(xAᵀ)Bᵀ را بدهد؛ r تعداد سطرهای A است. W شکل (M,C)، A شکل (r,C)، B شکل (M,r) دارد. هیچ Tensor را درجا تغییر ندهید.",
        "starter": """def lora_forward(x, W, A, B, alpha):
    # TODO: شاخهٔ کم‌رتبه به خروجی پایه اضافه می‌شود
    return None
""",
        "check": """def test_exercise():
    result = lora_forward(x,W,A,B,2.0)
    if result is None:
        return False
    torch.testing.assert_close(result,x@W.T)
    trial_b = torch.ones_like(B)
    torch.testing.assert_close(lora_forward(x,W,A,trial_b,4.0),x@(W+2.0*(trial_b@A)).T)
    target = torch.ones_like(result)
    ((result-target)**2).mean().backward()
    assert W.grad is None
    assert A.grad is not None and torch.count_nonzero(A.grad)==0
    assert B.grad is not None and B.grad.norm()>0
    assert A.numel()+B.numel()==40
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: lora_forward')
""",
        "solution": """def lora_forward(x, W, A, B, alpha):
    return x@W.T+(alpha/A.shape[0])*(x@A.T)@B.T
""",
        "vary": "فقط Rank را عوض کنید و پارامترهای شاخه را بشمارید. کاهش Rank به‌تنهایی کیفیت Fine-Tuning را پیش‌بینی نمی‌کند.",
        "vary_code": """for rank in (1,2,4):
    a = torch.zeros(rank,12)
    b = torch.zeros(8,rank)
    print(rank,'adapter:',a.numel()+b.numel(),'base still needed:',W.numel())
""",
        "debug": "صفرکردن هر دو عامل، Gradient هرکدام را در عامل صفر دیگر ضرب می‌کند. initialize_adapter(out_features,in_features,rank) دو nn.Parameter بسازد: A با مقدار تصادفی کوچک و B صفر.",
        "bug_code": """wrong_a = nn.Parameter(torch.zeros(2,12))
wrong_b = nn.Parameter(torch.zeros(8,2))
((x@W.T+(x@wrong_a.T)@wrong_b.T-1)**2).mean().backward()
print('both-zero gradient norms:',wrong_a.grad.norm().item(),wrong_b.grad.norm().item())
""",
        "fix": """def initialize_adapter(out_features, in_features, rank):
    # TODO: خروجی آغازین پایه بماند، اما شاخه بتواند یاد بگیرد
    return None
""",
        "fix_check": """def test_repair():
    result = initialize_adapter(8,12,2)
    if result is None:
        return False
    a,b = result
    assert isinstance(a,nn.Parameter) and isinstance(b,nn.Parameter)
    assert a.shape==(2,12) and b.shape==(8,2)
    assert torch.count_nonzero(a)>0 and torch.count_nonzero(b)==0
    a2,b2 = initialize_adapter(3,5,1)
    assert a2.shape==(1,5) and b2.shape==(3,1)
    assert a2.requires_grad and b2.requires_grad
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: initialize_adapter')
""",
        "fix_solution": """def initialize_adapter(out_features, in_features, rank):
    return (nn.Parameter(torch.randn(rank,in_features)*0.01),
            nn.Parameter(torch.zeros(out_features,rank)))
""",
        "connection": "W یک کپی ثابت از سر خروجی MiniGPT است؛ شاخهٔ آموزشی به خود مدل نصب نشده است. LoRA روش انتخاب Parameterهای قابل آموزش است و می‌تواند کنار هدف SFT به کار رود؛ هدف آموزشی جداگانه‌ای نیست.",
        "takeaway": "چرا ۴۰ پارامتر قابل آموزش به معنی ۴۰ پارامتر لازم برای اجرای مدل نیست؟",
    },
    "66-preference": {
        "title": "دو پاسخ، یک ترجیح قابل آزمایش",
        "goal": "اثر ترجیح را در یک مسئلهٔ دوپاسخی کوچک ببینید؛ آن را با سنجش حقیقت اشتباه نگیرید.",
        "prerequisite": "Log probability، Softmax، Gradient و مدل مرجع ثابت.",
        "predict": "اگر دادهٔ ترجیح پاسخ غلط را برنده اعلام کند، آیا تابع هدف خودش این اشتباه را کشف می‌کند؟",
        "setup": """import math
import torch
from torch.nn import functional as F
torch.set_num_threads(1)
responses = ['پاسخ درست و کوتاه','پاسخ روان اما غلط']
reference_logits = torch.tensor([0.0,0.0])
policy_logits = torch.tensor([0.0,0.0],requires_grad=True)
print(list(zip(responses,policy_logits.softmax(-1).detach().tolist())))
print('Each outcome here is a whole answer, not one token from MiniGPT.')
""",
        "task": "preference_loss(policy_logits,reference_logits,chosen,beta) را برای دقیقاً دو پاسخ بنویسید. chosen اندیس پاسخ ترجیح‌داده‌شده است و دیگری 1-chosen. log_softmax هر دو توزیع را بگیرید. margin اختلاف log-prob برنده و بازنده در Policy، منهای همین اختلاف در مرجع ثابت است. Loss برابر softplus(-beta*margin) است؛ softplus(z)=log(1+exp(z)) و نسخهٔ PyTorch پایدارتر است. این نمونهٔ محدودِ هدف DPO است، نه فرایند کامل آموزش مدل زبان.",
        "starter": """def preference_loss(policy_logits, reference_logits, chosen, beta):
    # TODO: مقایسهٔ نسبی با مرجع ثابت
    return None
""",
        "check": """def test_exercise():
    result = preference_loss(policy_logits,reference_logits,0,0.5)
    if result is None:
        return False
    assert abs(result.item()-math.log(2))<1e-6
    favored = torch.tensor([2.0,0.0])
    assert preference_loss(favored,reference_logits,0,0.5)<result
    assert preference_loss(favored,reference_logits,1,0.5)>result
    torch.testing.assert_close(preference_loss(favored,favored,0,0.5),torch.tensor(math.log(2)))
    candidate = torch.tensor([0.0,0.0],requires_grad=True)
    ref = torch.tensor([0.0,0.0],requires_grad=True)
    preference_loss(candidate,ref,0,0.5).backward()
    assert candidate.grad[0]<0 and candidate.grad[1]>0
    assert ref.grad is None
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: preference_loss')
""",
        "solution": """def preference_loss(policy_logits, reference_logits, chosen, beta):
    rejected = 1-chosen
    policy = policy_logits.log_softmax(-1)
    reference = reference_logits.detach().log_softmax(-1)
    margin = (policy[chosen]-policy[rejected])-(reference[chosen]-reference[rejected])
    return F.softplus(-beta*margin)
""",
        "vary": "فقط برچسب ترجیح را عوض کنید؛ دو پاسخ و Policy آغازین ثابت‌اند. فرمول زیر صرفاً یک گام عددی مستقل را نشان می‌دهد، نه آزمون درستی پاسخ.",
        "vary_code": """for chosen in (0,1):
    scores = torch.tensor([0.0,0.0],requires_grad=True)
    logp = scores.log_softmax(-1)
    margin = logp[chosen]-logp[1-chosen]
    F.softplus(-0.5*margin).backward()
    with torch.no_grad():
        scores -= 0.2*scores.grad
    print('preferred:',chosen,'new probabilities:',scores.softmax(-1).detach().tolist())
""",
        "debug": "اگر ترتیب برنده و بازنده را برعکس کنید، کم‌کردن Loss پاسخ نامطلوب را تقویت می‌کند. preference_margin(policy_logps,reference_logps,chosen,rejected) اختلاف درست نسبت به مرجع را برگرداند؛ ورودی‌ها از قبل log-prob هستند.",
        "bug_code": """scores = torch.tensor([0.0,0.0],requires_grad=True)
logp = scores.log_softmax(-1)
wrong_margin = logp[1]-logp[0]  # The data actually prefers answer 0.
F.softplus(-wrong_margin).backward()
with torch.no_grad():
    scores -= 0.1*scores.grad
print('wrong-sign update:',scores.softmax(-1).tolist())
""",
        "fix": """def preference_margin(policy_logps, reference_logps, chosen, rejected):
    # TODO: برنده منهای بازنده، در هر دو مدل
    return None
""",
        "fix_check": """def test_repair():
    p = torch.tensor([0.8,0.2]).log()
    r = torch.tensor([0.5,0.5]).log()
    result = preference_margin(p,r,0,1)
    if result is None:
        return False
    torch.testing.assert_close(result,torch.tensor(math.log(4)))
    torch.testing.assert_close(preference_margin(p,r,1,0),torch.tensor(-math.log(4)))
    torch.testing.assert_close(preference_margin(r,r,0,1),torch.tensor(0.0))
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: preference_margin')
""",
        "fix_solution": """def preference_margin(policy_logps, reference_logps, chosen, rejected):
    return (policy_logps[chosen]-policy_logps[rejected])-(reference_logps[chosen]-reference_logps[rejected])
""",
        "connection": "این آزمایش دوپاسخی از صورت هدف در <a href='https://arxiv.org/abs/2305.18290'>مقالهٔ DPO</a> استفاده می‌کند. در مدل زبان، log-prob پاسخ از Tokenهای پاسخ به دست می‌آید. MiniGPT فعلی این فرایند، مدل پاداش یا حلقهٔ RLHF ندارد؛ DPO و RLHF دو ایستگاه اجباری پشت‌سرهم نیستند.",
        "takeaway": "اگر ترجیح‌ها معیار بدی داشته باشند، کاهش این Loss چه چیزی را بهتر می‌کند و چه چیزی را تضمین نمی‌کند؟",
    },
    "67-rag": {
        "title": "سند مرتبط را پیدا کنید، پاسخ را جعل نکنید",
        "goal": "بازیابی کوچک با ارجاع بسازید و وقتی شاهدی ندارید، نبودِ آن را صریح گزارش کنید.",
        "prerequisite": "رشته، مجموعهٔ واژه‌ها و جدایی Retrieval از Generation.",
        "predict": "اگر هیچ واژه‌ای مشترک نباشد، آیا نخستین سندِ فهرست می‌تواند شاهد مناسبی برای پاسخ باشد؟",
        "setup": """documents = [
    {'id':'d1','text':'کتابخانه شنبه باز است'},
    {'id':'d2','text':'آزمایشگاه یکشنبه تعطیل است'},
    {'id':'d3','text':'زمان کلاس دوشنبه است'},
]
question = 'کتابخانه شنبه'
words = set(question.split())
print([(doc['id'],len(words & set(doc['text'].split()))) for doc in documents])
print('This is lexical retrieval only; no language model answers are generated.')
""",
        "task": "retrieve(documents, question) سند با بیشترین تعداد واژهٔ متمایز مشترک را برگرداند؛ در تساوی نخستین سند انتخاب شود. اگر اسناد خالی‌اند یا بهترین امتیاز صفر است None برگردانید. رشته‌ها در این تمرین با split ساده شکسته می‌شوند.",
        "starter": """def retrieve(documents, question):
    # TODO: نبودِ شاهد نیز یک نتیجه است
    return None
""",
        "check": """def test_exercise():
    result = retrieve(documents,question)
    if result is None:
        return False
    assert result['id']=='d1'
    assert retrieve(documents,'آزمایشگاه تعطیل')['id']=='d2'
    assert retrieve(documents,'زلزله') is None
    assert retrieve([],'کتابخانه') is None
    assert retrieve(documents,'است')['id']=='d1'
    assert retrieve(documents,'کتابخانه کتابخانه شنبه')['id']=='d1'
    return True
exercise_complete = test_exercise()
print('PASS' if exercise_complete else 'INCOMPLETE: retrieve')
""",
        "solution": """def retrieve(documents, question):
    words = set(question.split())
    if not documents:
        return None
    best = max(documents,key=lambda doc:len(words & set(doc['text'].split())))
    return best if words & set(best['text'].split()) else None
""",
        "vary": "فقط عبارت پرسش را عوض کنید. جست‌وجوی واژه‌ای نمی‌فهمد «کتابخانه» و «مرکز امانت کتاب» ممکن است به یک جا اشاره کنند.",
        "vary_code": """for query in ('کتابخانه شنبه','مرکز امانت کتاب'):
    scores = [(doc['id'],len(set(query.split()) & set(doc['text'].split()))) for doc in documents]
    print(query,scores)
""",
        "debug": "بیشترین امتیازِ صفر هم یک برندهٔ ظاهری دارد؛ نباید برای آن ارجاع معتبر بسازیم. evidence_record(document) اگر ورودی None است {'status':'insufficient','source':None} وگرنه {'status':'retrieved','source':id,'text':text} بدهد. retrieved به معنی پاسخ تأییدشده نیست.",
        "bug_code": """unknown_question = 'زلزله'
wrong = max(documents,key=lambda doc:len(set(unknown_question.split()) & set(doc['text'].split())))
print('false evidence for unrelated query:',wrong)
""",
        "fix": """def evidence_record(document):
    # TODO: سند بازیابی‌شده را با پاسخ تأییدشده اشتباه نگیرید
    return None
""",
        "fix_check": """def test_repair():
    result = evidence_record(None)
    if result is None:
        return False
    assert result=={'status':'insufficient','source':None}
    assert evidence_record(documents[0])=={'status':'retrieved','source':'d1','text':documents[0]['text']}
    assert evidence_record(documents[2])['source']=='d3'
    return True
repair_complete = test_repair()
print('PASS' if repair_complete else 'INCOMPLETE: evidence_record')
""",
        "fix_solution": """def evidence_record(document):
    if document is None:
        return {'status':'insufficient','source':None}
    return {'status':'retrieved','source':document['id'],'text':document['text']}
""",
        "connection": "این قطعه مرحلهٔ Retrieval است، نه دستیار اسناد. افزودن سند به Prompt وزن MiniGPT را آموزش نمی‌دهد؛ مدل کوچک ما توان پاسخ‌گویی مستند را تضمین نمی‌کند. سند بیرونی نیز دستور معتبر سامانه نیست.",
        "takeaway": "برای ادعای درست‌بودن پاسخ، علاوه بر بازیابی سند باید ارتباط، تازگی، تناقض و پشتیبانی هر ادعا را چگونه بررسی کنید؟",
    },
}
