"""Authored, lesson-specific laboratory exercises for parts one through four.

Solutions are author-only inputs to verification, never student notebook cells.
"""


def _check(first, assertions, repair=False):
    name = 'test_repair' if repair else 'test_exercise'
    flag = 'repair_complete' if repair else 'exercise_complete'
    lines = [f'def {name}():', f'    result = {first}',
             '    if result is None:', '        return False']
    lines.extend('    ' + line for line in assertions.strip().splitlines())
    lines += ['    return True', '', f'{flag} = {name}()',
              f"print('PASS' if {flag} else 'INCOMPLETE: implement the TODO and rerun')"]
    return '\n'.join(lines)


EXERCISES = {
    '01-model': dict(
        title='زمان خواندن روی کارت مقاله',
        goal='از تعداد بندهای یک مقاله، زمان بسازید و سه ضریب را با همان داده مقایسه کنید.',
        prerequisite='فقط تابع، فهرست و حلقهٔ Python؛ هیچ کتابخانهٔ عددی لازم نیست.',
        predict='مقالهٔ تازه چهار بند دارد. ضریب‌های ۱، ۲ و ۳ چه زمان‌هایی می‌دهند؟ کدام زمان مشاهده شده و کدام فقط حدس ماست؟',
        setup='''articles = [(1, 2), (2, 4), (3, 6)]  # paragraphs, measured minutes; synthetic data
print("Paragraphs -> measured minutes:", articles)
''',
        task='تابع report(data, w) را بنویسید. data فهرست جفت‌های تعداد بند و زمان واقعی است. خروجی یک tuple شامل فهرست پیش‌بینی‌های w*x و میانگین مربع اختلاف با زمان واقعی باشد. دادهٔ خالی را با ValueError رد کنید.',
        starter='''def report(data, w):
    # TODO: return (predictions, mean_squared_error)
    return None
''',
        check=_check('report(articles, 2)', '''assert result == ([2, 4, 6], 0)
predictions, loss = report(articles, 1)
assert predictions == [1, 2, 3] and abs(loss - 14/3) < 1e-12
assert report([(4, 9)], 2) == ([8], 1)
try:
    report([], 2)
except ValueError:
    pass
else:
    raise AssertionError("Empty data must be rejected")'''),
        solution='''def report(data, w):
    if not data:
        raise ValueError("Data must not be empty")
    predictions = [w*x for x, target in data]
    loss = sum((prediction-target)**2 for prediction, (_, target) in zip(predictions, data))/len(data)
    return predictions, loss
''',
        vary='فقط زمان ثبت‌شدهٔ مقالهٔ سه‌بندی را از ۶ به ۷ تغییر دهید؛ ضریب را ۲ نگه دارید. پیش‌بینی باید ثابت بماند، اما خطا نه.',
        vary_code='''changed = [(1, 2), (2, 4), (3, 7)]
fixed_predictions = [2*x for x, _ in changed]
print("Unchanged predictions:", fixed_predictions)
print("Squared errors:", [(p-y)**2 for p, (_, y) in zip(fixed_predictions, changed)])
''',
        debug='دو خطای با علامت مخالف داریم. چرا میانگین خودِ اختلاف‌ها، کیفیت پیش‌بینی را درست نشان نمی‌دهد؟ نسخهٔ خراب را ببینید و تابع squared_error_mean را اصلاح کنید.',
        bug_code='''errors = [-2, 2]
wrong_score = sum(errors)/len(errors)
print("Broken score:", wrong_score, "Errors:", errors)
assert wrong_score == 0 and any(error != 0 for error in errors)
''',
        fix='''def squared_error_mean(errors):
    # TODO: return the mean of squared errors
    return None
''',
        fix_check=_check('squared_error_mean([-2, 2])', 'assert result == 4\nassert squared_error_mean([0, 3]) == 4.5', True),
        fix_solution='''def squared_error_mean(errors):
    return sum(error**2 for error in errors)/len(errors)
''',
        connection='در mini_gpt/train.py نیز پیش‌بینی و هدف جدا هستند. فعلاً آن فایل را اجرا نمی‌کنیم؛ اینجا همان جدایی را با یک ضریب و زمانِ دقیقه‌ای قابل مشاهده کردیم.',
        takeaway='چرا تغییر پاسخ واقعی، بدون تغییر ضریب، نباید پیش‌بینی را عوض کند؟'),

    '01-learning': dict(
        title='انتخاب ضریب با داده، نه با حدس',
        goal='جست‌وجوی Parameter را از اجرای مدلِ ثابت جدا کنید.',
        prerequisite='درس 01-model و محاسبهٔ میانگین مربع خطا با Python.',
        predict='اگر فقط ضریب‌های ۰، ۱ و ۳ را امتحان کنیم، آیا می‌توانیم ضریب ۲ را پیدا کنیم؟ اگر دو گزینه هم‌خطا باشند، کدام را انتخاب می‌کنید؟',
        setup='''data = [(1, 2), (2, 4), (3, 6)]
candidates = [0, 0.5, 1, 1.5, 2, 2.5]
print("Observed examples:", data, "Allowed parameters:", candidates)
''',
        task='تابع fit(data, candidates) بهترین ضریب را بر اساس میانگین مربع خطا برگرداند. در تساوی، نخستین گزینهٔ فهرست را نگه دارید. تابع نباید data یا candidates را تغییر دهد؛ هر دو ورودی در این تمرین غیرخالی‌اند.',
        starter='''def fit(data, candidates):
    # TODO: return the best candidate; break ties by input order
    return None
''',
        check=_check('fit(data, candidates)', '''assert result == 2
assert fit([(1, 2)], [1, 3]) == 1
assert fit([(1, 2)], [3, 1]) == 3
assert fit([(1, 3), (2, 6)], candidates) == 2.5
assert data == [(1, 2), (2, 4), (3, 6)]
assert candidates == [0, 0.5, 1, 1.5, 2, 2.5]'''),
        solution='''def fit(data, candidates):
    return min(candidates, key=lambda w: sum((w*x-y)**2 for x, y in data)/len(data))
''',
        vary='فقط شمارهٔ دو موضوع متن را عوض کنید. آیا پاسخ «این دو موضوع یکسان‌اند؟» تغییر می‌کند؟ فاصلهٔ عددی شناسه‌ها را هم چاپ کنید تا تفاوت نام و مقدار را ببینید.',
        vary_code='''topics = ["science", "sport", "science"]
for mapping in [{"science": 2, "sport": 7}, {"science": 100, "sport": 1}]:
    ids = [mapping[topic] for topic in topics]
    print(ids, "same topic:", ids[0] == ids[2], "ID distance:", abs(ids[0]-ids[1]))
''',
        debug='برای ساختن هدف از خود متن، جفت‌ها باید همسایه باشند. نسخهٔ خراب هر حرف را با خودش جفت می‌کند. تابع next_pairs را برای دنبالهٔ دلخواه اصلاح کنید.',
        bug_code='''text = "بابا"
wrong_pairs = list(zip(text, text))
print("Broken input/target pairs:", wrong_pairs)
assert all(current == target for current, target in wrong_pairs)
''',
        fix='''def next_pairs(sequence):
    # TODO: return adjacent (current, next) pairs
    return None
''',
        fix_check=_check('next_pairs("بابا")', 'assert result == [("ب", "ا"), ("ا", "ب"), ("ب", "ا")]\nassert next_pairs([1, 2, 3]) == [(1, 2), (2, 3)]\nassert next_pairs("x") == []', True),
        fix_solution='''def next_pairs(sequence):
    return list(zip(sequence, sequence[1:]))
''',
        connection='mini_gpt/stages/v0.py همین جفت‌های مجاور را می‌شمارد. بعداً mini_gpt/dataset.py هدف‌های بیشتری را در یک پنجره کنار هم می‌گذارد؛ هدف همچنان از خود متن می‌آید.',
        takeaway='در جست‌وجوی شما چه چیزی از داده انتخاب شد و چه چیزهایی را هنوز خودتان از پیش تعیین کرده بودید؟'),

    '02-token': dict(
        title='یک متن، دو نگاشت و یک آزمون رفت‌وبرگشت',
        goal='نگاشت قابل برگشت نویسه و شناسه بسازید و خرابی ترتیب واژگان را پیدا کنید.',
        prerequisite='درس‌های 01-model و 01-learning؛ دیکشنری و فهرست در Python.',
        predict='اگر فهرست واژگان را پس از Encoding برعکس کنیم ولی IDها را ثابت بگذاریم، طول متن بازگشتی تغییر می‌کند یا محتوا؟',
        setup='''text = "سلام hello!"
vocabulary = sorted(set(text))
to_id = {char: index for index, char in enumerate(vocabulary)}
print("Vocabulary:", list(enumerate(vocabulary)))
''',
        task='تابع encode(value, mapping) فهرست شناسه‌ها و decode(ids, vocabulary) رشته را برگرداند. نویسهٔ ناشناخته باید همان KeyError طبیعی دیکشنری را بدهد؛ فعلاً نشانهٔ ناشناخته اضافه نکنید.',
        starter='''def encode(value, mapping):
    # TODO
    return None

def decode(ids, vocabulary):
    # TODO
    return None
''',
        check=_check('encode(text, to_id)', '''if decode([], vocabulary) is None:
    return False
assert decode(result, vocabulary) == text
assert encode("", to_id) == [] and decode([], vocabulary) == ""
assert encode("abba", {"a": 5, "b": 9}) == [5, 9, 9, 5]
try:
    encode("ژ", to_id)
except KeyError:
    pass
else:
    raise AssertionError("An unknown character must fail")'''),
        solution='''def encode(value, mapping):
    return [mapping[char] for char in value]

def decode(ids, vocabulary):
    return "".join(vocabulary[index] for index in ids)
''',
        vary='فقط فاصله را با نیم‌فاصله جایگزین کنید. با دیدن repr و تعداد نویسه‌ها توضیح دهید چرا شباهت دیداری، یکسان‌بودن رشته نیست.',
        vary_code='''for value in ["می روم", "می‌روم"]:
    print(repr(value), len(value), [ord(char) for char in value])
''',
        debug='جدول Encoding و فهرست Decoding باید یک ترتیب داشته باشند. تابع mapping_for را طوری بنویسید که ترتیب فهرست ورودی را حفظ کند، نه اینکه مستقلاً آن را مرتب کند.',
        bug_code='''saved_vocabulary = ["ب", "ا"]
wrong_mapping = {char: i for i, char in enumerate(sorted(saved_vocabulary))}
ids = [wrong_mapping[char] for char in "با"]
wrong_text = "".join(saved_vocabulary[i] for i in ids)
print("Original: با; broken round trip:", wrong_text)
assert wrong_text != "با"
''',
        fix='''def mapping_for(vocabulary):
    # TODO: keep the supplied order
    return None
''',
        fix_check=_check('mapping_for(["ب", "ا"])', 'assert result == {"ب": 0, "ا": 1}\nassert mapping_for(["z", "a", "q"]) == {"z": 0, "a": 1, "q": 2}', True),
        fix_solution='''def mapping_for(vocabulary):
    return {char: index for index, char in enumerate(vocabulary)}
''',
        connection='این نسخهٔ کوچک را با mini_gpt/tokenizer.py اشتباه نگیرید: آن نسخه بعداً Unknown token و اعتبارسنجی فایل را اضافه می‌کند. اصل حفظ ترتیب در هر دو یکی است.',
        takeaway='کدام آزمون نشان می‌دهد IDها به همان نویسهٔ قبلی اشاره می‌کنند، نه فقط به یک نویسهٔ مجاز؟'),

    '03-counts': dict(
        title='حافظهٔ یک‌نویسه‌ای را آزمایش کنید',
        goal='جدول انتقال را خودتان بسازید و با نسخهٔ واقعی شمارشی مقایسه کنید.',
        prerequisite='درس 02-token و دیکشنری؛ Sampling و Seed را در درس جاری بخوانید.',
        predict='آیا دو متن آغازین متفاوت که هر دو به «م» ختم می‌شوند، با Seed یکسان ادامه‌های متفاوتی می‌گیرند؟',
        setup='''from collections import Counter, defaultdict
from mini_gpt.stages.v0 import transition_counts, generate
ids = [1, 0, 1, 0]
print("Input IDs:", ids)
''',
        task='تابع count_pairs(ids) دیکشنری current → دیکشنری next → count برگرداند. فقط همسایه‌های واقعی دنباله را بشمارید؛ میان آخر و اول پیوند تازه نسازید.',
        starter='''def count_pairs(ids):
    # TODO: count adjacent pairs
    return None
''',
        check=_check('count_pairs(ids)', '''assert result == {1: {0: 2}, 0: {1: 1}}
assert count_pairs([7, 7, 7]) == {7: {7: 2}}
assert count_pairs([1]) == {}
assert result == transition_counts(ids)'''),
        solution='''def count_pairs(ids):
    counts = {}
    for current, following in zip(ids, ids[1:]):
        row = counts.setdefault(current, {})
        row[following] = row.get(following, 0) + 1
    return counts
''',
        vary='فقط ابتدای Prompt را عوض کنید و آخرین ID را ثابت نگه دارید. نسخهٔ واقعی v0 را با Seed ثابت صدا بزنید؛ فقط بخش تازهٔ خروجی را مقایسه کنید.',
        vary_code='''counts = transition_counts(ids)
first = generate(counts, [0, 1], 12, 2, seed=42)
second = generate(counts, [1, 1, 1], 12, 2, seed=42)
print(first[2:], second[3:])
assert first[2:] == second[3:]
''',
        debug='جمع‌زدن شمارش تمام سطرها، Context را از بین می‌برد. تابع next_weights فقط سطر آخرین ID را بخواند و برای هر ID مجاز یک واحد هموارسازی اضافه کند.',
        bug_code='''rows = {0: {0: 8}, 1: {1: 8}}
wrong = [sum(row.get(i, 0) for row in rows.values())+1 for i in range(2)]
print("Broken weights for either context:", wrong)
assert wrong == [9, 9]
''',
        fix='''def next_weights(counts, current, vocabulary_size):
    # TODO: use only the current row, with add-one smoothing
    return None
''',
        fix_check=_check('next_weights(rows, 0, 2)', 'assert result == [9, 1]\nassert next_weights(rows, 1, 2) == [1, 9]\nassert next_weights(rows, 9, 2) == [1, 1]', True),
        fix_solution='''def next_weights(counts, current, vocabulary_size):
    row = counts.get(current, {})
    return [row.get(index, 0)+1 for index in range(vocabulary_size)]
''',
        connection='count_pairs را مستقیماً با transition_counts واقعی مقایسه کردیم و برای تولید از generate در mini_gpt/stages/v0.py استفاده کردیم؛ هیچ شبکه‌ای در این آزمایش نیست.',
        takeaway='با کدام شاهد نشان دادید محدودیت مدل در اطلاعات ورودی آن است، نه در Seed نامناسب؟'),

    '04-splits': dict(
        title='یک امتیاز خوب با دادهٔ لو‌رفته',
        goal='جداسازی سندها و اثر تکرار را پیش از آموزش شبکه قابل مشاهده کنید.',
        prerequisite='درس 03-counts؛ set، برش فهرست و نسبت دو تعداد.',
        predict='یک برنامه فقط جواب متن‌های دیده‌شده را حفظ کرده است. اگر دو متن ارزیابی را از آموزش کپی کنیم، امتیازش چه می‌شود؟ آیا قابلیت تازه‌ای یاد گرفته است؟',
        setup='''documents = [f"article-{i}" for i in range(10)]
answers = {document: i % 2 for i, document in enumerate(documents)}
print(documents)
''',
        task='تابع split_documents(items) سه برش ۶، ۲ و ۲تایی برگرداند. این تمرین دقیقاً ده سندِ یکتا دارد. تابع overlap(left, right) مجموعهٔ سندهای مشترک را برگرداند.',
        starter='''def split_documents(items):
    # TODO: return train, validation, test
    return None

def overlap(left, right):
    # TODO
    return None
''',
        check=_check('split_documents(documents)', '''if overlap([], []) is None:
    return False
train, valid, test = result
assert [len(train), len(valid), len(test)] == [6, 2, 2]
assert train + valid + test == documents
assert overlap(train, valid) == set() and overlap(train, test) == set()
assert overlap(["a", "b"], ["b", "c"]) == {"b"}'''),
        solution='''def split_documents(items):
    return items[:6], items[6:8], items[8:]

def overlap(left, right):
    return set(left) & set(right)
''',
        vary='فقط دو سند ارزیابی را با دو سند آموزش جایگزین کنید. مدلِ حفظ‌کننده و پاسخ‌ها ثابت‌اند. این مثال عمداً ساده است؛ ادعا نمی‌کند هر نشت داده همیشه همین مقدار به امتیاز اضافه می‌کند.',
        vary_code='''memory = {name: answers[name] for name in documents[:6]}
for evaluation in [documents[6:8], documents[:2]]:
    score = sum(memory.get(name) == answers[name] for name in evaluation)/len(evaluation)
    print("Evaluation:", evaluation, "Exact-match score:", score)
''',
        debug='دو فهرست، شیءهای متفاوت‌اند ولی ممکن است محتوای مشترک داشته باشند. بررسی is not هیچ نشت محتوایی را رد نمی‌کند. تابع assert_disjoint باید اشتراک محتوا را رد کند و در حالت سالم True بدهد.',
        bug_code='''training = ["first", "second"]
validation = ["second", "third"]
print("Broken independence check:", training is not validation)
assert training is not validation and "second" in set(training) & set(validation)
''',
        fix='''def assert_disjoint(train, validation):
    # TODO: raise ValueError on shared content; otherwise return True
    return None
''',
        fix_check=_check('assert_disjoint(["a"], ["b"])', '''assert result is True
try:
    assert_disjoint(training, validation)
except ValueError:
    pass
else:
    raise AssertionError("Shared document content must be rejected")''', True),
        fix_solution='''def assert_disjoint(train, validation):
    if set(train) & set(validation):
        raise ValueError("Document content overlaps")
    return True
''',
        connection='mini_gpt/data.py ابتدا متن را جدا می‌کند و سپس واژگان آموزش را می‌سازد. آن تابع خودش حذف سند تکراری انجام نمی‌دهد؛ بررسی سندها مسئولیتی است که این آزمایش روشن می‌کند.',
        takeaway='تفاوت «دو متغیر جدا» و «دو مجموعهٔ مستقل برای ادعای آزمایش» چیست؟'),

    '05-shape': dict(
        title='نشانی هر عدد در یک دستهٔ متن',
        goal='سه محور را با مقدار قابل ردیابی بسازید؛ نه فقط با یک tuple از اندازه‌ها.',
        prerequisite='فهرست تو‌در‌تو و حلقهٔ Python؛ هنوز PyTorch لازم نیست.',
        predict='در B=2، T=3 و C=4، مقدار 100*b+10*t+c در x[1][2][0] چیست؟ اگر فقط T دو برابر شود، کدام شمار تغییر می‌کند؟',
        setup='''B, T, C = 2, 3, 4
print("Axes: sample, token position, feature", (B, T, C))
''',
        task='تابع make_grid(B,T,C) فهرست سه‌محوری بسازد و خانهٔ b,t,c را برابر 100*b+10*t+c بگذارد. اندازه‌ها در آزمون مثبت‌اند. فهرست‌های درونی باید مستقل باشند.',
        starter='''def make_grid(B, T, C):
    # TODO: construct independent nested lists
    return None
''',
        check=_check('make_grid(2, 3, 4)', '''assert len(result) == 2 and all(len(sample) == 3 for sample in result)
assert all(len(row) == 4 for sample in result for row in sample)
assert result[1][2][0] == 120 and result[0][1][3] == 13
assert make_grid(1, 2, 2) == [[[0, 1], [10, 11]]]
result[0][0][0] = -1
assert result[1][0][0] == 100'''),
        solution='''def make_grid(B, T, C):
    return [[[100*b+10*t+c for c in range(C)] for t in range(T)] for b in range(B)]
''',
        vary='فقط تعداد موقعیت‌ها را از ۳ به ۶ تغییر دهید؛ تعداد نمونه و ویژگی ثابت باشند. تعداد عددها را قبل از دیدن جدول حساب کنید.',
        vary_code='''for positions in [3, 6]:
    lengths = [[C for _ in range(positions)] for _ in range(B)]
    print("Shape:", (B, positions, C), "Elements:", sum(sum(sample) for sample in lengths))
''',
        debug='ساخت جدول با ضرب list، یک فهرست مشترک را چند بار ارجاع می‌دهد. تابع zeros_grid باید جدول سه‌محوری صفر بسازد که تغییر یک خانه، خانهٔ دیگری را تغییر ندهد.',
        bug_code='''aliased = [[[0]*C]*T]*B
aliased[0][0][0] = 99
print("Unintended second-sample change:", aliased[1][0][0])
assert aliased[1][0][0] == 99
''',
        fix='''def zeros_grid(B, T, C):
    # TODO: make each inner list independently
    return None
''',
        fix_check=_check('zeros_grid(2, 3, 4)', 'assert result == [[[0]*4 for _ in range(3)] for _ in range(2)]\nresult[0][0][0] = 99\nassert result[0][1][0] == 0 and result[1][0][0] == 0', True),
        fix_solution='''def zeros_grid(B, T, C):
    return [[[0 for _ in range(C)] for _ in range(T)] for _ in range(B)]
''',
        connection='در Mini-GPT شناسه‌ها شکل (B,T) و نمایش هر شناسه شکل (B,T,C) دارد. مقدارهای این آزمایش فقط برچسب نشانی‌اند؛ ویژگی زبانیِ یادگرفته‌شده نیستند.',
        takeaway='اگر تعداد عنصرها درست باشد، چه آزمونی هنوز برای اطمینان از معنای محور لازم است؟'),
}

EXERCISES.update({
    '05a-vector-operations': dict(
        title='ترکیب دو نمایش بدون تغییر تعداد ویژگی‌ها',
        goal='جمع وزن‌دار Vectorها را از چسباندن list و ضرب داخلی جدا کنید.',
        prerequisite='درس 05-shape؛ حلقه و zip در Python.',
        predict='در ترکیب ۰٫۲۵ از u و ۰٫۷۵ از v، خروجی چند عدد دارد؟ با ضریب‌های ۱ و صفر چه باید برگردد؟',
        setup='''u, v = [1, 2], [3, -1]
print("Two features per vector:", u, v)
''',
        task='تابع mix(u,v,a,b) فهرست مؤلفه‌های a*u+b*v را برگرداند. طول نابرابر را با ValueError رد کنید. ضریب‌ها لازم نیست مثبت باشند یا جمعشان یک شود.',
        starter='''def mix(u, v, a, b):
    # TODO: return a weighted vector, not a scalar
    return None
''',
        check=_check('mix(u, v, 0.25, 0.75)', '''assert result == [2.5, -0.25]
assert mix(u, v, 1, 0) == u
assert mix(u, v, 1, -1) == [-2, 3]
try:
    mix([1], [2, 3], 1, 1)
except ValueError:
    pass
else:
    raise AssertionError("Unequal lengths must fail")'''),
        solution='''def mix(u, v, a, b):
    if len(u) != len(v):
        raise ValueError("Equal lengths required")
    return [a*x+b*y for x, y in zip(u, v)]
''',
        vary='فقط ضریب a را میان صفر، ۰٫۵ و یک تغییر دهید و b را یک منهای آن بگذارید. ببینید هر مؤلفه میان دو مقدار ورودی می‌ماند؛ این مشاهده برای همین ضریب‌های میانگین وزن‌دار است.',
        vary_code='''for a in [0, 0.5, 1]:
    print(a, [a*x+(1-a)*y for x, y in zip(u, v)])
''',
        debug='u+v برای list محاسبهٔ موردنظر نیست. تابع add_vectors را بنویسید تا دو Vector هم‌اندازه را مؤلفه‌به‌مؤلفه جمع کند.',
        bug_code='''wrong = u + v
print("Broken vector addition:", wrong, "length:", len(wrong))
assert len(wrong) == 4
''',
        fix='''def add_vectors(a, b):
    # TODO: inputs have equal lengths
    return None
''',
        fix_check=_check('add_vectors(u, v)', 'assert result == [4, 1]\nassert add_vectors([0, -2, 4], [3, 2, 1]) == [3, 0, 5]', True),
        fix_solution='''def add_vectors(a, b):
    return [x+y for x, y in zip(a, b)]
''',
        connection='جمع Token و Position در mini_gpt/model.py و جمع وزن‌دار Valueها در mini_gpt/attention.py تعداد ویژگی‌ها را حفظ می‌کنند؛ اینجا فقط حساب لازم را ساختیم.',
        takeaway='چه زمانی جمع وزن‌دار شما میانگین هم هست، و چه زمانی نیست؟'),

    '06-dot': dict(
        title='یک امتیاز از چند ترجیح',
        goal='اثر هر مؤلفه و اندازهٔ Vector را در امتیاز نهایی ردیابی کنید.',
        prerequisite='05a-vector-operations؛ ضرب مؤلفه‌های متناظر و جمع.',
        predict='q=[2,1] به a=[3,0] و b=[1,4] چه امتیازی می‌دهد؟ آیا امتیاز یکسان یعنی Vector یکسان؟',
        setup='''q, a, b = [2, 1], [3, 0], [1, 4]
print("Preference:", q, "Candidates:", a, b)
''',
        task='تابع dot(left,right) یک عدد برگرداند: جمع ضرب مؤلفه‌های متناظر. طول نابرابر را با ValueError رد کنید؛ خروجی دو Vector خالی را صفر در نظر بگیرید.',
        starter='''def dot(left, right):
    # TODO
    return None
''',
        check=_check('dot(q, a)', '''assert result == 6 and dot(q, b) == 6
assert dot([1, 2, 3], [4, 0, -1]) == 1
assert dot([], []) == 0
try:
    dot([1, 2], [3])
except ValueError:
    pass
else:
    raise AssertionError("zip must not silently truncate a vector")'''),
        solution='''def dot(left, right):
    if len(left) != len(right):
        raise ValueError("Equal lengths required")
    return sum(x*y for x, y in zip(left, right))
''',
        vary='فقط اندازهٔ b را دو برابر کنید، نه جهت آن را. سپس فقط وزن ترجیح اول را تغییر دهید. در هر آزمایش بگویید چه چیزی ثابت مانده است.',
        vary_code='''for candidate in [b, [2*value for value in b]]:
    print("candidate:", candidate, "score:", sum(x*y for x, y in zip(q, candidate)))
print("Changed preference:", sum(x*y for x, y in zip([1, 1], b)))
''',
        debug='نسخهٔ خراب حاصل‌های ضرب را برمی‌گرداند؛ هنوز Scalar نساخته است. تابع combine_products باید سهم‌های عددی آماده را به یک امتیاز تبدیل کند.',
        bug_code='''products = [q[i]*b[i] for i in range(len(q))]
print("Not yet a dot product:", products)
assert products == [2, 4]
''',
        fix='''def combine_products(products):
    # TODO: return one scalar
    return None
''',
        fix_check=_check('combine_products([2, 4])', 'assert result == 6\nassert combine_products([4, 0, -3]) == 1\nassert combine_products([]) == 0', True),
        fix_solution='''def combine_products(products):
    return sum(products)
''',
        connection='امتیاز هر جفت Query و Key در mini_gpt/attention.py از همین جمع ساخته می‌شود. نام ویژگی‌های مثال ما قراردادی است؛ ستون‌های مدل الزاماً نام انسانی ندارند.',
        takeaway='چرا نمی‌توان از بزرگ‌بودن امتیاز، بدون دانستن نمایش و اندازهٔ Vectorها، شباهت معنایی نتیجه گرفت؟'),

    '07-matmul': dict(
        title='همهٔ درخواست‌ها در برابر همهٔ متن‌ها',
        goal='جدول مقایسهٔ جفت‌ها را با ضرب ماتریسی خودتان بسازید.',
        prerequisite='06-dot و مفهوم Transpose در درس جاری؛ فقط Python.',
        predict='A دو سطر و سه ستون و B سه سطر و دو ستون دارد. خروجی چند خانه دارد و خانهٔ سطر صفر، ستون یک از کدام عددها ساخته می‌شود؟',
        setup='''a = [[1, 2, 3], [4, 5, 6]]
b = [[1, 0], [0, 1], [1, 1]]
print("A:", a, "B:", b)
''',
        task='تابع matmul(a,b) حاصل‌ضرب دو ماتریس مستطیلی غیرخالی را برگرداند. تعداد ستون a باید با تعداد سطر b برابر باشد؛ در غیر این صورت ValueError بدهید. اعتبار مستطیلی‌بودن در این تمرین مفروض است.',
        starter='''def matmul(a, b):
    # TODO: each output cell is a row-column dot product
    return None
''',
        check=_check('matmul(a, b)', '''assert result == [[4, 5], [10, 11]]
assert matmul([[1, 2], [3, 4]], [[5], [6]]) == [[17], [39]]
assert matmul([[2]], [[3, 4]]) == [[6, 8]]
try:
    matmul(a, a)
except ValueError:
    pass
else:
    raise AssertionError("Inner dimensions differ")'''),
        solution='''def matmul(a, b):
    if len(a[0]) != len(b):
        raise ValueError("Inner dimensions differ")
    return [[sum(a[i][k]*b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]
''',
        vary='فقط A[0][1] را یک واحد زیاد کنید. تغییر خروجی باید به سطر صفر محدود باشد؛ مقدار تغییر هر ستون از سطر یک B می‌آید.',
        vary_code='''delta = [[0, 1, 0], [0, 0, 0]]
effect = [[sum(delta[i][k]*b[k][j] for k in range(3)) for j in range(2)] for i in range(2)]
print("Change in product:", effect)
assert effect == [[0, 1], [0, 0]]
''',
        debug='برای مقایسهٔ سطرها باید K را Transpose کنیم؛ برعکس‌کردن ترتیب سطرها Transpose نیست. تابع transpose را برای ماتریس مستطیلی غیرخالی بنویسید.',
        bug_code='''keys = [[1, 2], [3, 4], [5, 6]]
wrong = keys[::-1]
print("Reversed rows:", wrong, "shape:", (len(wrong), len(wrong[0])))
assert len(wrong) == 3
''',
        fix='''def transpose(matrix):
    # TODO: swap row and column roles
    return None
''',
        fix_check=_check('transpose(keys)', 'assert result == [[1, 3, 5], [2, 4, 6]]\nassert transpose([[1, 2, 3]]) == [[1], [2], [3]]\nassert transpose(result) == keys', True),
        fix_solution='''def transpose(matrix):
    return [list(column) for column in zip(*matrix)]
''',
        connection='عمل q @ k.transpose(-2,-1) در mini_gpt/attention.py همین جدول جفت‌ها را می‌سازد؛ در آن فایل محورهای Batch و Head نیز وجود دارند که بعداً اضافه می‌کنیم.',
        takeaway='چرا آزمون غیرمربعی برای پیدا‌کردن اشتباه Transpose مفیدتر است؟'),

    '08-probability': dict(
        title='از سه بار دیده‌شدن تا احتمال و جریمه',
        goal='نسبت شمارش‌ها و منفی لگاریتم احتمال را با رفتار عددی‌شان بشناسید.',
        prerequisite='03-counts و log در همین درس؛ کتابخانهٔ استاندارد math.',
        predict='اگر هر دو شمارش [3,1] دو برابر شوند، احتمال‌ها تغییر می‌کنند؟ جریمهٔ احتمال ۰٫۱ چند برابر یا چند واحد با ۰٫۰۱ فرق دارد؟',
        setup='''import math
counts = [3, 1]
print("Observed continuation counts:", counts)
''',
        task='تابع probabilities(counts) شمارش‌های نامنفی با مجموع مثبت را به فهرست احتمال تبدیل کند. مجموع صفر یا شمارش منفی را با ValueError رد کنید.',
        starter='''def probabilities(counts):
    # TODO: normalize valid counts
    return None
''',
        check=_check('probabilities(counts)', '''assert result == [0.75, 0.25]
assert probabilities([6, 2]) == result
assert probabilities([0, 4]) == [0, 1]
for invalid in ([0, 0], [-1, 3]):
    try:
        probabilities(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid counts must fail")'''),
        solution='''def probabilities(counts):
    if any(value < 0 for value in counts) or sum(counts) <= 0:
        raise ValueError("Nonnegative counts with positive total required")
    total = sum(counts)
    return [value/total for value in counts]
''',
        vary='فقط احتمال پاسخ درست را ده برابر کوچک‌تر کنید. اختلاف جریمه‌ها را چاپ کنید؛ این آزمایش نسبت شمارش نیست، رفتار تابع log است.',
        vary_code='''for p in [0.1, 0.01, 0.001]:
    print(p, -math.log(p))
print("Penalty increment:", -math.log(0.01)+math.log(0.1))
''',
        debug='جمع دو احتمال شرطی، احتمال دو گام پیاپی نیست. تابع sequence_nll باید مجموع منفی لگاریتم احتمال‌های شرطی مثبت را برگرداند؛ احتمالی بیرون بازهٔ (0,1] را رد کنید.',
        bug_code='''conditional = [0.5, 0.25]
wrong_joint = sum(conditional)
print("Broken joint probability:", wrong_joint, "Actual product:", math.prod(conditional))
assert wrong_joint != math.prod(conditional)
''',
        fix='''def sequence_nll(probabilities):
    # TODO: return summed negative log probabilities
    return None
''',
        fix_check=_check('sequence_nll([0.5, 0.25])', '''assert math.isclose(result, math.log(8))
assert sequence_nll([1, 1]) == 0
try:
    sequence_nll([0])
except ValueError:
    pass
else:
    raise AssertionError("Zero probability needs an explicit boundary decision")''', True),
        fix_solution='''def sequence_nll(probabilities):
    if any(not 0 < p <= 1 for p in probabilities):
        raise ValueError("Probabilities must lie in (0,1]")
    return -sum(math.log(p) for p in probabilities)
''',
        connection='Loss مدل در mini_gpt/model.py بر احتمال هدف تکیه دارد؛ این آزمایش فقط حساب احتمال را انجام می‌دهد و هنوز آن فایل PyTorch را وارد نمی‌کند.',
        takeaway='چرا ضرب احتمال‌های شرطی دو گام، به معنی فرض استقلال آن دو نشانه نیست؟'),

    '09-softmax': dict(
        title='امتیازها را جابه‌جا کنید، احتمال را بسنجید',
        goal='Softmax پایدار را بنویسید و ثابت‌ماندن آن در برابر افزودن ثابت را آزمایش کنید.',
        prerequisite='08-probability؛ exp و جمع؛ بدون PyTorch.',
        predict='آیا [2,1,0] و [1002,1001,1000] احتمال یکسان می‌دهند؟ ضرب همهٔ امتیازها در دو چطور؟',
        setup='''import math
logits = [2., 1., 0.]
print("Raw scores:", logits, "sum:", sum(logits))
''',
        task='تابع softmax(logits) برای فهرست غیرخالی عددهای متناهی، احتمال‌های پایدار برگرداند. پیش از exp، بیشینه را کم کنید. ورودی خالی یا غیرمتناهی را با ValueError رد کنید.',
        starter='''def softmax(logits):
    # TODO: return stable probabilities
    return None
''',
        check=_check('softmax(logits)', '''assert len(result) == 3 and math.isclose(sum(result), 1)
assert math.isclose(result[0], 0.6652409557748218)
assert all(math.isclose(x, y) for x, y in zip(result, softmax([1002, 1001, 1000])))
assert softmax([7, 7]) == [0.5, 0.5]
for invalid in ([], [float("inf")]):
    try:
        softmax(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid logits must fail")'''),
        solution='''def softmax(logits):
    if not logits or not all(math.isfinite(z) for z in logits):
        raise ValueError("Nonempty finite logits required")
    maximum = max(logits)
    values = [math.exp(z-maximum) for z in logits]
    total = sum(values)
    return [value/total for value in values]
''',
        vary='فقط فاصلهٔ دو امتیاز را تغییر دهید: [0,gap] برای gapهای صفر، یک و دو. ترتیب ثابت است، اما اطمینان چطور تغییر می‌کند؟',
        vary_code='''for gap in [0, 1, 2]:
    weights = [math.exp(-gap), 1.]
    print("gap:", gap, "probabilities:", [w/sum(weights) for w in weights])
''',
        debug='محاسبهٔ مستقیم exp(1000) پیش از تقسیم سرریز می‌کند. تابع shifted_exp فقط نماییِ امتیازها پس از کم‌کردن بیشینه از آن‌ها را برگرداند؛ هنوز تقسیم انجام ندهید.',
        bug_code='''try:
    math.exp(1000)
except OverflowError as error:
    print("Expected overflow:", error)
else:
    raise AssertionError("The direct exponential should overflow")
''',
        fix='''def shifted_exp(logits):
    # TODO: input is nonempty and finite
    return None
''',
        fix_check=_check('shifted_exp([1000, 1001])', 'assert math.isclose(result[0], math.exp(-1)) and result[1] == 1\nassert shifted_exp([-4, -4]) == [1, 1]', True),
        fix_solution='''def shifted_exp(logits):
    maximum = max(logits)
    return [math.exp(z-maximum) for z in logits]
''',
        connection='mini_gpt/attention.py روی موقعیت‌ها Softmax می‌گیرد؛ mini_gpt/sampling.py روی گزینه‌های Vocabulary. اینجا یک سطر را ساختیم تا معنای محور بعداً گم نشود.',
        takeaway='کدام تغییر فقط مبدأ امتیازها را عوض کرد و کدام تغییر فاصلهٔ آن‌ها را؟'),

    '10-entropy': dict(
        title='یک پیش‌بینی، دو پاسخ درست متفاوت',
        goal='NLL را از Logits حساب کنید و وزن‌دهی نادرست Batchها را تشخیص دهید.',
        prerequisite='08-probability و09-softmax؛ این آزمایش دربارهٔ Cross Entropy با هدف تک‌کلاسه است.',
        predict='با Logits ثابت [2,1,0]، تغییر target از صفر به دو چه چیزی را عوض می‌کند: احتمال‌ها، Loss، یا هر دو؟',
        setup='''import math
logits = [2., 1., 0.]
print("Fixed scores:", logits, "Compare target IDs 0 and 2")
''',
        task='تابع cross_entropy(logits,target) از روش پایدار LogSumExp زیان هدف صحیح را برگرداند. logits غیرخالی و متناهی است؛ target باید int در محدوده باشد، نه bool یا اندیس منفی.',
        starter='''def cross_entropy(logits, target):
    # TODO: return one stable loss value
    return None
''',
        check=_check('cross_entropy(logits, 0)', '''assert math.isclose(result, 0.4076059644443806)
assert math.isclose(cross_entropy(logits, 2)-result, 2)
assert math.isclose(cross_entropy([1002, 1001, 1000], 0), result)
assert math.isclose(cross_entropy([0, 0], 1), math.log(2))
for invalid in (-1, 3, True):
    try:
        cross_entropy(logits, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid class ID")'''),
        solution='''def cross_entropy(logits, target):
    if type(target) is not int or not 0 <= target < len(logits):
        raise ValueError("Valid integer class ID required")
    maximum = max(logits)
    return maximum-logits[target]+math.log(sum(math.exp(z-maximum) for z in logits))
''',
        vary='فقط target را تغییر دهید. برای مقایسه، احتمال‌های ثابتی که قبلاً محاسبه شده‌اند را بخوانید؛ Sampling در این آزمایش نقشی ندارد.',
        vary_code='''fixed_probabilities = [0.6652409557748218, 0.24472847105479764, 0.09003057317038046]
for target in range(3):
    print("Target:", target, "Loss:", -math.log(fixed_probabilities[target]))
''',
        debug='میانگین دو Loss مربوط به Batchهای ۸ و ۲هدفه، به همهٔ هدف‌ها وزن برابر نمی‌دهد. تابع aggregate_loss جفت‌های (mean_loss, count) را درست ترکیب کند؛ countها مثبت‌اند.',
        bug_code='''batch_summaries = [(1., 8), (3., 2)]
wrong = sum(loss for loss, count in batch_summaries)/len(batch_summaries)
print("Broken batch mean:", wrong)
assert wrong == 2
''',
        fix='''def aggregate_loss(summaries):
    # TODO: weight each batch mean by its target count
    return None
''',
        fix_check=_check('aggregate_loss(batch_summaries)', 'assert math.isclose(result, 1.4)\nassert aggregate_loss([(2, 1), (4, 1)]) == 3\nassert aggregate_loss([(7, 5)]) == 7', True),
        fix_solution='''def aggregate_loss(summaries):
    return sum(loss*count for loss, count in summaries)/sum(count for loss, count in summaries)
''',
        connection='mini_gpt/evaluate.py زیان را بر اساس تعداد هدف‌ها وزن می‌دهد. mini_gpt/model.py نیز Logits را مستقیم به Cross Entropy می‌دهد، نه خروجی Sampling را.',
        takeaway='چرا کیفیت توزیع پیش‌بینی بدون دانستن هدف، عدد یکتایی به نام Loss ندارد؟'),

    '11-derivative': dict(
        title='یک تغییر کوچک چه اثری دارد؟',
        goal='مشتق عددی را از اختلاف خروجی‌ها بسازید و محدودیت اندازهٔ گام را ببینید.',
        prerequisite='تابع‌های Python و فرمول تفاضل مرکزی در درس؛ نیازی به PyTorch نیست.',
        predict='برای w² در w=3، اگر w را ۰٫۰۱ زیاد کنیم، خروجی تقریباً چقدر زیاد می‌شود؟ آیا کوچک‌ترین h همیشه دقیق‌ترین است؟',
        setup='''import math
def square(w):
    return w*w
print("Nearby values:", square(3), square(3.01))
''',
        task='تابع derivative(function,x,h) مشتق را با تفاضل مرکزی تقریب بزند. h باید مثبت باشد؛ h صفر یا منفی را با ValueError رد کنید.',
        starter='''def derivative(function, x, h):
    # TODO
    return None
''',
        check=_check('derivative(square, 3., 1e-4)', '''assert abs(result-6) < 1e-7
assert abs(derivative(lambda x: 4*x+7, -2, 1e-3)-4) < 1e-8
assert abs(derivative(lambda x: x*x*x, 2., 1e-4)-12) < 1e-6
try:
    derivative(square, 3, 0)
except ValueError:
    pass
else:
    raise AssertionError("h must be positive")'''),
        solution='''def derivative(function, x, h):
    if h <= 0:
        raise ValueError("h must be positive")
    return (function(x+h)-function(x-h))/(2*h)
''',
        vary='فقط h را تغییر دهید و خطای تقریب را ثبت کنید. برای h بسیار کوچک، فقط مشاهده را گزارش کنید؛ بدترشدن دقیقاً یکسان در همهٔ رایانه‌ها را assert نکنید.',
        vary_code='''for h in [1e-2, 1e-5, 1e-12]:
    estimate = (square(3+h)-square(3-h))/(2*h)
    print(h, estimate, "absolute error:", abs(estimate-6))
''',
        debug='برای مشتق جزئی باید فقط یک ورودی تغییر کند. نسخهٔ خراب w و b را با هم تغییر می‌دهد. تابع partial_w حساسیت به w را با b ثابت برگرداند.',
        bug_code='''def surface(w, b):
    return (w-2)**2+(b+1)**2
h = 1e-4
wrong = (surface(h, h)-surface(-h, -h))/(2*h)
print("Combined direction, not partial w:", wrong)
assert abs(wrong+2) < 1e-7
''',
        fix='''def partial_w(function, w, b, h):
    # TODO: hold b fixed
    return None
''',
        fix_check=_check('partial_w(surface, 0, 0, 1e-4)', 'assert abs(result+4) < 1e-7\nassert abs(partial_w(surface, 3, 8, 1e-4)-2) < 1e-7', True),
        fix_solution='''def partial_w(function, w, b, h):
    return (function(w+h, b)-function(w-h, b))/(2*h)
''',
        connection='Autograd بعداً همین حساسیت‌ها را برای Parameterهای Mini-GPT حساب می‌کند. تفاضل عددی ابزار بررسی است؛ برای آموزش هر Parameter این روش را تکرار نمی‌کنیم.',
        takeaway='در مشتق جزئی چه چیزی را تغییر دادید و چه چیزی را عمداً ثابت نگه داشتید؟'),

    '12-chain': dict(
        title='مسیر وزن تا Loss، با یک شاخه و دو شاخه',
        goal='ضرب حساسیت‌های یک مسیر و جمع سهم مسیرهای مختلف را جدا آزمایش کنید.',
        prerequisite='11-derivative؛ قاعدهٔ زنجیره‌ای و Learning rate در درس جاری.',
        predict='در w=1,x=2,y=5، علامت Gradient چه می‌گوید؟ اگر وزن هم‌زمان در دو شاخه استفاده شود، کدام سهم‌ها باید برگردند؟',
        setup='''import math
w, x, target = 1., 2., 5.
print("Prediction:", w*x, "Error:", w*x-target, "Loss:", (w*x-target)**2)
''',
        task='تابع one_step(w,x,target,rate) یک tuple شامل Gradient، وزن تازه و Loss با وزن تازه برگرداند. مدل w*x و Loss مربع خطاست.',
        starter='''def one_step(w, x, target, rate):
    # TODO: compute the gradient before changing w
    return None
''',
        check=_check('one_step(1., 2., 5., 0.1)', '''gradient, updated, loss = result
assert gradient == -12 and math.isclose(updated, 2.2) and math.isclose(loss, 0.36)
assert one_step(2., 1., 2., 0.2) == (0., 2., 0.)
assert one_step(1., 2., 5., 0.) == (-12., 1., 9.)'''),
        solution='''def one_step(w, x, target, rate):
    gradient = 2*(w*x-target)*x
    updated = w-rate*gradient
    return gradient, updated, (updated*x-target)**2
''',
        vary='فقط Learning rate را برای تابع (w-2)² تغییر دهید. همهٔ مسیرها از w=-3 شروع می‌شوند؛ این تابع با مثال اصلی متفاوت است و نرخ‌هایش نسخهٔ عمومی نیستند.',
        vary_code='''for rate in [0.1, 1., 1.1]:
    value = -3.
    path = [value]
    for _ in range(6):
        value -= rate*2*(value-2)
        path.append(value)
    print(rate, path)
''',
        debug='در L=(a*w+b*w)²، حذف سهم شاخهٔ دوم گرادیان غلط می‌سازد. تابع branched_gradient مشتق کل را برای هر w,a,b برگرداند.',
        bug_code='''shared, a, b = 1., 2., 3.
s = a*shared+b*shared
wrong = 2*s*a
print("Only the first branch:", wrong, "Both branch factors:", a, b)
assert wrong == 20
''',
        fix='''def branched_gradient(w, a, b):
    # TODO: include both paths through the shared weight
    return None
''',
        fix_check=_check('branched_gradient(1, 2, 3)', 'assert result == 50\nassert branched_gradient(1, 2, 4) == 72\nassert branched_gradient(2, 1, -1) == 0', True),
        fix_solution='''def branched_gradient(w, a, b):
    return 2*(a*w+b*w)*(a+b)
''',
        connection='یک Parameter در مدل می‌تواند بر چند خروجی اثر بگذارد؛ جمع سهم‌ها در backward انجام می‌شود. تغییر واقعی Parameterها در mini_gpt/train.py کار optimizer.step است.',
        takeaway='کدام محاسبه تنها حساسیت را به دست آورد و کدام دستور واقعاً وزن را جابه‌جا کرد؟'),

    '12-sgd': dict(
        title='جهت حرکت از یک نمونه یا یک دسته',
        goal='Gradient کل داده را با برآورد دسته مقایسه کنید، بدون ادعای بهبود در هر گام.',
        prerequisite='12-chain؛ میانگین و random.Random در Python.',
        predict='در w=0، چرا سه نمونهٔ (1,2)، (2,4)، (3,6) Gradient یکسان نمی‌دهند؟ تکرار همهٔ نمونه‌های یک Batch چه اثری بر میانگین دارد؟',
        setup='''import random
import math
data = [(1., 2.), (2., 4.), (3., 6.)]
print("Training examples:", data)
''',
        task='تابع batch_gradient(w,batch) Gradient میانگین مربع خطا برای مدل w*x را برگرداند. batch غیرخالی است.',
        starter='''def batch_gradient(w, batch):
    # TODO: average per-example gradients
    return None
''',
        check=_check('batch_gradient(0, data)', 'assert math.isclose(result, -56/3)\nassert batch_gradient(0, data[:2]) == -10\nassert batch_gradient(2, data) == 0\nassert batch_gradient(0, data+data) == result'),
        solution='''def batch_gradient(w, batch):
    return sum(2*(w*x-y)*x for x, y in batch)/len(batch)
''',
        vary='فقط اندازهٔ Batch را از یک به دو تغییر دهید؛ وزن ثابت و Seed ثابت بمانند. چند Gradient را ثبت کنید، نه فقط آخرین عدد را.',
        vary_code='''for size in [1, 2]:
    rng = random.Random(42)
    estimates = []
    for _ in range(8):
        selected = rng.choices(data, k=size)
        estimates.append(sum(-2*y*x for x, y in selected)/size)
    print("batch size:", size, "gradients at w=0:", estimates)
''',
        debug='نسخهٔ خراب Gradientها را جمع می‌کند؛ با تکرار Batch اندازهٔ گام دو برابر می‌شود. تابع mean_gradient باید از فهرست Gradientهای آماده، میانگین بسازد.',
        bug_code='''gradients = [-4., -16.]
print("Broken sum:", sum(gradients), "Duplicated batch:", sum(gradients*2))
assert sum(gradients*2) == 2*sum(gradients)
''',
        fix='''def mean_gradient(gradients):
    # TODO: input is nonempty
    return None
''',
        fix_check=_check('mean_gradient([-4., -16.])', 'assert result == -10\nassert mean_gradient([-4., -16.]*2) == result\nassert mean_gradient([3.]) == 3', True),
        fix_solution='''def mean_gradient(gradients):
    return sum(gradients)/len(gradients)
''',
        connection='mini_gpt/train.py پنجره‌ها را با جایگذاری انتخاب می‌کند. شیء Optimizer داده انتخاب نمی‌کند؛ این مسئولیت حلقه‌ای است که Batch را می‌سازد.',
        takeaway='برای مقایسهٔ هزینهٔ دو روش، چرا ثبت تعداد گام به‌تنهایی کافی نیست؟'),

    '12b-neuron': dict(
        title='دو ویژگی، یک Neuron و یک اصلاح واقعی',
        goal='Forward و Gradient دستی یک Neuron با ReLU را بسازید.',
        prerequisite='جمع وزن‌دار،12-chain و12-sgd؛ فقط Python.',
        predict='برای x=[1,2]، w=[0.5,-0.25] و Bias=1، پیش‌بینی چیست؟ اگر فقط Bias را -1 کنیم، آیا Loss غیرصفر حتماً Gradient غیرصفر می‌دهد؟',
        setup='''import math
x, w, bias, target = [1., 2.], [0.5, -0.25], 1., 2.
print("Input:", x, "Weights:", w, "Bias:", bias, "Target:", target)
''',
        task='تابع neuron(inputs,weights,bias) tupleِ (z, prediction) برگرداند: z جمع وزن‌دار همراه Bias و prediction برابر ReLU آن است. اندازه‌های نابرابر را رد کنید.',
        starter='''def neuron(inputs, weights, bias):
    # TODO: return preactivation and prediction
    return None
''',
        check=_check('neuron(x, w, bias)', '''assert result == (1., 1.)
assert neuron(x, w, -1.) == (-1., 0.)
assert neuron([0, 0], w, 3) == (3, 3)
try:
    neuron([1], [2, 3], 0)
except ValueError:
    pass
else:
    raise AssertionError("Feature and weight counts differ")'''),
        solution='''def neuron(inputs, weights, bias):
    if len(inputs) != len(weights):
        raise ValueError("Feature and weight counts differ")
    z = sum(x*w for x, w in zip(inputs, weights))+bias
    return z, max(0, z)
''',
        vary='فقط Target را از ۲ به ۳ تغییر دهید؛ x,w,b ثابت‌اند. پیش‌بینی را یک بار بسازید و Loss هر دو هدف را مقایسه کنید.',
        vary_code='''prediction = max(0., sum(a*b for a, b in zip(x, w))+bias)
for goal in [2., 3.]:
    print("Target:", goal, "Prediction:", prediction, "Loss:", (prediction-goal)**2)
''',
        debug='نسخهٔ خراب حتی در z منفی شیب ReLU را یک می‌گیرد. تابع neuron_gradients باید (grad_weights, grad_bias) را برای Loss مربع خطا برگرداند. در z=0 به‌جای فرض مشتق، ValueError بدهید.',
        bug_code='''negative_z = sum(a*b for a, b in zip(x, w))-1
wrong_grad = 2*(max(0, negative_z)-target)
print("z:", negative_z, "Broken bias gradient:", wrong_grad)
assert negative_z == -1 and wrong_grad == -4
''',
        fix='''def neuron_gradients(inputs, weights, bias, target):
    # TODO: include the local ReLU derivative; reject z=0
    return None
''',
        fix_check=_check('neuron_gradients(x, w, 1., 2.)', '''assert result == ([-2., -4.], -2.)
assert neuron_gradients(x, w, -1., 2.) == ([0., 0.], 0.)
new_w = [weight-0.1*gradient for weight, gradient in zip(w, result[0])]
new_b = bias-0.1*result[1]
assert math.isclose(max(0, sum(a*b for a, b in zip(x, new_w))+new_b), 2.2)
try:
    neuron_gradients([0], [1], 0, 2)
except ValueError:
    pass
else:
    raise AssertionError("The two-sided derivative at zero is not defined")''', True),
        fix_solution='''def neuron_gradients(inputs, weights, bias, target):
    z = sum(x*w for x, w in zip(inputs, weights))+bias
    if z == 0:
        raise ValueError("ReLU is not differentiable at zero")
    grad_z = 2*(max(0, z)-target)*(1 if z > 0 else 0)
    return [grad_z*x for x in inputs], grad_z
''',
        connection='لایه‌های mini_gpt/transformer.py نیز جمع وزن‌دار و Activation دارند، اما از GELU استفاده می‌کنند. این Neuron نسخهٔ کوچک برای فهم مسیر است، نه کپی معماری GPT.',
        takeaway='کدام مشاهده نشان داد Gradient صفر با «پاسخ درست» یکی نیست؟'),
})

EXERCISES.update({
    '21-tokenizer': dict(
        title='نویسهٔ ندیده و واژگانِ ذخیره‌شده',
        goal='قرارداد Unknown token را پیاده و با Tokenizer واقعی مقایسه کنید.',
        prerequisite='02-token و21-tokenizer؛ JSON و Path فقط در نمایش ذخیره‌سازی استفاده می‌شوند.',
        predict='اگر «ژ» و «چ» در Vocabulary نباشند، آیا پس از تبدیل هر دو به صفر می‌توان متن اصلی را دقیق بازسازی کرد؟',
        setup='''from pathlib import Path
from tempfile import TemporaryDirectory
from mini_gpt.tokenizer import CharacterTokenizer
text = "می‌روم home"
tokenizer = CharacterTokenizer.from_text(text)
vocabulary = list(tokenizer.id_to_token)
print("Vocabulary:", list(enumerate(vocabulary)))
''',
        task='تابع encode_unknown(text,vocabulary) فهرست IDها را بسازد. فهرست Vocabulary معتبر است و خانهٔ صفر آن <|unk|> است. هر نویسهٔ غایب را به صفر ببرید؛ ترتیب فهرست را حفظ کنید.',
        starter='''def encode_unknown(text, vocabulary):
    # TODO: encode known characters and map unknown ones to zero
    return None
''',
        check=_check('encode_unknown(text, vocabulary)', 'assert result == tokenizer.encode(text)\nassert encode_unknown("ژچ", vocabulary) == [0, 0]\nassert encode_unknown("", vocabulary) == []\nassert encode_unknown("با", ["<|unk|>", "ب", "ا"]) == [1, 2]\nassert tokenizer.decode(result) == text'),
        solution='''def encode_unknown(text, vocabulary):
    mapping = {token: index for index, token in enumerate(vocabulary)}
    return [mapping.get(char, 0) for char in text]
''',
        vary='فقط محل نگهداری Vocabulary را تغییر دهید: از حافظه به فایل موقت و دوباره به حافظه. ترتیب و خروجی باید ثابت بمانند؛ فایل موقت پس از آزمایش حذف می‌شود.',
        vary_code='''with TemporaryDirectory() as directory:
    path = Path(directory)/"vocabulary.json"
    tokenizer.save(path)
    restored = CharacterTokenizer.load(path)
    print("Preserved order:", restored.id_to_token == vocabulary)
    assert restored.encode(text) == tokenizer.encode(text)
print("Unknown round trip:", tokenizer.decode(tokenizer.encode("ژچ")))
''',
        debug='اندازهٔ Vocabulary کافی نیست. نسخهٔ خراب دو شناسهٔ معمولی را جابه‌جا می‌کند. تابع compatible_vocab باید فقط وقتی True بدهد که نگاشت شماره به نشانه دقیقاً یکسان مانده باشد.',
        bug_code='''changed = vocabulary[:]
changed[1], changed[2] = changed[2], changed[1]
print("Broken compatibility check:", len(changed) == len(vocabulary))
assert len(changed) == len(vocabulary) and changed != vocabulary
''',
        fix='''def compatible_vocab(before, after):
    # TODO: compare the full ID-to-token mapping
    return None
''',
        fix_check=_check('compatible_vocab(vocabulary, vocabulary[:])', 'assert result is True\nassert compatible_vocab(vocabulary, changed) is False\nassert compatible_vocab(["<|unk|>", "a"], ["<|unk|>", "b"]) is False', True),
        fix_solution='''def compatible_vocab(before, after):
    return list(before) == list(after)
''',
        connection='در این دفتر CharacterTokenizer واقعی از mini_gpt/tokenizer.py را اجرا کردیم. Checkpoint مدل همین ترتیب را نگه می‌دارد؛ جدول وزن بدون نگاشت درست قابل تفسیر نیست.',
        takeaway='چرا رفت‌وبرگشت متن شناخته‌شده و متن دارای Unknown token دو انتظار متفاوت دارند؟'),

    '22-bpe': dict(
        title='یک ادغام، بدون بلعیدن نویسه‌ها',
        goal='ادغام غیرهم‌پوشان و اهمیت ترتیب قواعد BPE را خودتان پیاده کنید.',
        prerequisite='21-tokenizer و جفت‌های مجاور؛ این نسخهٔ آموزشی جایگزین Tokenizer پروژه نیست.',
        predict='با ادغام جفت a,a در aaaa، چند قطعهٔ aa به دست می‌آید؟ آیا می‌توان هر سه جفت هم‌پوشان را نگه داشت؟',
        setup='''from collections import Counter
tokens = list("aaaa")
words = [("کار", 2), ("کارگر", 1), ("گرم", 1)]
print("Initial pieces:", tokens)
''',
        task='تابع merge(tokens,pair) از چپ به راست حرکت کند؛ دو قطعهٔ متناظر با pair را بچسباند و هر قطعهٔ ورودی را فقط یک بار مصرف کند. ورودی را تغییر ندهید.',
        starter='''def merge(tokens, pair):
    # TODO: merge non-overlapping adjacent pairs
    return None
''',
        check=_check('merge(tokens, ("a", "a"))', 'assert result == ["aa", "aa"]\nassert merge(list("aaa"), ("a", "a")) == ["aa", "a"]\nassert merge(["کا", "ر", "گ", "ر"], ("کا", "ر")) == ["کار", "گ", "ر"]\nassert merge(["a", "b"], ("x", "y")) == ["a", "b"]\nassert tokens == list("aaaa")'),
        solution='''def merge(tokens, pair):
    output, index = [], 0
    while index < len(tokens):
        if index+1 < len(tokens) and (tokens[index], tokens[index+1]) == pair:
            output.append(tokens[index]+tokens[index+1])
            index += 2
        else:
            output.append(tokens[index])
            index += 1
    return output
''',
        vary='فقط فراوانی «گرم» را از یک به ده تغییر دهید. شمارش جفت‌ها را تازه بسازید و ببینید نخستین انتخاب چه تغییری می‌کند؛ قانون تساوی ثابت است.',
        vary_code='''for frequency in [1, 10]:
    counts = Counter()
    for word, count in [("کار", 2), ("کارگر", 1), ("گرم", frequency)]:
        for pair in zip(word, word[1:]):
            counts[pair] += count
    chosen = min(counts, key=lambda pair: (-counts[pair], pair))
    print("Frequency:", frequency, "Chosen pair:", chosen, "Counts:", counts)
''',
        debug='قواعد ذخیره‌شده باید به ترتیب آموخته‌شده اجرا شوند. تابع apply_rules(tokens,rules) تمام قواعد را با ادغام غیرهم‌پوشان اجرا کند؛ مستقل از TODO اصلی بنویسید تا بتوان هر تمرین را جدا آزمود.',
        bug_code='''rules = [("a", "b"), ("ab", "c")]
wrong_order = list(reversed(rules))
pieces = list("abc")
for left, right in wrong_order:
    index = 0
    while index+1 < len(pieces):
        if pieces[index:index+2] == [left, right]:
            pieces[index:index+2] = [left+right]
        index += 1
print("Rules in reverse order:", pieces)
assert pieces == ["ab", "c"]
''',
        fix='''def apply_rules(tokens, rules):
    # TODO: apply the supplied rules in their given order
    return None
''',
        fix_check=_check('apply_rules(list("abc"), rules)', 'assert result == ["abc"]\nassert apply_rules(list("aaaa"), [("a", "a")]) == ["aa", "aa"]\nassert apply_rules(list("abc"), []) == ["a", "b", "c"]', True),
        fix_solution='''def apply_rules(tokens, rules):
    pieces = list(tokens)
    for pair in rules:
        output, index = [], 0
        while index < len(pieces):
            if index+1 < len(pieces) and (pieces[index], pieces[index+1]) == pair:
                output.append(pieces[index]+pieces[index+1])
                index += 2
            else:
                output.append(pieces[index])
                index += 1
        pieces = output
    return pieces
''',
        connection='Mini-GPT فعلی عمداً از CharacterTokenizer استفاده می‌کند. BPE این دفتر یک آزمایش مستقل دربارهٔ طول دنباله است، نه API پنهان یا پیاده‌سازی کامل BPE در پروژه.',
        takeaway='برای بازسازی Encoding، چرا داشتن فهرست قطعه‌های نهایی به‌تنهایی کافی نیست؟'),

    '23-shift': dict(
        title='هدف هر موقعیت را با چشم دنبال کنید',
        goal='پنجره‌های ورودی/هدف را خودتان بسازید و با Dataset واقعی تطبیق دهید.',
        prerequisite='20-loader و21-tokenizer؛ برش Tensor.',
        predict='از پنج ID و T=3 چند نمونه می‌سازیم؟ آخرین هدفِ آخرین نمونه از کدام خانهٔ دنبالهٔ خام می‌آید؟',
        setup='''import torch
from mini_gpt.dataset import NextTokenDataset
ids, T = [1, 2, 3, 4, 5], 3
reference = NextTokenDataset(ids, T)
print("Raw IDs:", ids, "Window length:", T)
''',
        task='تابع make_windows(ids,T) فهرست جفت‌های (x,y) را با listهای Python بسازد. T باید int مثبت و طول ids بیشتر از T باشد؛ در غیر این صورت ValueError بدهید. هر هدف دقیقاً یک خانه جلوتر است.',
        starter='''def make_windows(ids, T):
    # TODO: include all valid starts and the final target
    return None
''',
        check=_check('make_windows(ids, T)', '''assert result == [([1, 2, 3], [2, 3, 4]), ([2, 3, 4], [3, 4, 5])]
assert result == [(x.tolist(), y.tolist()) for x, y in reference]
assert make_windows([4, 5], 1) == [([4], [5])]
for invalid_T in (0, 5, True):
    try:
        make_windows(ids, invalid_T)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid context length must fail")'''),
        solution='''def make_windows(ids, T):
    if type(T) is not int or T < 1 or len(ids) <= T:
        raise ValueError("Positive integer T shorter than the input required")
    return [(ids[start:start+T], ids[start+1:start+T+1]) for start in range(len(ids)-T)]
''',
        vary='فقط T را از دو به چهار تغییر دهید. از NextTokenDataset واقعی استفاده کنید و تعداد نمونه‌ها و آخرین هدف را کنار هم ببینید.',
        vary_code='''for length in [2, 3, 4]:
    dataset = NextTokenDataset(ids, length)
    last_x, last_y = dataset[len(dataset)-1]
    print(length, "samples:", len(dataset), "last:", last_x.tolist(), last_y.tolist())
''',
        debug='برابری طول x و y ثابت نمی‌کند هدف درست است. نسخهٔ خراب y=x است. تابع aligned_target(raw,start,T,target) بررسی کند target دقیقاً برش یک‌خانه‌جلوتر است.',
        bug_code='''wrong_x = ids[:T]
wrong_y = wrong_x[:]
print("Broken equal-length pair:", wrong_x, wrong_y)
assert len(wrong_x) == len(wrong_y)
''',
        fix='''def aligned_target(raw, start, T, target):
    # TODO: return a bool; check the whole target including its last item
    return None
''',
        fix_check=_check('aligned_target(ids, 0, 3, [2, 3, 4])', 'assert result is True\nassert aligned_target(ids, 0, 3, wrong_y) is False\nassert aligned_target(ids, 1, 3, [3, 4, 4]) is False\nassert aligned_target(ids, 1, 3, [3, 4, 5]) is True', True),
        fix_solution='''def aligned_target(raw, start, T, target):
    return len(target) == T and list(target) == list(raw[start+1:start+T+1])
''',
        connection='تمام جفت‌ها با mini_gpt/dataset.py مقایسه شدند. درست‌بودن Shift هنوز جلوی دیدن آینده را نمی‌گیرد؛ آن محدودیت را بعداً در Attention آزمایش می‌کنیم.',
        takeaway='چرا فقط مقایسهٔ y[:-1] با x[1:] برای بررسی آخرین هدف کافی نیست؟'),

    '24-data-contract': dict(
        title='پیش از آموزش، داده واقعاً چیست؟',
        goal='طول پنجره و نرخ Unknown را با قرارداد واقعی prepare_corpus بررسی کنید.',
        prerequisite='21-tokenizer و23-shift؛ شمارش شناسه‌ها و فایل UTF-8.',
        predict='اگر نویسهٔ Z فقط در انتهای اعتبارسنجی باشد، آیا باید وارد Vocabulary آموزش شود؟ آیا طول غیرصفر برای ساخت پنجره کافی است؟',
        setup='''from pathlib import Path
from tempfile import TemporaryDirectory
from mini_gpt.data import prepare_corpus
with TemporaryDirectory() as directory:
    path = Path(directory)/"corpus.txt"
    path.write_text("abcabcabcZ", encoding="utf-8")
    train_ids, valid_ids, tokenizer, info = prepare_corpus(path, train_fraction=0.6)
print("Train:", train_ids, "Validation:", valid_ids, "Metadata:", info)
''',
        task='تابع audit_ids(train_ids,valid_ids,T) دیکشنری با کلیدهای train_windows، valid_windows و unknown_rate برگرداند. تعداد پنجره‌ها len(ids)-T است. اگر هر بخش حداکثر T شناسه دارد ValueError بدهید. ID ناشناخته صفر است؛ T در این تمرین مثبت است.',
        starter='''def audit_ids(train_ids, valid_ids, T):
    # TODO: audit the actual encoded input, not just the original text
    return None
''',
        check=_check('audit_ids(train_ids, valid_ids, 2)', '''assert result == {"train_windows": 4, "valid_windows": 2, "unknown_rate": 0.25}
assert result["unknown_rate"] == info["validation_unknown_rate"]
assert "Z" not in tokenizer.token_to_id
assert audit_ids([1, 2, 3], [0, 0, 1], 1)["unknown_rate"] == 2/3
try:
    audit_ids([1, 2], [1, 2, 3], 2)
except ValueError:
    pass
else:
    raise AssertionError("A final target must exist in both splits")'''),
        solution='''def audit_ids(train_ids, valid_ids, T):
    if len(train_ids) <= T or len(valid_ids) <= T:
        raise ValueError("Both splits must contain more than T IDs")
    return {"train_windows": len(train_ids)-T,
            "valid_windows": len(valid_ids)-T,
            "unknown_rate": valid_ids.count(0)/len(valid_ids)}
''',
        vary='فقط یک نویسه از محتوای فایل موقت را عوض کنید. اثر انگشت باید عوض شود؛ این تغییر چیزی دربارهٔ بهتر یا بدترشدن کیفیت متن نمی‌گوید.',
        vary_code='''import hashlib
for text in ["abcabcabcZ", "abcabcabcY"]:
    print(text, hashlib.sha256(text.encode("utf-8")).hexdigest())
''',
        debug='ساخت Vocabulary از کل متن، Unknown اعتبارسنجی را پنهان می‌کند. تابع train_tokenizer(text,fraction) فقط برش آموزش را به CharacterTokenizer بدهد؛ fraction معتبر و برش آموزش غیرخالی است.',
        bug_code='''from mini_gpt.tokenizer import CharacterTokenizer
whole_text = "abcabcabcZ"
wrong_tokenizer = CharacterTokenizer.from_text(whole_text)
print("Leaked Z ID:", wrong_tokenizer.encode("Z"))
assert wrong_tokenizer.encode("Z") != [0]
''',
        fix='''def train_tokenizer(text, fraction):
    # TODO: build vocabulary only from the training prefix
    return None
''',
        fix_check=_check('train_tokenizer(whole_text, 0.6)', 'assert result.encode("Z") == [0]\nassert result.id_to_token == tokenizer.id_to_token\nassert train_tokenizer("aaaaZZ", 2/3).encode("Z") == [0]', True),
        fix_solution='''def train_tokenizer(text, fraction):
    return CharacterTokenizer.from_text(text[:int(len(text)*fraction)])
''',
        connection='این آزمایش prepare_corpus واقعی را با فایل موقت اجرا می‌کند. فایل data/sample.txt و داده‌های کاربر را تغییر نمی‌دهد؛ اطلاعات خروجی همان اطلاعات ذخیره‌شده در آموزش پروژه است.',
        takeaway='چرا کاهش نرخ Unknown با دیدن کل داده، لزوماً به معنی بهترشدن آزمایش نیست؟'),

    '25-embedding': dict(
        title='شمارهٔ سطر یاد نمی‌گیرد؛ عددهای سطر یاد می‌گیرند',
        goal='Lookup را خودتان بنویسید و Gradient سطرهای تکراری را در مدل واقعی ببینید.',
        prerequisite='17-autograd،18-module و21-tokenizer؛ Embedding در درس جاری.',
        predict='برای IDهای [1,3,1] و Loss برابر مجموع خروجی، کدام سطر جدول دو سهم Gradient می‌گیرد و کدام سطرها هیچ سهمی ندارند؟',
        setup='''import torch
from torch import nn
from mini_gpt.stages.v1 import TokenOnly
torch.manual_seed(7)
table = torch.arange(15, dtype=torch.float32).reshape(5, 3)
ids = torch.tensor([[1, 3, 1]], dtype=torch.long)
print("Known table:\\n", table, "IDs:", ids)
''',
        task='تابع lookup(table,ids) با Indexing سطرهای جدول را انتخاب کند. table شکل (V,C) و ids شکل (B,T) دارد. خروجی باید (B,T,C) باشد؛ IDها معتبر و long هستند.',
        starter='''def lookup(table, ids):
    # TODO: preserve both ID axes and add the feature axis
    return None
''',
        check=_check('lookup(table, ids)', 'assert tuple(result.shape) == (1, 3, 3)\nassert result.tolist() == [[[3., 4., 5.], [9., 10., 11.], [3., 4., 5.]]]\nassert torch.equal(result[0, 0], result[0, 2])\nassert tuple(lookup(table, torch.tensor([[0], [4]])).shape) == (2, 1, 3)'),
        solution='''def lookup(table, ids):
    return table[ids]
''',
        vary='فقط تعداد تکرار ID یک را تغییر دهید. جدول تازه با همان مقدارهای ثابت بسازید تا تفاوت Gradient فقط از تعداد استفاده بیاید. اینجا from_pretrained مقدار آغاز جدول را از Tensor می‌گیرد و freeze=False یعنی عددهای جدول قابل آموزش بمانند.',
        vary_code='''for sequence in [[1, 3, 1], [1, 3, 1, 1]]:
    embedding = nn.Embedding.from_pretrained(table.clone(), freeze=False)
    embedding(torch.tensor([sequence])).sum().backward()
    print(sequence, "row-1 gradient:", embedding.weight.grad[1].tolist())
model = TokenOnly(vocab_size=5, channels=3)
print("Actual v1 logits shape:", tuple(model(ids).shape))
''',
        debug='میانگین‌گرفتن سهم‌های یک ID، قاعدهٔ جمع شاخه‌ها را عوض می‌کند. تابع row_counts(ids,V) تعداد وقوع هر ID را برگرداند؛ برای Loss جمع خروجی، این عدد Gradient هر مؤلفهٔ همان سطر است.',
        bug_code='''wrong = [int(i in ids.flatten().tolist()) for i in range(5)]
print("Presence is not usage count:", wrong)
assert wrong[1] == 1
''',
        fix='''def row_counts(ids, V):
    # TODO: count every occurrence, not just presence
    return None
''',
        fix_check=_check('row_counts(ids, 5)', 'assert result == [0, 2, 0, 1, 0]\nassert row_counts(torch.tensor([[0, 0], [2, 0]]), 3) == [3, 0, 1]', True),
        fix_solution='''def row_counts(ids, V):
    values = ids.flatten().tolist()
    return [values.count(index) for index in range(V)]
''',
        connection='TokenOnly از mini_gpt/stages/v1.py واقعاً اجرا شد: Embedding سپس head. این مدل هنوز اطلاعات موقعیت‌های دیگر را ترکیب نمی‌کند و شباهت معنایی جدول تصادفی را ادعا نمی‌کنیم.',
        takeaway='چه چیزی آموختنی است: ID، عمل انتخاب سطر، یا مقدارهای جدول؟'),

    '26-positions': dict(
        title='یک نشانه در دو جای متفاوت',
        goal='نمایش Token و Position را با حفظ محورهای Batch، زمان و ویژگی ترکیب کنید.',
        prerequisite='15-broadcast و25-embedding؛ هنوز ساخت Attention لازم نیست.',
        predict='دو وقوع ID یک، قبل و بعد از افزودن موقعیت چه رابطه‌ای دارند؟ آیا جمع دو Vector Cتایی خروجی 2Cتایی می‌دهد؟',
        setup='''import torch
from torch import nn
token_table = torch.tensor([[0., 0.], [2., 3.], [4., 5.]])
position_table = torch.tensor([[0., 1.], [1., 0.], [-1., 2.], [2., -1.]])
ids = torch.tensor([[1, 2, 1]], dtype=torch.long)
print("Same token at positions 0 and 2:", token_table[ids])
''',
        task='تابع add_positions(ids,token_table,position_table) سطرهای Token را بردارد و برای موقعیت t سطر t از جدول Position را جمع کند. موقعیت‌ها برای هر نمونه از صفر شروع می‌شوند. طول بیش از تعداد سطر Position را با ValueError رد کنید.',
        starter='''def add_positions(ids, token_table, position_table):
    # TODO: add position vectors across the batch axis
    return None
''',
        check=_check('add_positions(ids, token_table, position_table)', '''assert tuple(result.shape) == (1, 3, 2)
assert result.tolist() == [[[2., 4.], [5., 5.], [1., 5.]]]
two = add_positions(ids.repeat(2, 1), token_table, position_table)
assert torch.equal(two[0], two[1])
try:
    add_positions(torch.ones(1, 5, dtype=torch.long), token_table, position_table)
except ValueError:
    pass
else:
    raise AssertionError("Position table has a finite context limit")'''),
        solution='''def add_positions(ids, token_table, position_table):
    T = ids.shape[1]
    if T > len(position_table):
        raise ValueError("Context exceeds position table")
    return token_table[ids]+position_table[:T]
''',
        vary='فقط جدول Position را صفر کنید؛ Tokenها و IDها ثابت‌اند. برابری دو وقوع یک ID باید برگردد. این مشاهده دربارهٔ همین مرحله است، نه تمام اطلاعات ترتیب در معماری کامل.',
        vary_code='''base = token_table[ids]
for positions in [position_table, torch.zeros_like(position_table)]:
    shown = base+positions[:ids.shape[1]]
    print("Position 0:", shown[0, 0].tolist(), "Position 2:", shown[0, 2].tolist())
''',
        debug='در نسخهٔ خراب، ID نشانه به‌جای شمارهٔ موقعیت به جدول Position داده می‌شود. تابع positions_for(ids) Tensor شماره‌های ۰ تا T-1 را روی دستگاه ids برگرداند؛ شکل خروجی (T,) است.',
        bug_code='''wrong = token_table[ids]+position_table[ids]
print("Broken repeated-token equality:", torch.equal(wrong[0, 0], wrong[0, 2]))
assert torch.equal(wrong[0, 0], wrong[0, 2])
''',
        fix='''def positions_for(ids):
    # TODO: depend on sequence length, not on token values
    return None
''',
        fix_check=_check('positions_for(ids)', 'assert result.tolist() == [0, 1, 2]\nassert result.dtype == torch.long and result.device == ids.device\nassert positions_for(torch.tensor([[2, 2]])).tolist() == [0, 1]', True),
        fix_solution='''def positions_for(ids):
    return torch.arange(ids.shape[1], device=ids.device)
''',
        connection='mini_gpt/stages/v4.py و MiniGPT.forward همین انتخاب موقعیت با arange و جمع دو جدول را انجام می‌دهند. اینجا آن جزء را بدون اجرای Attention جدا کردیم.',
        takeaway='چرا متفاوت‌شدن دو بردار، هنوز شاهد یادگرفتن معنای ترتیب نیست؟'),
})

EXERCISES.update({
    '13-torch': dict(
        title='همان جدول Python، این بار در Tensor',
        goal='یک Batch شناسه بسازید و معنای shape، ndim و numel را روی آن بررسی کنید.',
        prerequisite='05-shape و نصب PyTorch در همان محیط Jupyter؛ نیازی به NumPy نیست.',
        predict='جدول دو متنِ سه‌نشانه‌ای چند محور و چند عنصر دارد؟ آیا عدد سه در شکل (2,3) تعداد ویژگی‌هاست؟',
        setup='''import torch
torch.set_num_threads(1)
rows = [[1, 2, 3], [4, 5, 6]]
print("PyTorch:", torch.__version__, "Python rows:", rows)
''',
        task='تابع ids_tensor(rows) جدول مستطیلی شناسه‌ها را به Tensor با dtype=torch.long و device=cpu تبدیل کند. rows در این تمرین شامل int است.',
        starter='''def ids_tensor(rows):
    # TODO: return a CPU integer tensor
    return None
''',
        check=_check('ids_tensor(rows)', '''assert tuple(result.shape) == (2, 3)
assert result.ndim == 2 and result.numel() == 6
assert result.dtype == torch.long and result.device.type == "cpu"
assert result.tolist() == rows
assert tuple(ids_tensor([[9, 8]]).shape) == (1, 2)'''),
        solution='''def ids_tensor(rows):
    return torch.tensor(rows, dtype=torch.long, device="cpu")
''',
        vary='فقط یک متن سه‌نشانه‌ای به Batch اضافه کنید. سپس فقط یک موقعیت به هر متن اضافه کنید. در هر بار نام محور تغییرکرده را بنویسید.',
        vary_code='''for value in [rows, rows+[[7, 8, 9]], [[1, 2, 3, 7], [4, 5, 6, 8]]]:
    tensor = torch.tensor(value)
    print(tuple(tensor.shape), tensor.ndim, tensor.numel())
''',
        debug='دو سطر با طول متفاوت یک جدول مستطیلی نمی‌سازند. خطای عمدی را ببینید و تابع is_rectangular را بنویسید تا پیش از ساخت Tensor این وضعیت را تشخیص دهد؛ جدول خالی را False در نظر بگیرید.',
        bug_code='''try:
    torch.tensor([[1, 2], [3]])
except (ValueError, RuntimeError) as error:
    print("Expected ragged-data failure:", error)
else:
    raise AssertionError("Ragged rows should fail")
''',
        fix='''def is_rectangular(rows):
    # TODO: require at least one row and one column
    return None
''',
        fix_check=_check('is_rectangular([[1, 2], [3, 4]])', 'assert result is True\nassert is_rectangular([[1, 2], [3]]) is False\nassert is_rectangular([]) is False\nassert is_rectangular([[]]) is False', True),
        fix_solution='''def is_rectangular(rows):
    return bool(rows) and bool(rows[0]) and all(len(row) == len(rows[0]) for row in rows)
''',
        connection='ورودی MiniGPT.forward در mini_gpt/model.py شناسه‌های (B,T) است. هنوز محور نمایش C را نساخته‌ایم؛ آن را در آزمایش Embedding اضافه می‌کنیم.',
        takeaway='از روی کدام ویژگی Tensor، تعداد محور را می‌فهمید و از روی کدام، تعداد کل عددها را؟'),

    '14-index-device': dict(
        title='یک نمونه را بردارید، Batch را نگه دارید',
        goal='انتخاب داده، نوع عدد و دستگاه را جدا بررسی کنید.',
        prerequisite='13-torch؛ Indexing و Slicing درس جاری.',
        predict='برای x با شکل (2,3)، x[0] و x[0:1] چه شکل‌هایی دارند؟ تبدیل dtype آیا تعداد عنصرها را تغییر می‌دهد؟',
        setup='''import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
print("Input:", x, x.dtype, x.device)
''',
        task='تابع select_sample(x,index) یک نمونه را با حفظ محور Batch برگرداند. خروجی باید نوع عدد و دستگاه x را نگه دارد؛ اندیس‌های آزمون در محدوده‌اند.',
        starter='''def select_sample(x, index):
    # TODO: preserve the batch axis
    return None
''',
        check=_check('select_sample(x, 0)', 'assert tuple(result.shape) == (1, 3)\nassert torch.equal(result, torch.tensor([[1, 2, 3]]))\nassert result.dtype == x.dtype and result.device == x.device\nassert select_sample(x, 1).tolist() == [[4, 5, 6]]\nassert tuple(select_sample(x[:1], 0).shape) == (1, 3)'),
        solution='''def select_sample(x, index):
    return x[index:index+1]
''',
        vary='فقط نوع عدد را از long به float32 تغییر دهید. مقدار، شکل و دستگاه را کنار هم چاپ کنید؛ نیازی به GPU نداریم.',
        vary_code='''converted = x.float()
for value in [x, converted]:
    print(value.tolist(), tuple(value.shape), value.dtype, value.device)
assert torch.equal(converted.long(), x)
''',
        debug='فراخوانی float بدون نگه‌داشتن خروجی، متغیر اصلی را عوض نمی‌کند. تابع as_float_cpu نتیجهٔ تبدیل را برگرداند و ورودی را دست‌نخورده بگذارد.',
        bug_code='''original = torch.tensor([1, 2], dtype=torch.long)
original.float()
print("Still integer:", original.dtype)
assert original.dtype == torch.long
''',
        fix='''def as_float_cpu(value):
    # TODO: return the conversion result
    return None
''',
        fix_check=_check('as_float_cpu(original)', 'assert result.dtype == torch.float32 and result.device.type == "cpu"\nassert result.tolist() == [1., 2.]\nassert original.dtype == torch.long\nassert as_float_cpu(torch.tensor([[3.]])).shape == (1, 1)', True),
        fix_solution='''def as_float_cpu(value):
    return value.to(device="cpu", dtype=torch.float32)
''',
        connection='Mini-GPT شناسهٔ صحیح و وزن اعشاری را جدا نگه می‌دارد. تبدیل این تمرین برای مشاهده است؛ شناسهٔ float را به جدول Embedding نمی‌دهیم.',
        takeaway='چرا «عددها همان‌اند» برای اثبات سازگاری دو Tensor کافی نیست؟'),

    '15-broadcast': dict(
        title='میانگین هر سطر باید از همان سطر کم شود',
        goal='یک خطای آشکار و یک خطای خاموش Broadcasting را با مقدارها پیدا کنید.',
        prerequisite='13-torch و14-index-device؛ mean، dim و keepdim در درس.',
        predict='چرا حذف keepdim در جدول ۲×۳ خطا می‌دهد، ولی در جدول ۳×۳ ممکن است بی‌خطا و اشتباه اجرا شود؟',
        setup='''import torch
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
square = torch.arange(1., 10.).reshape(3, 3)
print("Input rows:", x)
''',
        task='تابع center_rows(x) از هر خانه میانگین سطر خودش را کم کند. x یک Matrix اعشاری است؛ شکل خروجی باید دقیقاً شکل ورودی بماند.',
        starter='''def center_rows(x):
    # TODO: keep the reduced column axis
    return None
''',
        check=_check('center_rows(x)', 'assert result.shape == x.shape\nassert torch.allclose(result, torch.tensor([[-1., 0., 1.], [-1., 0., 1.]]))\nassert torch.allclose(center_rows(square).mean(1), torch.zeros(3))\nassert torch.equal(center_rows(torch.tensor([[5.]])), torch.zeros(1, 1))'),
        solution='''def center_rows(x):
    return x-x.mean(dim=1, keepdim=True)
''',
        vary='فقط تعداد سطرها را از دو به سه ببرید. نسخهٔ بدون keepdim را در هر دو حالت امتحان کنید و پیام خطا یا میانگین سطرهای خروجی را ثبت کنید.',
        vary_code='''for matrix in [x, square]:
    try:
        wrong = matrix-matrix.mean(1)
    except RuntimeError as error:
        print("Visible failure:", error)
    else:
        print("Silent error; row means:", wrong.mean(1).tolist())
''',
        debug='میانگین (3,) با ستون‌ها هم‌تراز می‌شود. تابع expand_row_means باید میانگین‌ها را با شکل (B,1) برگرداند، نه با تکرار فیزیکی برای همهٔ ستون‌ها.',
        bug_code='''wrong_means = square.mean(1)
wrong = square-wrong_means
print("Wrong means shape:", tuple(wrong_means.shape), "Output row 0:", wrong[0])
assert not torch.allclose(wrong.mean(1), torch.zeros(3))
''',
        fix='''def expand_row_means(x):
    # TODO: return one mean per row with its column axis retained
    return None
''',
        fix_check=_check('expand_row_means(square)', 'assert tuple(result.shape) == (3, 1)\nassert torch.equal(result, torch.tensor([[2.], [5.], [8.]]))\nassert tuple(expand_row_means(x).shape) == (2, 1)', True),
        fix_solution='''def expand_row_means(x):
    return x.mean(dim=1, keepdim=True)
''',
        connection='LayerNorm در mini_gpt/transformer.py ویژگی‌های هر موقعیت را جدا نرمال می‌کند. اینجا فقط کم‌کردن میانگین را می‌سازیم؛ هنوز واریانس و Parameterهای آن لایه را اضافه نکرده‌ایم.',
        takeaway='کدام آزمون معنای محورها را سنجید، حتی وقتی Python هیچ خطایی نداد؟'),

    '16-reshape': dict(
        title='هم‌شکل ولی نه هم‌معنا',
        goal='گروه‌بندی دوباره را از جابه‌جایی محور با عددهای متمایز جدا کنید.',
        prerequisite='13 تا15؛ reshape، transpose و squeeze در درس جاری.',
        predict='reshape(3,2) و Transpose یک جدول ۲×۳ هر دو چه شکلی دارند؟ عدد ۴ در هرکدام کجا قرار می‌گیرد؟',
        setup='''import torch
small = torch.arange(1, 7).reshape(2, 3)
x = torch.arange(24).reshape(2, 3, 4)
print("Unique source values:", small)
''',
        task='تابع swap_time_features(x) ورودی (B,T,C) را به (B,C,T) ببرد، به‌طوری که عدد هر نمونه و موقعیت واقعاً جابه‌جا شود؛ فقط reshape نکنید.',
        starter='''def swap_time_features(x):
    # TODO: exchange axes 1 and 2
    return None
''',
        check=_check('swap_time_features(x)', 'assert tuple(result.shape) == (2, 4, 3)\nassert result[1, 2, 0].item() == x[1, 0, 2].item()\nassert torch.equal(swap_time_features(result), x)\nassert not torch.equal(result, x.reshape(2, 4, 3))'),
        solution='''def swap_time_features(x):
    return x.transpose(1, 2)
''',
        vary='فقط عملیات را عوض کنید، نه داده را: یک بار reshape و یک بار transpose. جدول عددها را کنار Shape چاپ کنید.',
        vary_code='''for name, value in [("regrouped", small.reshape(3, 2)), ("transposed", small.T)]:
    print(name, tuple(value.shape), value.tolist())
''',
        debug='squeeze بدون dim در Batch تک‌نمونه‌ای، محور Batch را هم حذف می‌کند. تابع remove_middle_unit ورودی (B,1,C) را به (B,C) ببرد؛ برای B=1 نیز Batch بماند.',
        bug_code='''single = torch.arange(3).reshape(1, 1, 3)
wrong = single.squeeze()
print("Lost batch axis:", tuple(wrong.shape))
assert tuple(wrong.shape) == (3,)
''',
        fix='''def remove_middle_unit(x):
    # TODO: remove only axis 1
    return None
''',
        fix_check=_check('remove_middle_unit(single)', 'assert tuple(result.shape) == (1, 3)\nassert torch.equal(result, torch.tensor([[0, 1, 2]]))\nassert tuple(remove_middle_unit(torch.zeros(2, 1, 4)).shape) == (2, 4)', True),
        fix_solution='''def remove_middle_unit(x):
    return x.squeeze(dim=1)
''',
        connection='تقسیم و اتصال Headها در mini_gpt/attention.py هم reshape دارد و هم transpose. این دو را نمی‌توان صرفاً چون خروجی هم‌شکل است جای هم گذاشت.',
        takeaway='چرا استفاده از عددهای متمایز برای این آزمون از یک Tensor پر از صفر بهتر است؟'),

    '17-autograd': dict(
        title='مشتق خودکار؛ وزن هنوز ثابت است',
        goal='ثبت Graph، محاسبهٔ Gradient و تغییر وزن را با سه شاهد جدا بررسی کنید.',
        prerequisite='12-chain و کار با Tensorهای اعشاری؛ Autograd در درس جاری.',
        predict='پس از backward برای (2w-5)² در w=1، مقدار w چیست و w.grad چیست؟ اجرای تازهٔ دوباره بدون پاک‌کردن Gradient چه می‌کند؟',
        setup='''import torch
torch.set_num_threads(1)
print("Reference: w=1, input=2, target=5")
''',
        task='تابع inspect_gradient(value) Tensor تک‌عنصری تازه با requires_grad=True بسازد، Loss=(2w-5)² را backward کند و tupleِ عددی (loss, gradient, unchanged_weight) برگرداند. هیچ update انجام ندهید.',
        starter='''def inspect_gradient(value):
    # TODO: return three Python numbers after backward
    return None
''',
        check=_check('inspect_gradient(1.)', 'assert result == (9., -12., 1.)\nassert inspect_gradient(2.) == (1., -4., 2.)\nassert inspect_gradient(2.5) == (0., 0., 2.5)'),
        solution='''def inspect_gradient(value):
    w = torch.tensor(float(value), requires_grad=True)
    loss = (2*w-5)**2
    loss.backward()
    return loss.item(), w.grad.item(), w.item()
''',
        vary='فقط پاک‌کردن Gradient بین دو اجرای تازه را روشن و خاموش کنید؛ وزن در هر دو آزمایش ثابت بماند.',
        vary_code='''for clear in [False, True]:
    w = torch.tensor(1., requires_grad=True)
    values = []
    for _ in range(2):
        if clear:
            w.grad = None
        ((2*w-5)**2).backward()
        values.append(w.grad.item())
    print("clear:", clear, "gradients:", values, "weight:", w.item())
''',
        debug='item برای گزارش است؛ عدد Python مسیر مشتق ندارد. تابع connected_loss باید Loss تنسوری را برگرداند تا فراخواننده بتواند backward کند.',
        bug_code='''w = torch.tensor(1., requires_grad=True)
wrong = ((2*w-5)**2).item()
try:
    wrong.backward()
except AttributeError as error:
    print("Expected graph loss:", error)
else:
    raise AssertionError("A Python number has no backward")
''',
        fix='''def connected_loss(w):
    # TODO: keep the tensor connected to w
    return None
''',
        fix_check=_check('connected_loss(w)', 'assert isinstance(result, torch.Tensor) and result.requires_grad\nw.grad = None\nresult.backward()\nassert w.grad.item() == -12\nfresh = torch.tensor(2., requires_grad=True)\nconnected_loss(fresh).backward()\nassert fresh.grad.item() == -4', True),
        fix_solution='''def connected_loss(w):
    return (2*w-5)**2
''',
        connection='mini_gpt/train.py مقدار item را برای گزارش می‌گیرد، اما backward را روی خود Tensor زیان اجرا می‌کند. همان جدایی ساده، در مدل بزرگ هم ضروری است.',
        takeaway='اگر Gradient درست باشد ولی وزن عوض نشود، آیا مشکل الزاماً در Autograd است؟'),

    '18-module': dict(
        title='از فرمول Affine تا Parameter ثبت‌شده',
        goal='محاسبهٔ لایه و ثبت وزن در Module را جدا بسازید و بررسی کنید.',
        prerequisite='07-matmul،17-autograd و nn.Module در درس جاری.',
        predict='با W سه‌در‌دو، ورودی یک‌در‌دو و Bias سه‌تایی، خروجی چند عدد دارد؟ آیا هر Tensor دارای Gradient خودکار در parameters پیدا می‌شود؟',
        setup='''import torch
from torch import nn
x = torch.tensor([[1., 2.]])
weight = torch.tensor([[0.5, -0.25], [1., 0.], [0., 1.]])
bias = torch.tensor([1., -2., 0.])
print("Input and weight shapes:", x.shape, weight.shape)
''',
        task='تابع affine(x,weight,bias) محاسبهٔ nn.Linear را با @، Transpose و جمع بنویسد. وزن به شکل (out,in) است؛ کپی یک nn.Linear تازه با وزن متفاوت پاسخ این تمرین نیست.',
        starter='''def affine(x, weight, bias):
    # TODO: use the supplied weight and bias
    return None
''',
        check=_check('affine(x, weight, bias)', 'assert torch.equal(result, torch.tensor([[1., -1., 2.]]))\nassert tuple(affine(x.repeat(2, 1), weight, bias).shape) == (2, 3)\nassert torch.equal(affine(torch.zeros_like(x), weight, bias), bias[None])'),
        solution='''def affine(x, weight, bias):
    return x @ weight.T+bias
''',
        vary='فقط Bias را صفر کنید؛ وزن و ورودی همان باشند. اختلاف خروجی‌ها باید برابر همان Bias باشد، نه وابسته به تعداد نمونه‌ها. این مقایسه از تابع آمادهٔ F.linear استفاده می‌کند؛ بازسازی محاسبه با @ همچنان کار شما در تمرین است.',
        vary_code='''from torch.nn import functional as F

with_bias = F.linear(x, weight, bias)
without_bias = F.linear(x, weight)
print("Difference:", with_bias-without_bias)
assert torch.equal(with_bias-without_bias, bias[None])
''',
        debug='Tensor معمولی حتی با requires_grad=True به‌عنوان Parameter ثبت نمی‌شود. تابع registered_affine یک nn.Module با weight و bias ثبت‌شده و forward درست برگرداند؛ از وزن‌های ورودی کپی مستقل بگیرید.',
        bug_code='''broken = nn.Module()
broken.weight = weight.clone().requires_grad_()
print("Unregistered parameters:", list(broken.parameters()))
assert list(broken.parameters()) == []
''',
        fix='''def registered_affine(weight, bias):
    # TODO: return a callable Module with registered parameters
    return None
''',
        fix_check=_check('registered_affine(weight, bias)', 'assert sum(p.numel() for p in result.parameters()) == 9\nassert set(dict(result.named_parameters())) == {"weight", "bias"}\nassert torch.equal(result(x), torch.tensor([[1., -1., 2.]]))\nassert result.weight.data_ptr() != weight.data_ptr()\nresult(x).sum().backward()\nassert result.weight.grad is not None and result.bias.grad is not None', True),
        fix_solution='''def registered_affine(weight, bias):
    class Affine(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(weight.clone())
            self.bias = nn.Parameter(bias.clone())
        def forward(self, x):
            return x @ self.weight.T+self.bias
    return Affine()
''',
        connection='head در mini_gpt/stages/v1.py از nn.Linear استفاده می‌کند. محاسبهٔ این لایه را ساختید و دیدید چرا ثبت Parameter، موضوعی جدا از درست‌بودن فرمول است.',
        takeaway='کدام آزمون فرمول را بررسی کرد و کدام آزمون امکان پیدا‌کردن وزن‌ها توسط Optimizer را؟'),

    '19-network': dict(
        title='دو ورودی برابرند یا متفاوت؟',
        goal='شبکهٔ XOR و یک گام آموزش را خودتان بنویسید و نقش Activation را جدا بسنجید.',
        prerequisite='17-autograd و18-module؛ Cross Entropy و Adam در درس جاری.',
        predict='آیا زیادکردن تعداد لایه‌های Affine، بدون Activation، دو قطر مربع XOR را با یک مرز خطی جدا می‌کند؟',
        setup='''import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1)
torch.manual_seed(42)
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
print("Features:", x.tolist(), "Targets:", targets.tolist())
''',
        task='build_network(width) شبکهٔ Linear(2,width)، Tanh و Linear(width,2) بسازد. train_step(model,optimizer,x,targets) گرادیان قبلی را پاک کند، Cross-Entropy را مستقیماً از Logits محاسبه و backward کند و یک step انجام دهد؛ Loss پیش از update را به‌صورت float برگرداند.',
        starter='''def build_network(width):
    # TODO: return the three-layer Sequential
    return None

def train_step(model, optimizer, x, targets):
    # TODO: one complete update; return pre-update loss as a float
    return None
''',
        check=_check('build_network(8)', '''assert sum(p.numel() for p in result.parameters()) == 42
assert tuple(result(x).shape) == (4, 2)
torch.manual_seed(42)
model = build_network(8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
before = [p.detach().clone() for p in model.parameters()]
first = train_step(model, optimizer, x, targets)
if first is None:
    return False
assert isinstance(first, float)
assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
for _ in range(399):
    train_step(model, optimizer, x, targets)
assert torch.equal(model(x).argmax(-1), targets)
assert F.cross_entropy(model(x), targets).item() < first'''),
        solution='''def build_network(width):
    return nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, 2))

def train_step(model, optimizer, x, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), targets)
    loss.backward()
    optimizer.step()
    return loss.item()
''',
        vary='فقط وجود Activation را تغییر دهید و وزن‌ها را ثابت نگه دارید. خروجی در نقطهٔ میانی دو ورودی را با میانگین خروجی‌های آن‌ها مقایسه کنید؛ تبدیل Affine این برابری را حفظ می‌کند، اما تبدیل غیرخطی الزاماً نه.',
        vary_code='''for nonlinear in [False, True]:
    torch.manual_seed(42)
    candidate = nn.Sequential(nn.Linear(2, 8), nn.Tanh() if nonlinear else nn.Identity(), nn.Linear(8, 2))
    midpoint = (x[0]+x[3])/2
    difference = candidate(midpoint)-(candidate(x[0])+candidate(x[3]))/2
    print("Activation:", nonlinear, "Midpoint difference:", difference.detach().tolist())
''',
        debug='در قرارداد هدف تک‌کلاسه، شناسه‌ها باید long باشند. تابع class_ids فقط فهرست intهای نامنفی را بپذیرد و Tensor نوع long برگرداند؛ float را با گردکردن پنهان نکنید.',
        bug_code='''try:
    F.cross_entropy(torch.zeros(4, 2), targets.float())
except RuntimeError as error:
    print("Expected target dtype failure:", error)
else:
    raise AssertionError("These class IDs must be integer tensors")
''',
        fix='''def class_ids(values):
    # TODO: validate before converting
    return None
''',
        fix_check=_check('class_ids([0, 1, 1, 0])', '''assert result.dtype == torch.long and torch.equal(result, targets)
for invalid in ([0.5, 1], [True, 0], [-1]):
    try:
        class_ids(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Do not silently round or reinterpret class IDs")''', True),
        fix_solution='''def class_ids(values):
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Nonnegative integer IDs required")
    return torch.tensor(values, dtype=torch.long)
''',
        connection='mini_gpt/experiments.py آزمایش network را با همین XOR اجرا می‌کند. FFN در mini_gpt/transformer.py نیز دو تبدیل Affine و یک Activation دارد، با ابعاد و تابع متفاوت.',
        takeaway='در این آزمایش کدام شاهد دربارهٔ ظرفیت است و چرا چهار پاسخ درست، شاهد تعمیم به دادهٔ ندیده نیست؟'),

    '20-loader': dict(
        title='پنج نمونه، سه Batch و یک آخرِ کوچک‌تر',
        goal='Dataset را خودتان بسازید و پوشش همهٔ نمونه‌ها را بررسی کنید.',
        prerequisite='14-index-device و19-network؛ Dataset و DataLoader درس جاری.',
        predict='پنج نمونه با batch_size=2 چند Batch می‌سازند؟ آخرین نمونه با drop_last=True چه می‌شود؟',
        setup='''import torch
from torch.utils.data import Dataset, DataLoader
print("Indices to visit:", list(range(5)))
''',
        task='تابع make_dataset(n) یک Dataset برگرداند: ورودی نمونهٔ i برابر Tensor اعشاری [i] و هدف، Tensor صحیحِ تکی i%2 باشد. __len__ باید n باشد و اندیس بیرون [0,n) خطای IndexError بدهد.',
        starter='''def make_dataset(n):
    # TODO: return a Dataset implementing __len__ and __getitem__
    return None
''',
        check=_check('make_dataset(5)', '''assert len(result) == 5
features, target = result[3]
assert features.tolist() == [3.] and features.dtype == torch.float32
assert target.shape == () and target.dtype == torch.long and target.item() == 1
batches = list(DataLoader(result, batch_size=2, shuffle=False, num_workers=0))
assert [len(features) for features, _ in batches] == [2, 2, 1]
assert torch.cat([features[:, 0] for features, _ in batches]).tolist() == [0, 1, 2, 3, 4]
try:
    result[5]
except IndexError:
    pass
else:
    raise AssertionError("Dataset index must be bounded")'''),
        solution='''def make_dataset(n):
    class Examples(Dataset):
        def __len__(self):
            return n
        def __getitem__(self, index):
            if not 0 <= index < n:
                raise IndexError("Index outside dataset")
            return torch.tensor([float(index)]), torch.tensor(index % 2, dtype=torch.long)
    return Examples()
''',
        vary='فقط drop_last را عوض کنید. برای این مقایسه از فهرست Tensor آماده استفاده می‌کنیم تا به پیاده‌سازی تمرین وابسته نباشد.',
        vary_code='''samples = [torch.tensor([i]) for i in range(5)]
for drop in [False, True]:
    loader = DataLoader(samples, batch_size=2, shuffle=False, drop_last=drop, num_workers=0)
    print("drop_last:", drop, "visited:", [batch.flatten().tolist() for batch in loader])
''',
        debug='حلقهٔ خراب با floor division آخرین دسته را فراموش می‌کند. تابع batch_ranges(n,size) جفت‌های (start,stop) بسازد تا هر اندیس دقیقاً یک بار پوشش داده شود؛ size مثبت است.',
        bug_code='''wrong_ranges = [(i*2, (i+1)*2) for i in range(5//2)]
print("Broken ranges:", wrong_ranges)
assert wrong_ranges[-1][1] == 4
''',
        fix='''def batch_ranges(n, size):
    # TODO: keep the incomplete final batch
    return None
''',
        fix_check=_check('batch_ranges(5, 2)', 'assert result == [(0, 2), (2, 4), (4, 5)]\nassert batch_ranges(4, 2) == [(0, 2), (2, 4)]\nassert batch_ranges(1, 8) == [(0, 1)]', True),
        fix_solution='''def batch_ranges(n, size):
    return [(start, min(start+size, n)) for start in range(0, n, size)]
''',
        connection='ارزیابی در mini_gpt/evaluate.py همهٔ نمونه‌های NextTokenDataset را با DataLoader می‌بیند. آموزش تصادفی پروژه قرارداد دیگری دارد؛ هر تعداد step الزاماً یک Epoch نیست.',
        takeaway='اگر آخرین Batch کوچک‌تر باشد، چرا نباید Loss آن را هم‌وزن یک Batch بزرگ میانگین بگیرید؟'),
})
