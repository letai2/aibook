"""Seven independent model-to-system laboratories; solutions stay author-only."""


def _check(first, assertions, repair=False):
    name = "test_repair" if repair else "test_exercise"
    flag = "repair_complete" if repair else "exercise_complete"
    lines = [f"def {name}():", f"    result = {first}",
             "    if result is None:", "        return False"]
    lines.extend("    "+line for line in assertions.strip().splitlines())
    lines += ["    return True", "", f"{flag} = {name}()",
              f"print('PASS' if {flag} else 'INCOMPLETE: implement the TODO and rerun')"]
    return "\n".join(lines)


EXERCISES = {
    "68-context": dict(
        title="بودجهٔ سند با بودجهٔ کل ورودی یکی نیست",
        goal="هزینهٔ پرسش، قالب و پاسخِ رزروشده را وارد تصمیم انتخاب شاهد کنید.",
        prerequisite="26-positions و تولید Mini-GPT؛ درس 68-context را بخوانید. آزمایش فعلی فقط Python است.",
        predict="اگر متن یک سند ۱۰ کاراکتر باشد، آیا پنجرهٔ ۱۰تایی برای پرسش، شناسهٔ منبع و آن سند کافی است؟ برای پاسخ چه مقدار جا مانده است؟",
        setup='''import json
from mini_gpt.context import ContextItem, build_context

def render(question, items):
    return json.dumps({"question": question, "context": items},
                      ensure_ascii=False, separators=(",", ":"))

question = "Where?"
items = [{"id": "guide", "text": "Use Jupyter."}]
print("Document length:", len(items[0]["text"]))
print("Complete input:", render(question, items), len(render(question, items)))
''',
        task="تابع `fits_input(question, items, window, reserved, counter)` را بنویسید. با `render` رشتهٔ کامل را بسازید و فقط وقتی طولِ محاسبه‌شده با `counter` به‌اضافهٔ `reserved` حداکثر `window` است، `True` بدهید. مقدارهای عددی این تمرین معتبر و نامنفی‌اند؛ `counter` تعداد Token همان قالب را می‌دهد.",
        starter='''def fits_input(question, items, window, reserved, counter):
    # TODO: count the complete serialized input, not only the document text
    return None
''',
        check=_check('fits_input(question, items, 500, 20, len)', '''assert result is True
size = len(render(question, items))
assert fits_input(question, items, size+20, 20, len) is True
assert fits_input(question, items, size+19, 20, len) is False
assert fits_input(question, [], 10, 0, len) is False
assert fits_input(question, items, 9, 2, lambda text: 7) is True
assert fits_input(question, items, 8, 2, lambda text: 7) is False'''),
        solution='''def fits_input(question, items, window, reserved, counter):
    return counter(render(question, items))+reserved <= window
''',
        vary="فقط سقف پنجره را عوض کنید و پرسش، شاهد و سهم پاسخ را ثابت بگذارید. از تابع واقعی پروژه استفاده کنید؛ به شناسه‌های حذف‌شده هم نگاه کنید، نه فقط کوتاهی رشته.",
        vary_code='''evidence = [ContextItem("guide", "Use Jupyter."), ContextItem("note", "CPU works.")]
for window in [90, 130, 220]:
    pack = build_context("Where?", evidence, max_tokens=window, reserve_tokens=20, count_tokens=len)
    print(window, pack.token_count, pack.included_ids, pack.omitted_ids)
''',
        debug="اگر نخستین سند جا نشود، توقف کامل حلقه می‌تواند سند کوتاه‌تر بعدی را هم حذف کند. تابع `select_whole_items` قطعه‌ها را به ترتیب امتحان کند، قطعهٔ بزرگ را رد کند و بررسی بقیه را ادامه دهد. `budget` فقط بودجهٔ ورودی است؛ پرسشِ خالی از شاهد در آن جا می‌شود.",
        bug_code='''priority = [{"id": "long", "text": "x"*400}, {"id": "short", "text": "yes"}]
wrong = []
for item in priority:
    if len(render(question, wrong+[item])) > 100:
        break
    wrong.append(item)
print("Broken early stop:", wrong)
assert wrong == []
''',
        fix='''def select_whole_items(question, items, budget, counter):
    # TODO: return selected IDs, skipping oversized items without stopping
    return None
''',
        fix_check=_check('select_whole_items(question, priority, 100, len)', '''assert result == ["short"]
assert select_whole_items(question, priority, 1000, len) == ["long", "short"]
assert select_whole_items(question, [], 100, len) == []
assert priority[0]["text"] == "x"*400''', True),
        fix_solution='''def select_whole_items(question, items, budget, counter):
    selected = []
    for item in items:
        if counter(render(question, selected+[item])) <= budget:
            selected.append(item)
    return [item["id"] for item in selected]
''',
        connection="`mini_gpt/context.py` همین تصمیم را با `ContextItem` و شمارندهٔ واقعی Tokenizer انجام می‌دهد. به‌جای فرض تعداد واژه‌ها، برای Mini-GPT می‌توان `lambda text: len(tokenizer.encode(text))` داد.",
        takeaway="کدام شاهد نشان می‌دهد حذف یک سند تصمیم برنامه بوده است، نه تغییر حافظه یا وزن مدل؟",
    ),
    "69-chunks": dict(
        title="پاسخ درست روی مرز دو قطعه افتاده است",
        goal="اندیس‌های قطعه را خودتان بسازید و ثابت کنید متن اصلی و هویت منبع گم نشده‌اند.",
        prerequisite="67-rag و 68-context؛ برش رشته و `re.finditer` در Python. کلمه‌های این تمرین Token مدل نیستند.",
        predict="با اندازهٔ چهار کلمه و هم‌پوشانی صفر، عبارت `four five` در کدام قطعه کامل است؟ هم‌پوشانی یک چه چیزی را عوض می‌کند؟",
        setup='''import re
from mini_gpt.retrieval import Document, chunk_document

text = "one two three four five six seven"
print("Word spans:", [(m.group(), m.start(), m.end()) for m in re.finditer(r"\\S+", text)])
''',
        task="تابع `chunk_spans(text, size, overlap)` جفت‌های `(start,end)` قطعه‌ها را برگرداند. از همان الگوی `re.finditer` در Cell آماده‌سازی استفاده کنید؛ هر قطعه حداکثر `size` کلمه داشته باشد و شروع‌ها `size-overlap` جلو بروند. وقتی قطعه به آخر متن رسید متوقف شوید. شرط `0 <= overlap < size` را بررسی کنید؛ اندازه‌ها `int` هستند. متن خالی فهرست خالی بدهد.",
        starter='''def chunk_spans(text, size, overlap):
    # TODO: preserve exact source-string offsets
    return None
''',
        check=_check('chunk_spans(text, 4, 1)', '''assert [text[a:b] for a,b in result] == ["one two three four", "four five six seven"]
assert [text[a:b] for a,b in chunk_spans(text,4,0)] == ["one two three four", "five six seven"]
assert chunk_spans("",4,1) == []
assert chunk_spans("  الف\\tب  ",8,0) == [(2,7)]
try:
    chunk_spans(text,4,4)
except ValueError:
    pass
else:
    raise AssertionError("The stride must stay positive")'''),
        solution='''def chunk_spans(text, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    words = list(re.finditer(r"\\S+", text))
    result = []
    for first in range(0, len(words), size-overlap):
        last = min(first+size, len(words))
        result.append((words[first].start(), words[last-1].end()))
        if last == len(words):
            break
    return result
''',
        vary="فقط هم‌پوشانی را صفر، یک و دو بگذارید. وجود عبارت کامل و مجموع تعداد کلمه‌های قطعه‌ها را مقایسه کنید؛ افزایش شمارِ ذخیره‌شده، افزایش طول سند نیست.",
        vary_code='''document = Document("guide", text)
for overlap in [0, 1, 2]:
    chunks = chunk_document(document, chunk_words=4, overlap_words=overlap)
    print(overlap, [c.text for c in chunks], "stored words:", sum(len(c.text.split()) for c in chunks),
          "phrase retained:", any("four five" in c.text for c in chunks))
''',
        debug="ذخیرهٔ متن قطعه بدون بررسی منبع می‌تواند ارجاع غلط بسازد. تابع `verified_source` متن قطعه را فقط وقتی برگرداند که شناسهٔ سند موجود و برش `start:end` دقیقاً برابر متن قطعه باشد؛ وگرنه `ValueError` بدهید.",
        bug_code='''from mini_gpt.retrieval import Chunk
sources = {"a": "CPU only", "b": "GPU only"}
wrong_chunk = Chunk("bad", "b", "CPU only", 0, 8)
print("Claimed quote:", wrong_chunk.text, "Actual source:", sources[wrong_chunk.document_id])
''',
        fix='''def verified_source(chunk, sources):
    # TODO: validate source identity and the entire quoted span
    return None
''',
        fix_check=_check('verified_source(Chunk("a:0-8","a","CPU only",0,8), sources)', '''assert result == "CPU only"
assert verified_source(Chunk("a:0-3","a","CPU",0,3),sources) == "CPU"
for bad in [wrong_chunk, Chunk("missing","z","CPU",0,3), Chunk("range","a","CPU",-8,3)]:
    try:
        verified_source(bad,sources)
    except ValueError:
        pass
    else:
        raise AssertionError("A wrong or invalid source span must fail")''', True),
        fix_solution='''def verified_source(chunk, sources):
    source = sources.get(chunk.document_id)
    if source is None or not 0 <= chunk.start < chunk.end <= len(source):
        raise ValueError("Missing source or invalid span")
    if source[chunk.start:chunk.end] != chunk.text:
        raise ValueError("The quote does not match its source")
    return chunk.text
''',
        connection="`Chunk` در `mini_gpt/retrieval.py` متن، شناسهٔ سند و اندیس‌ها را با هم نگه می‌دارد. این شاهد بعداً وارد Context و Citation می‌شود؛ رشتهٔ جداشده بدون هویت کافی نیست.",
        takeaway="چرا دو Chunk دارای متن مشترک را نباید دو تأیید مستقل برای یک ادعا شمرد؟",
    ),
    "70-vectors": dict(
        title="بردار شمارشی و بردار آموزش‌دیده را با هم اشتباه نگیریم",
        goal="Cosine و رتبه‌بندی را پیاده کنید و تغییر واقعی یک Embedding را روی جفت‌های آموزشی ببینید.",
        prerequisite="06-dot، 19-network، 25-embedding و 69-chunks؛ PyTorch از همان محیط پروژه استفاده می‌شود.",
        predict="پرسش «ذخیره» با سند «نگهداری» هیچ واژهٔ مشترکی ندارد. چرا عددهای تصادفی به‌تنهایی این شکست را حل نمی‌کنند؟ پس از آموزش همین جفت، دربارهٔ واژه‌های ندیده چه ادعایی هنوز نمی‌توان کرد؟",
        setup='''import math
from collections import Counter
from mini_gpt.retrieval import train_tiny_embeddings, VectorIndex, Document, chunk_document

words, initial, learned, initial_loss, final_loss = train_tiny_embeddings()
print("Provided related pairs:", words[:2], words[2:])
print("Initial vectors:", initial, "Learned vectors:", learned)
print("Loss on the provided pairs, not held-out quality:", initial_loss, final_loss)
assert final_loss < initial_loss
''',
        task="تابع `rank_vectors(query, rows, k)` شمارهٔ حداکثر `k` سطر را با Cosine نزولی برگرداند. خودتان Dot product و Norm را حساب کنید؛ امتیاز Vector صفر را صفر بگذارید و امتیازهای صفر یا منفی را برنگردانید. در تساوی، شمارهٔ کوچک‌تر جلوتر باشد. اندازه‌ها سازگار، عددها متناهی و `k` مثبت است.",
        starter='''def rank_vectors(query, rows, k):
    # TODO: calculate cosine scores, then rank row indices
    return None
''',
        check=_check('rank_vectors([1.,0.], [[2.,0.],[1.,1.],[0.,0.],[-1.,0.]], 4)', '''assert result == [0,1]
assert rank_vectors([1,0],[[1,0],[4,0]],1) == [0]
assert rank_vectors([0,0],[[1,0],[0,1]],2) == []
assert rank_vectors(learned[0],[learned[1],learned[2],learned[3]],1) == [0]
assert rank_vectors([0,1],[[1,0],[0,1]],2) == [1]'''),
        solution='''def rank_vectors(query, rows, k):
    qnorm = math.sqrt(sum(value*value for value in query))
    scored = []
    for index, row in enumerate(rows):
        norm = math.sqrt(sum(value*value for value in row))
        score = sum(a*b for a,b in zip(query,row))/(qnorm*norm) if qnorm and norm else 0.
        if score > 0:
            scored.append((index,score))
    scored.sort(key=lambda item: (-item[1],item[0]))
    return [index for index,score in scored[:k]]
''',
        vary="فقط تعداد گام آموزش جدول کوچک را از صفر به صد ببرید؛ مقدار آغاز و جفت‌ها ثابت‌اند. امتیاز جفت مرتبط و نامرتبط را جدا ببینید. سپس شکست همان پرسش در Index شمارشی را ثبت کنید؛ Index با این آموزش عوض نشده است.",
        vary_code='''from mini_gpt.retrieval import cosine_similarity
for steps in [0, 100]:
    names, before, after, first, final = train_tiny_embeddings(steps=steps)
    print(steps, "related:", cosine_similarity(after[0],after[1]),
          "unrelated:", cosine_similarity(after[0],after[2]), "loss:",final)
chunks = chunk_document(Document("guide", "نگهداری"),chunk_words=4,overlap_words=0)
index = VectorIndex(chunks)
assert index.search("ذخیره") == []
print("Count-vector retrieval for unseen synonym:", index.search("ذخیره"))
''',
        debug="دو Vector تنها وقتی قابل مقایسه‌اند که ستون‌های متناظر معنای یکسان داشته باشند. کد خراب Vocabulary پرسش را جدا مرتب می‌کند. تابع `encode_shared(text, vocabulary)` شمارش واژه‌های جداشده با فاصله را دقیقاً به ترتیب Vocabulary داده‌شده برگرداند؛ واژه‌های بیرون آن سهمی ندارند.",
        bug_code='''vocabulary = ["ذخیره", "مدل"]
document_vector = [1, 0]
wrong_query_order = list(reversed(vocabulary))
wrong_query = [int(word == "ذخیره") for word in wrong_query_order]
print("Document axes:", vocabulary, "Query axes:", wrong_query_order, "Wrong query:", wrong_query)
assert sum(a*b for a,b in zip(document_vector,wrong_query)) == 0
''',
        fix='''def encode_shared(text, vocabulary):
    # TODO: preserve the supplied shared column order
    return None
''',
        fix_check=_check('encode_shared("ذخیره", vocabulary)', '''assert result == [1,0]
assert encode_shared("مدل ذخیره مدل",vocabulary) == [1,2]
assert encode_shared("ناشناخته",vocabulary) == [0,0]
assert encode_shared("مدل",list(reversed(vocabulary))) == [1,0]''', True),
        fix_solution='''def encode_shared(text, vocabulary):
    counts = Counter(text.split())
    return [counts[word] for word in vocabulary]
''',
        connection="Index پروژه شمارشی است؛ جدول چهارواژه‌ای جدا و واقعاً با SGD آموزش دیده است. بهبود روی همین جفت‌ها شاهد تعمیم نیست و هیچ Weight از Mini-GPT با این آزمایش عوض نمی‌شود.",
        takeaway="در هر نمایش، منبع معنا کجاست: قرارداد ستون‌ها، برچسب‌های آموزش، یا صرفاً Shape بردار؟",
    ),
    "71-grounding": dict(
        title="یک Citation واقعی کنار یک نقل غلط",
        goal="هویت منبع و وفاداری نقل مستقیم را جدا از درستی علمی پاسخ بررسی کنید.",
        prerequisite="68-context تا 70-vectors؛ بازیابی، Chunk و امتیاز ارتباط. ابزار بررسی این دفتر معنای آزاد جمله‌ها را نمی‌فهمد.",
        predict="اگر شناسهٔ منبع درست باشد ولی متن نقل‌شده فقط در منبع دیگری باشد، کدام آزمون باید شکست بخورد؟ آیا گذشتن از آن آزمون، حقیقت بیرونی را تضمین می‌کند؟",
        setup='''from mini_gpt.retrieval import Document, chunk_document, keyword_search, answer_from_hits

sources = {"cpu": "Use CPU for this exercise.", "gpu": "This separate experiment needs a GPU."}
quotes = [("cpu", "Use CPU"), ("gpu", "needs a GPU")]
print("Sources:", sources, "Claims to trace:", quotes)
''',
        task="تابع `checked_citations(quotes, sources)` را بنویسید. هر جفت `(source_id, quote)` باید منبع موجود و نقل غیرخالی داشته باشد و نقل عیناً در همان منبع پیدا شود؛ وگرنه `ValueError` بدهید. خروجی `tuple` شناسه‌های یکتا به ترتیب نخستین استفاده باشد. این قرارداد مخصوص نقل مستقیم است، نه بازنویسی آزاد.",
        starter='''def checked_citations(quotes, sources):
    # TODO: check the association between each quote and its own source
    return None
''',
        check=_check('checked_citations(quotes, sources)', '''assert result == ("cpu","gpu")
assert checked_citations([("cpu","CPU"),("cpu","exercise")],sources) == ("cpu",)
assert checked_citations([],sources) == ()
for invalid in [[("gpu","Use CPU")],[("missing","CPU")],[("cpu","")]]:
    try:
        checked_citations(invalid,sources)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid source or unfaithful quote")'''),
        solution='''def checked_citations(quotes, sources):
    identifiers = []
    for identifier, quote in quotes:
        if identifier not in sources or not quote or quote not in sources[identifier]:
            raise ValueError("Invalid source or unfaithful quote")
        if identifier not in identifiers:
            identifiers.append(identifier)
    return tuple(identifiers)
''',
        vary="فقط سند دوم را به جمله‌ای متعارض با سند اول تغییر دهید. روش نقل مستقیم را اجرا کنید و هر دو منبع را ببینید؛ این تابع تعارض را خودکار حل نمی‌کند و خروجی آن را پاسخ تولیدشدهٔ مدل ننامید.",
        vary_code='''docs = [Document("old","تمرین فقط با CPU اجرا می‌شود."),
        Document("new","تمرین فقط با GPU اجرا می‌شود.")]
chunks = [c for d in docs for c in chunk_document(d,chunk_words=20,overlap_words=0)]
answer = answer_from_hits(keyword_search("تمرین",chunks))
print("Extractive passages; conflict not resolved:",answer.text)
assert len(answer.citations) == 2
''',
        debug="وجود یک سطرِ نتیجه کافی نیست؛ ممکن است امتیازش صفر باشد. تابع `has_evidence(scores, threshold)` تنها با وجود امتیاز مثبتِ حداقل برابر آستانه، `True` بدهد. امتیازها و آستانه متناهی و آستانه نامنفی است؛ این بررسی فقط شرط ورود است، نه تضمین درستی پاسخ.",
        bug_code='''scores = [0.0]
wrong_decision = bool(scores)
print("Broken evidence decision:",wrong_decision,"Scores:",scores)
assert wrong_decision is True
''',
        fix='''def has_evidence(scores, threshold):
    # TODO: zero relevance is not evidence, even when threshold is zero
    return None
''',
        fix_check=_check('has_evidence([0.0],0.0)', '''assert result is False
assert has_evidence([],0.1) is False
assert has_evidence([0.2,0.7],0.7) is True
assert has_evidence([0.2,0.7],0.8) is False''', True),
        fix_solution='''def has_evidence(scores, threshold):
    return any(score > 0 and score >= threshold for score in scores)
''',
        connection="`GroundedAnswer` در پروژه ارجاع‌ها و حالت `abstained` را جدا نگه می‌دارد. در دستیار نهایی، همین مرز اجازه می‌دهد خرابی Retrieval را با نثر روان مدل اشتباه نگیریم.",
        takeaway="کدام خطاها را با آزمون نقل مستقیم گرفتید و کدام خطاهای معنایی هنوز خارج از توان آن‌اند؟",
    ),
    "72-history": dict(
        title="تاریخچه در برنامه هست، اما در ورودی نیست",
        goal="پیام‌های کامل را انتخاب کنید و اثر حذف یک محدودیت قدیمی را پیش از اجرای مدل ببینید.",
        prerequisite="68-context و 71-grounding؛ نقش پیام و تاریخچه در درس 72-history. هیچ آموزش تازه‌ای انجام نمی‌شود.",
        predict="اگر CPU فقط در پیام نخست آمده باشد، نگه‌داشتن دو پیام آخر چه اطلاعاتی را حذف می‌کند؟ مقدار limit=0 باید چه برگرداند؟",
        setup='''import json
from mini_gpt.memory import Message, format_history

messages = [Message("user","CPU only."), Message("assistant","Open Jupyter."),
            Message("user","Which command now?")]
print("Full history:",format_history(messages))
''',
        task="تابع `recent_messages(messages, limit)` یک list تازه از حداکثر `limit` پیام آخر، با همان ترتیب، برگرداند. صفر یعنی تاریخچهٔ خالی و مقدار منفی خطای `ValueError` است؛ `limit` در این تمرین `int` است. ورودی را تغییر ندهید و محتوا و نقش هر پیام را حفظ کنید.",
        starter='''def recent_messages(messages, limit):
    # TODO: keep complete messages, including their roles
    return None
''',
        check=_check('recent_messages(messages,2)', '''assert result == messages[1:]
assert recent_messages(messages,0) == []
assert recent_messages(messages,20) == messages
assert recent_messages(messages,20) is not messages
assert len(messages) == 3
assert all(isinstance(message,Message) for message in result)
try:
    recent_messages(messages,-1)
except ValueError:
    pass
else:
    raise AssertionError("Negative history limit")'''),
        solution='''def recent_messages(messages, limit):
    if limit < 0:
        raise ValueError("Negative history limit")
    return list(messages[-limit:]) if limit else []
''',
        vary="فقط تعداد پیام‌های انتخابی را تغییر دهید و با `format_history` واقعی رشتهٔ ورودی را ببینید. دربارهٔ وجود اطلاعات CPU نتیجه بگیرید، نه کیفیت پاسخ مدلی که هنوز اجرا نکرده‌ایم.",
        vary_code='''for count in [1,2,3]:
    text = format_history(messages[-count:])
    print(count,"CPU available:","CPU" in text,"characters:",len(text),text)
''',
        debug="چسباندن محتوا بدون نقش، گفتهٔ کاربر و پاسخ احتمالیِ دستیار را یکی می‌کند. تابع `serialize_messages` فهرست دیکشنری‌های `role` و `content` را با JSON سریال کند. آزمون قالب JSON را می‌خواند؛ فاصله‌گذاری رشته مهم نیست.",
        bug_code='''ambiguous = [Message("user","Do not use GPU."),Message("assistant","Use GPU.")]
wrong_text = " ".join(message.content for message in ambiguous)
print("Roles lost:",wrong_text)
''',
        fix='''def serialize_messages(messages):
    # TODO: retain both role and content for each message
    return None
''',
        fix_check=_check('serialize_messages(ambiguous)', '''assert json.loads(result) == [{"role":"user","content":"Do not use GPU."},
                              {"role":"assistant","content":"Use GPU."}]
special = [Message("user",'A "quoted" instruction.\\nNext line.')]
assert json.loads(serialize_messages(special))[0]["content"] == special[0].content
assert json.loads(serialize_messages([])) == []''', True),
        fix_solution='''def serialize_messages(messages):
    return json.dumps([{"role":message.role,"content":message.content} for message in messages],
                      ensure_ascii=False)
''',
        connection="`format_history` یک قالب آموزشیِ آشکار می‌سازد. مدل Mini-GPT صرفاً همین متن را می‌بیند؛ برچسب role به‌تنهایی نه آموزش Chat انجام می‌دهد و نه صلاحیت اجرای ابزار می‌دهد.",
        takeaway="حفظ همهٔ پیام‌ها در list و فرستادن همهٔ پیام‌ها به مدل چه تفاوتی دارند؟",
    ),
    "73-summary": dict(
        title="شمارهٔ منبع باقی ماند، قید مهم حذف شد",
        goal="خلاصه‌ای استخراجی با منبع بسازید و کوتاه‌شدن را از حفظ محدودیت‌ها جدا بسنجید.",
        prerequisite="72-history و 68-context؛ این آزمایش دربارهٔ روش استخراجی است، نه ادعای کیفیت خلاصه‌سازی Mini-GPT.",
        predict="اگر روش خلاصه فقط خط اول هر پیام را بردارد، آیا افزایش بودجه قید CPU در خط دوم را برمی‌گرداند؟",
        setup='''from mini_gpt.memory import Message, summarize_history

messages = [Message("user","I will run the exercise.\\nCPU only."),
            Message("assistant","Open the notebook."),
            Message("user","No installation, please.")]
print("Original messages:",messages)
''',
        task="تابع `extract_summary(messages, selected)` برای شماره‌های یک‌مبنای پیام‌ها، نخستین خط غیرخالی را بردارد و خط `number:role: text` بسازد. خروجی زوج `(text, source_numbers)` باشد؛ متن با newline به هم وصل شود و شماره‌ها یک `tuple` با ترتیب درخواست باشند. شمارهٔ تکراری یا بیرون محدوده را با `ValueError` رد کنید. این تابع فقط انتخاب را انجام می‌دهد؛ بودجه‌بندی مرحله‌ای جداست.",
        starter='''def extract_summary(messages, selected):
    # TODO: keep visible provenance while extracting selected first lines
    return None
''',
        check=_check('extract_summary(messages,[1,3])', '''text,numbers = result
assert numbers == (1,3)
assert text == "1:user: I will run the exercise.\\n3:user: No installation, please."
assert "CPU" not in text
assert extract_summary(messages,[]) == ("",())
assert extract_summary(messages,[3])[1] == (3,)
for invalid in ([0],[4],[1,1]):
    try:
        extract_summary(messages,invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid or duplicate source message")'''),
        solution='''def extract_summary(messages, selected):
    if len(set(selected)) != len(selected) or any(type(i) is not int or not 1 <= i <= len(messages) for i in selected):
        raise ValueError("Invalid or duplicate source message")
    lines = []
    for number in selected:
        message = messages[number-1]
        first = next(line.strip() for line in message.content.splitlines() if line.strip())
        lines.append(f"{number}:{message.role}: {first}")
    return "\\n".join(lines),tuple(selected)
''',
        vary="فقط بودجهٔ `max_chars` را زیاد کنید. شماره‌های پیام‌های حذف‌شده و وجود CPU را جدا ببینید. اگر CPU با بودجهٔ بیشتر هم برنگشت، محدودیت از کدام سیاست می‌آید؟",
        vary_code='''for budget in [20,60,200]:
    summary = summarize_history(messages,max_chars=budget)
    print(budget,summary.source_turns,summary.omitted_turns,
          "CPU preserved:","CPU" in summary.text,repr(summary.text))
''',
        debug="بررسی طول خلاصه، حفظ قید را نمی‌سنجد. تابع `missing_constraints(summary, required)` فهرست عبارت‌های اجباریِ غایب را با همان ترتیب برگرداند. این آزمون تطابقِ عین عبارت است؛ معنای بازنویسی یا تناقض را کامل نمی‌فهمد.",
        bug_code='''summary_text = "The learner will run the exercise."
required = ["CPU only", "No installation"]
print("Broken quality check:",len(summary_text)<100,"Required constraints:",required)
''',
        fix='''def missing_constraints(summary, required):
    # TODO: report absent constraints rather than only checking length
    return None
''',
        fix_check=_check('missing_constraints(summary_text,required)', '''assert result == required
assert missing_constraints("CPU only; No installation",required) == []
assert missing_constraints("CPU only",required) == ["No installation"]
assert missing_constraints("anything",[]) == []''', True),
        fix_solution='''def missing_constraints(summary, required):
    return [constraint for constraint in required if constraint not in summary]
''',
        connection="خلاصهٔ `mini_gpt/memory.py` به‌عنوان یک ContextItem جدا قابل استفاده است. سابقهٔ منبع و آزمون قیدها، محدودیت این خلاصه را آشکار می‌کنند؛ Parameterها همچنان ثابت‌اند.",
        takeaway="چرا `source_turns=(1,)` به معنی حفظ تمام معنای پیام نخست نیست؟",
    ),
    "74-memory": dict(
        title="ترجیح را اصلاح کردیم؛ فایل هم عوض شد؟",
        goal="جایگزینی رکورد، ذخیرهٔ صریح و انتخاب حافظهٔ مرتبط را جدا پیاده کنید.",
        prerequisite="72-history، 73-summary و 68-context؛ اطلاعات این دفتر ساختگی‌اند و فایل فقط در پوشهٔ موقت نوشته می‌شود.",
        predict="اگر مقدار device را در RAM از CPU به GPU تغییر دهید ولی دوباره ذخیره نکنید، اجرای تازه چه مقداری از فایل می‌خواند؟",
        setup='''from pathlib import Path
from tempfile import TemporaryDirectory
from mini_gpt.memory import MemoryRecord, MemoryStore
from mini_gpt.context import ContextItem

records = {"device":MemoryRecord("device","CPU","user-message-1"),
           "lesson":MemoryRecord("lesson","74-memory","user-message-2")}
correction = MemoryRecord("device","GPU","user-message-3")
print("Synthetic records, explicitly approved for this exercise:",records)
''',
        task="تابع `updated_memory(records, record)` یک دیکشنری تازه برگرداند که رکوردِ `record.key` را اضافه یا جایگزین کرده است. بقیهٔ رکوردها حفظ شوند و ورودی تغییر نکند. مقدار تازه و منبع تازه باید با هم جایگزین شوند؛ رکوردهای ورودی از نوع `MemoryRecord` هستند.",
        starter='''def updated_memory(records, record):
    # TODO: replace one keyed record in a fresh dictionary
    return None
''',
        check=_check('updated_memory(records,correction)', '''assert len(result) == 2
assert result["device"].value == "GPU" and result["device"].source == "user-message-3"
assert records["device"].value == "CPU"
assert result["lesson"] == records["lesson"]
extra = MemoryRecord("language","Persian","user-message-4")
assert len(updated_memory(records,extra)) == 3
assert updated_memory({},extra) == {"language":extra}'''),
        solution='''def updated_memory(records, record):
    result = dict(records)
    result[record.key] = record
    return result
''',
        vary="فقط ذخیرهٔ دوباره را انجام دهید؛ اصلاح Store ثابت باشد. قبل و بعد از `save` فایل را در یک Store تازه بخوانید. سپس حذف و ذخیره را هم جدا بررسی کنید؛ هیچ فایل واقعیِ کاربر تغییر نمی‌کند.",
        vary_code='''store = MemoryStore(records.values())
with TemporaryDirectory() as directory:
    path = Path(directory)/"memory.json"
    store.save(path)
    store.upsert(correction)
    print("Before saving correction:",MemoryStore.load(path).records())
    store.save(path)
    print("After saving correction:",MemoryStore.load(path).records())
    store.forget("device")
    store.save(path)
    assert all(row.key != "device" for row in MemoryStore.load(path).records())
''',
        debug="فرستادن همهٔ حافظه در هر پرسش هم هزینه دارد، هم ممکن است اطلاعات نامرتبط وارد کند. تابع `memory_items(records, keys)` فقط کلیدهای مجازِ درخواستی را به ContextItem تبدیل کند؛ شناسه `memory:` به‌اضافهٔ کلید، kind برابر `memory` و متن برابر `value + ' (source: ' + source + ')'` باشد. کلیدِ ناموجود را کنار بگذارید؛ keys یکتا است.",
        bug_code='''requested_keys = ["device"]
wrong_context = [row.value for row in records.values()]
print("Requested:",requested_keys,"Unnecessarily included:",wrong_context)
assert "74-memory" in wrong_context
''',
        fix='''def memory_items(records, keys):
    # TODO: include only requested existing records, with their source
    return None
''',
        fix_check=_check('memory_items(records,["device"])', '''assert len(result) == 1
assert result[0].id == "memory:device" and result[0].kind == "memory"
assert result[0].text == "CPU (source: user-message-1)"
assert memory_items(records,["missing"]) == []
assert [item.id for item in memory_items(records,["lesson","device"])] == ["memory:lesson","memory:device"]
assert len(records) == 2''', True),
        fix_solution='''def memory_items(records, keys):
    items = []
    for key in keys:
        if key in records:
            row = records[key]
            items.append(ContextItem("memory:"+key,row.value+" (source: "+row.source+")","memory"))
    return items
''',
        connection="`MemoryStore` فقط با فراخوانی صریح فایل می‌نویسد. در `run_assistant` فراخواننده باید کلیدهای مجاز را با `memory_keys` انتخاب کند؛ مقدار پیش‌فرض `()` هیچ رکوردی را انتخاب نمی‌کند. سپس `build_context` جاگرفتن رکوردهای منتخب را می‌سنجد. کنترل‌کننده خودش رضایت یا ارتباط را تشخیص نمی‌دهد؛ وجود فایل مساوی حضور در Context نیست.",
        takeaway="تغییر Weight، پاک‌کردن KV cache و حذف رکورد از فایل چرا سه کار متفاوت‌اند؟",
    ),
}
