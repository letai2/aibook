"""Four independent, CPU-small labs for the model-to-system boundary."""
from textwrap import dedent, indent


def _check(body, repair=False):
    name = "test_repair" if repair else "test_exercise"
    flag = "repair_complete" if repair else "exercise_complete"
    return (f"def {name}():\n" + indent(dedent(body).strip(), "    ")
            + f'\n\n{flag} = {name}()\nprint("PASS" if {flag} else "INCOMPLETE: complete the TODO first")\n')


def _lab(**fields):
    fields = {key: dedent(value).strip() for key, value in fields.items()}
    fields["check"] = _check(fields.pop("checks"))
    fields["fix_check"] = _check(fields.pop("fix_checks"), True)
    return fields


EXERCISES = {
"75-tools": _lab(
    title="درخواست معتبر، اجرای محدود",
    goal="قرارداد درخواست را خودتان بررسی کنید و خطای ابزار را با پاسخ صفر اشتباه نگیرید.",
    prerequisite="دیکشنری، JSON و استثنای `ValueError`؛ مرز Context و اجرای Python را مرور کنید.",
    predict="نام تابع و دو Argument درست به نظر می‌رسند. آیا `b=True`، `b='48'` و `b=48` باید یکسان پذیرفته شوند؟",
    setup='''
    from mini_gpt.tools import execute_tool
    valid = {'name':'multiply','arguments':{'a':25,'b':48}}
    print('executed by Python, not by text generation:',execute_tool(valid))
    print('Python bool is an int subclass:',isinstance(True,int))
    ''',
    task="تابع `inspect_request(payload)` یک `tuple` سه‌عضوی `(name,a,b)` برگرداند. فقط کلیدهای `name` و `arguments`، نام‌های `add/subtract/multiply`، دو Argument دقیق `a/b` از نوع `int` و قدرمطلق حداکثر یک میلیون مجازند؛ `bool` را رد کنید. ورودی نامعتبر باید `ValueError` بدهد. خودتان قرارداد را بررسی کنید و `validate_call` را صدا نزنید.",
    starter='''
    def inspect_request(payload):
        # TODO
        return None
    ''',
    checks='''
    result = inspect_request(valid)
    if result is None: return False
    assert result == ('multiply',25,48)
    assert inspect_request({'name':'subtract','arguments':{'a':0,'b':-7}}) == ('subtract',0,-7)
    assert inspect_request({'name':'add','arguments':{'a':1000000,'b':1}}) == ('add',1000000,1)
    invalid = [None,{}, {'name':'shell','arguments':{'a':1,'b':2}},
               {'name':'add','arguments':{'a':1}},
               {'name':'add','arguments':{'a':1,'b':2,'c':3}},
               {'name':'add','arguments':{'a':1,'b':2},'extra':True}]
    invalid += [{'name':'add','arguments':{'a':1,'b':x}} for x in (True,'2',2.,1000001)]
    for payload in invalid:
        try: inspect_request(payload)
        except ValueError: pass
        else: raise AssertionError('invalid request was accepted')
    return True
    ''',
    solution='''
    def inspect_request(payload):
        if type(payload) is not dict or set(payload) != {'name','arguments'}:
            raise ValueError('invalid fields')
        name, args = payload['name'], payload['arguments']
        if type(name) is not str or name not in ('add','subtract','multiply'):
            raise ValueError('unknown tool')
        if type(args) is not dict or set(args) != {'a','b'}:
            raise ValueError('invalid arguments')
        if any(type(value) is not int or abs(value)>1000000 for value in args.values()):
            raise ValueError('invalid integer')
        return name,args['a'],args['b']
    ''',
    vary="فقط نام ابزار را برای عددهای ۷ و ۳ تغییر دهید. نتیجه‌های واقعی را مقایسه کنید؛ معتبر بودن Arguments تعیین نمی‌کند کدام عملیات برای سؤال مناسب است.",
    vary_code='''
    for name in ('add','subtract','multiply'):
        print(name,execute_tool({'name':name,'arguments':{'a':7,'b':3}}))
    ''',
    debug="کد خراب، خطای ابزار را به صفر تبدیل می‌کند. تابع `require_result(result)` مقدار یک `ToolResult` موفق را برگرداند و برای `ok=False`، `ValueError` بدهد؛ پاسخ موفقِ صفر باید حفظ شود.",
    bug_code='''
    failed = execute_tool({'name':'unknown','arguments':{'a':1,'b':2}})
    zero = execute_tool({'name':'subtract','arguments':{'a':2,'b':2}})
    print('wrong conversion hides a failure:',failed.value or 0, zero.value or 0)
    ''',
    fix='''
    def require_result(result):
        # TODO
        return None
    ''',
    fix_checks='''
    result = require_result(execute_tool(valid))
    if result is None: return False
    assert result == 1200
    assert require_result(zero) == 0
    assert require_result(execute_tool({'name':'subtract','arguments':{'a':1,'b':4}})) == -3
    try: require_result(failed)
    except ValueError: pass
    else: raise AssertionError('failure must not become a numeric answer')
    return True
    ''',
    fix_solution='''
    def require_result(result):
        if not result.ok:
            raise ValueError(result.error)
        return result.value
    ''',
    connection="`mini_gpt/tools.py` همین دروازه را برای Controller می‌سازد. هیچ `eval`، شبکه یا دسترسی فایل در ابزارهای این دفتر وجود ندارد؛ درخواست‌ها دستی‌اند، نه شاهد انتخاب ابزار توسط MiniGPT.",
    takeaway="اگر درخواست از متن یک مدل یا یک سند بازیابی‌شده آمده باشد، کدام بررسی‌ها باید همچنان در Python انجام شوند؟",
),
"76-reasoning": _lab(
    title="گام‌های درست، جواب درست؟",
    goal="یک برنامهٔ وابسته بسازید و جواب آن را از خود صورت مسئله وارسی کنید.",
    prerequisite="ابزارهای `add/multiply`، دیکشنری و تفاوت شناسهٔ گام با مقدار آن را از درس بخوانید.",
    predict="هر روز ۲۵ دقیقه مطالعه و ۲۰ دقیقه آزمایش داریم. چرا `25+20*3` با سه روز اجرای هر دو فعالیت برابر نیست؟",
    setup='''
    from mini_gpt.reasoning import execute_plan, verify_study_answer
    from mini_gpt.tools import ToolError
    print('independent task check:',verify_study_answer([25,20],3,135))
    print('a locally correct but wrong task answer:',verify_study_answer([25,20],3,85))
    ''',
    task="تابع `make_study_plan(reading, lab, days)` فهرست دو گام بدهد: گام `daily` جمع دو زمان و گام `total` ضرب نتیجهٔ `daily` در تعداد روزها. هر گام کلیدهای `id/name/arguments` دارد؛ Argument اولِ گام دوم رشتهٔ `'daily'` باشد، نه جواب محاسبه‌شدهٔ دستی. عددها نامنفی و در محدودهٔ مثال‌اند.",
    starter='''
    def make_study_plan(reading, lab, days):
        # TODO
        return None
    ''',
    checks='''
    result = make_study_plan(25,20,3)
    if result is None: return False
    for reading,lab,days in ((25,20,3),(10,5,4),(0,8,0),(1,0,7)):
        plan = make_study_plan(reading,lab,days)
        assert len(plan) == 2
        assert plan[0]['id'] == 'daily' and plan[1]['id'] == 'total'
        assert plan[1]['arguments']['a'] == 'daily'
        trace = execute_plan(plan)
        assert trace[0].value == reading+lab
        assert verify_study_answer([reading,lab],days,trace[-1].value)
    return True
    ''',
    solution='''
    def make_study_plan(reading, lab, days):
        return [{'id':'daily','name':'add','arguments':{'a':reading,'b':lab}},
                {'id':'total','name':'multiply','arguments':{'a':'daily','b':days}}]
    ''',
    vary="فقط سقف گام‌های یک برنامهٔ معلوم را از دو به یک کاهش دهید. برنامهٔ زیر یک محاسبهٔ جداست: زمان سه جلسهٔ ۱۰دقیقه‌ای را می‌گیرد و پنج دقیقه از کل کم می‌کند. چرا برنامهٔ معتبر هم ممکن است از بودجه عبور کند؟",
    vary_code='''
    separate_plan = [
        {'id':'sessions','name':'multiply','arguments':{'a':10,'b':3}},
        {'id':'remaining','name':'subtract','arguments':{'a':'sessions','b':5}},
    ]
    for limit in (2,1):
        try: print(limit,execute_plan(separate_plan,max_steps=limit)[-1].value)
        except ToolError as error: print('expected budget stop:',error)
    ''',
    debug="برنامهٔ خراب تمام ابزارها را درست اجرا کرده، ولی مطالعه را فقط یک روز شمرده است. تابع `check_answer(reading,lab,days,answer)` یک `bool` بدهد و جواب را مستقل از این برنامه بررسی کند. `bool` را به‌عنوان جواب عددی نپذیرید.",
    bug_code='''
    wrong_plan = [
        {'id':'labs','name':'multiply','arguments':{'a':20,'b':3}},
        {'id':'total','name':'add','arguments':{'a':25,'b':'labs'}},
    ]
    wrong_trace = execute_plan(wrong_plan)
    print('all tools executed, but answer is:',wrong_trace[-1].value)
    ''',
    fix='''
    def check_answer(reading, lab, days, answer):
        # TODO
        return None
    ''',
    fix_checks='''
    result = check_answer(25,20,3,135)
    if result is None: return False
    assert result is True
    assert check_answer(25,20,3,85) is False
    assert check_answer(10,5,4,60) is True
    assert check_answer(10,5,4,59) is False
    assert check_answer(0,0,3,0) is True
    assert check_answer(1,0,1,True) is False
    return True
    ''',
    fix_solution='''
    def check_answer(reading, lab, days, answer):
        return type(answer) is int and answer == (reading+lab)*days
    ''',
    connection="`execute_plan` گزارش ابزار واقعاً اجراشده را می‌دهد. `verify_study_answer` مسئله را جدا بررسی می‌کند. هیچ‌کدام متنِ توضیح مدل یا Activationهای داخل `MiniGPT` نیست.",
    takeaway="کدام شاهد در این دفتر دربارهٔ صحت عملیات بود و کدام دربارهٔ پاسخ‌دادن به سؤال درست؟",
),
"77-candidates": _lab(
    title="رأی پرتکرار را با شاهد مستقل بسنجید",
    goal="تساوی، توافقِ غلط و ضعف Verifier را در انتخاب میان Candidateها آشکار کنید.",
    prerequisite="شمارش با دیکشنری و تابعی که پاسخ را True یا False ارزیابی می‌کند؛ Candidateهای این دفتر دستی‌اند.",
    predict="در `[85,85,135]` چه عددی پرتکرار است؟ اگر مسئلهٔ اصلی جواب ۱۳۵ داشته باشد، آیا توافق بیشتر شاهد کافی است؟",
    setup='''
    from mini_gpt.reasoning import majority_answer, verify_study_answer
    candidates = [85,85,135]
    print('manual candidates; no model was sampled:',candidates)
    print('task-specific accepted answer:',[a for a in candidates if verify_study_answer([25,20],3,a)])
    ''',
    task="تابع `vote(answers)` دیکشنری با `answer/count/total/tied` برگرداند. پرتکرارترین عدد فقط وقتی یکتا باشد انتخاب شود؛ تساوی یعنی `answer=None` و `tied=True`. برای فهرست خالی، `count=total=0` و `tied=False` باشد. ورودی‌های این تمرین عدد صحیح‌اند؛ تابع آمادهٔ رأی را صدا نزنید.",
    starter='''
    def vote(answers):
        # TODO
        return None
    ''',
    checks='''
    result = vote(candidates)
    if result is None: return False
    assert result == {'answer':85,'count':2,'total':3,'tied':False}
    assert vote([135,85,135]) == {'answer':135,'count':2,'total':3,'tied':False}
    assert vote([2,1]) == {'answer':None,'count':1,'total':2,'tied':True}
    assert vote([]) == {'answer':None,'count':0,'total':0,'tied':False}
    assert vote([0,0,-1,2])['answer'] == 0
    assert vote([1,1,2,2])['tied'] is True
    return True
    ''',
    solution='''
    def vote(answers):
        counts = {}
        for answer in answers:
            counts[answer] = counts.get(answer,0)+1
        if not counts:
            return {'answer':None,'count':0,'total':0,'tied':False}
        maximum = max(counts.values())
        winners = [answer for answer,count in counts.items() if count == maximum]
        tied = len(winners) != 1
        return {'answer':None if tied else winners[0],'count':maximum,'total':len(answers),'tied':tied}
    ''',
    vary="فقط تعداد Candidateهای دیده‌شده از یک فهرست ثابت را زیاد کنید. پاسخ پرتکرار را با انتخابِ مبتنی بر Verifier مقایسه کنید و شمار فراخوانی‌های واقعی آن را هم گزارش کنید. این پیاده‌سازی حتی پاسخ‌های تکراری را دوباره بررسی می‌کند. بهترشدن یا بدترشدن این مثال دستی، ادعای عمومی دربارهٔ Sampling مدل نیست.",
    vary_code='''
    from mini_gpt.reasoning import select_verified
    ordered_candidates = [85,135,135,85,85]
    candidate_reports = []
    for budget in (1,3,5):
        visible_candidates = ordered_candidates[:budget]
        result = majority_answer(visible_candidates)
        verifier_inputs = []
        def counted_verifier(answer):
            verifier_inputs.append(answer)
            return verify_study_answer([25,20],3,answer)
        verified_answer = select_verified(visible_candidates,counted_verifier)
        report = {'candidate_budget':budget,'voted_answer':result.answer,
                  'votes':result.count,'verified_answer':verified_answer,
                  'verifier_calls':len(verifier_inputs)}
        candidate_reports.append(report)
        print(report)
    ''',
    debug="کد خراب رأی را با Verification یکی گرفته است. تابع `choose_checked(answers,verifier)` فقط یک پاسخ متمایزِ پذیرفته‌شده را برگرداند؛ در نبود پاسخ یا پذیرش چند عدد متفاوت، `None` بدهد. تکرارِ یک پاسخ پذیرفته‌شده ابهام تازه‌ای نیست.",
    bug_code='''
    wrong_selection = majority_answer(candidates).answer
    print('popular:',wrong_selection,'correct:',verify_study_answer([25,20],3,wrong_selection))
    print('a weak verifier can accept conflicting answers:',[a for a in (85,135) if a > 0])
    ''',
    fix='''
    def choose_checked(answers, verifier):
        # TODO
        return None
    ''',
    fix_checks='''
    result = choose_checked(candidates,lambda answer: answer == 135)
    if result is None: return False
    assert result == 135
    assert choose_checked([7,7],lambda answer: answer == 7) == 7
    assert choose_checked([1,2],lambda answer: answer > 0) is None
    assert choose_checked([1,2],lambda answer: answer == 3) is None
    assert choose_checked([],lambda answer: True) is None
    assert choose_checked([0,-1],lambda answer: answer == 0) == 0
    return True
    ''',
    fix_solution='''
    def choose_checked(answers, verifier):
        accepted = {answer for answer in answers if verifier(answer)}
        return next(iter(accepted)) if len(accepted) == 1 else None
    ''',
    connection="`select_verified` در `reasoning.py` انتخاب را از تولید جدا می‌کند. می‌توان خروجی چند اجرای Backend واقعی را به این مرحله داد؛ فهرست دستی نه SFT است، نه Self-consistency کامل و نه شاهد کیفیت MiniGPT.",
    takeaway="در یک مقایسهٔ منصفانه، علاوه بر درصد پاسخ درست، کدام هزینه و کدام محدودیت Verifier را باید گزارش کنید؟",
),
"80-controller": _lab(
    title="پیشنهاد، اجرا و پایان را جدا بشمارید",
    goal="Trace یک Controller واقعی را بخوانید و ثابت کنید تعداد پیشنهادها با ابزارهای اجراشده یکسان نیست.",
    prerequisite="Context، شاهد، Memory، Schema ابزار و Verifier؛ Fixture ازپیش‌نوشته‌شده را با مدل آموزش‌دیده یکی نگیرید.",
    predict="اگر سقف ابزار صفر باشد، یک پیشنهاد معتبر باید چند اجرای واقعی ایجاد کند؟ اگر پاسخ نهایی بدون Verifier برسد، verified باید چه باشد؟",
    setup='''
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(41)
    from mini_gpt.assistant import ScriptedFixture, MiniGPTBackend, run_assistant, PROTOCOL
    from mini_gpt.context import ContextItem,build_context
    from mini_gpt.config import ModelConfig
    from mini_gpt.model import MiniGPT
    from mini_gpt.tokenizer import CharacterTokenizer
    from mini_gpt.retrieval import COURSE_DOCUMENTS,chunk_document,keyword_search
    from mini_gpt.memory import MemoryStore,MemoryRecord
    calls = [{'action':'tool','name':'multiply','arguments':{'a':25,'b':48}},
             {'action':'finish','answer':'1200','citations':[]}]
    good = run_assistant('25 * 48?',ScriptedFixture(calls),
                         verify_answer=lambda answer,citations,tools: answer == '1200')
    blocked = run_assistant('25 * 48?',ScriptedFixture(calls),max_tool_calls=0)
    print(good.backend,good.status,blocked.status)
    # Full controller prompt, random weights, real generation, NO fallback.
    question = '25*48?'
    pack = build_context(question,[ContextItem('controller:protocol',PROTOCOL,'instruction')],
                         max_tokens=4096,reserve_tokens=4,count_tokens=len)
    tokenizer = CharacterTokenizer.from_text(pack.prompt)
    model = MiniGPT(ModelConfig(tokenizer.vocab_size,len(pack.prompt)+8,8,2,1,0.))
    real = MiniGPTBackend(model,tokenizer,max_new_tokens=4)
    real_run = run_assistant(question,real)
    raw = next(event['text'] for event in real_run.events if event['state']=='propose')
    print(real.label,'UNTRAINED raw output:',repr(raw),'status:',real_run.status)
    print('real model counts:',real.last_counts)
    ''',
    task="تابع `audit_run(result)` دیکشنری `proposals/executions/verified/safe_to_report` بدهد. دو شمار اول از رویدادهای `propose/execute` در `result.events` بیایند. `verified` را از خود نتیجه بخوانید و `safe_to_report` فقط وقتی True باشد که هم وضعیت `finished` و هم `verified=True` است. این معیار آموزشیِ گزارش پاسخِ بررسی‌شده است، نه تضمین جامع ایمنی.",
    starter='''
    def audit_run(result):
        # TODO
        return None
    ''',
    checks='''
    result = audit_run(good)
    if result is None: return False
    assert result == {'proposals':2,'executions':1,'verified':True,'safe_to_report':True}
    assert audit_run(blocked) == {'proposals':1,'executions':0,'verified':False,'safe_to_report':False}
    unchecked = run_assistant('answer?',ScriptedFixture([{'action':'finish','answer':'7','citations':[]}]))
    assert unchecked.status == 'finished'
    assert audit_run(unchecked) == {'proposals':1,'executions':0,'verified':False,'safe_to_report':False}
    rejected = run_assistant('answer?',ScriptedFixture([{'action':'finish','answer':'7','citations':[]}]),
                             verify_answer=lambda a,c,t: False)
    assert audit_run(rejected)['safe_to_report'] is False
    assert real.calls == 1 and real.last_counts['generated_tokens'] == 4
    assert real_run.status == 'invalid_action' and not real_run.tool_results
    return True
    ''',
    solution='''
    def audit_run(result):
        return {'proposals':sum(event['state']=='propose' for event in result.events),
                'executions':sum(event['state']=='execute' for event in result.events),
                'verified':result.verified,
                'safe_to_report':result.status=='finished' and result.verified}
    ''',
    vary="فقط انتخاب صریح یک کلید حافظه را در `memory_keys` تغییر دهید؛ خود Store، سندها، سؤال و پاسخ Fixture ثابت بمانند. شناسه‌های واردشده به Context را مقایسه کنید. بدون انتخاب کلید، هیچ رکوردی وارد نمی‌شود. پاسخ Fixture عمدی ثابت است: این آزمایش رسیدن دادهٔ مجاز به سامانه را نشان می‌دهد، نه استفادهٔ هوشمندانهٔ مدل از حافظه.",
    vary_code='''
    chunks = [c for document in COURSE_DOCUMENTS for c in chunk_document(document)]
    hit = keyword_search('Checkpoint',chunks,k=1)[0]
    memory = MemoryStore([MemoryRecord('study-unit','minutes','user explicitly stated the unit')])
    response = {'action':'finish','answer':hit.chunk.text,'citations':[hit.chunk.id]}
    for keys in ((),('study-unit',)):
        result = run_assistant('Checkpoint',ScriptedFixture([response]),chunks=chunks,memory=memory,memory_keys=keys,
                               verify_answer=lambda a,c,t: a == hit.chunk.text and c == (hit.chunk.id,))
        print('selected memory keys, status, included:',keys,result.status,result.context_ids)
    ''',
    debug="شرط خراب با OR اجازهٔ ابزار می‌دهد، حتی وقتی یکی از دو بودجه تمام شده است. تابع `can_execute(step_index,call_count,max_steps,max_calls)` برای چهار عدد صحیح نامنفی بنویسید: اجرای ابزار فقط پیش از رسیدن هر دو شمارنده به سقف مجاز است. این شرط مربوط به ابزار است؛ پاسخ نهاییِ بدون ابزار را ممنوع نمی‌کند.",
    bug_code='''
    step_index,call_count,max_steps,max_calls = 0,2,4,2
    print('wrong OR permission:',step_index < max_steps or call_count < max_calls)
    repeated = run_assistant('25 * 48?',ScriptedFixture([calls[0],calls[0]]))
    print('same tool call repeated:',repeated.status,'executed:',len(repeated.tool_results))
    ''',
    fix='''
    def can_execute(step_index, call_count, max_steps, max_calls):
        # TODO
        return None
    ''',
    fix_checks='''
    result = can_execute(0,0,4,2)
    if result is None: return False
    assert result is True
    for state in ((0,2,4,2),(4,0,4,2),(4,2,4,2),(0,0,1,0),(0,0,0,1)):
        assert can_execute(*state) is False
    assert can_execute(3,1,4,2) is True
    assert repeated.status == 'repeated_action' and len(repeated.tool_results) == 1
    return True
    ''',
    fix_solution='''
    def can_execute(step_index, call_count, max_steps, max_calls):
        return step_index < max_steps and call_count < max_calls
    ''',
    connection="`assistant.py` واقعاً Context، Retrieval، Memory و ابزارها را کنار هم می‌گذارد. در این دفتر هم مسیر Fixture و هم تولید Token از MiniGPT اجرا شد، اما فقط اولی سناریوی دست‌نویس ابزار را دنبال کرد. گزارش این دو را ادغام نکنید.",
    takeaway="کدام نتیجه دربارهٔ درستی Controller بود، کدام دربارهٔ اتصال واقعی مدل و کدام قابلیت هنوز با وزن‌های تصادفی ثابت نشده است؟",
),
}
