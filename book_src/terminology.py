"""Prose-only canonical terminology and static, accessible glossary links.

Never rewrite code, formulas, URLs, attributes, imported records or editable text.
Long phrases win over their component terms. Ordinary Persian senses are guarded.
"""
import html
from html.parser import HTMLParser
import posixpath
import re
from urllib.parse import quote

from .glossary import TERMS


ALIASES = {}
PATTERN = None


def configure_terms():
    global PATTERN
    ALIASES.clear()
    for slug, term in TERMS.items():
        for alias in (term.name, *term.aliases.split('|')):
            if alias:
                ALIASES[alias.casefold()] = slug
    PATTERN = re.compile(
        r'(?<![\w.\-/‌])(?:' + '|'.join(re.escape(k) for k in sorted(ALIASES,key=len,reverse=True))
        + r')(?![A-Za-z0-9_\-/]|\.[A-Za-z0-9_])(?=$|[^\w‌]|‌?ها(?:یی|ی(?:ش|شان|م|مان|ت|تان)?)?(?:\W|$)|ی(?:\W|$))', re.I)


configure_terms()

# Only context-specific technical phrases; never replace the general word خطا.
PHRASES = {
    'جدول نمایش': 'جدول Embedding',
    'جدول‌های نمایش': 'جدول‌های Embedding',
    'جدول نمایش‌ها': 'جدول Embeddingها',
    'پرسش، کلید و مقدار': 'Query، Key و Value',
    'پرسش/کلید/مقدار': 'Query/Key/Value',
    'سطر پرسش': 'سطر Query',
    'ستون کلید': 'ستون Key',
    'موقعیت پرسش': 'موقعیت Query',
    'موقعیت کلید': 'موقعیت Key',
    'خطای آموزش': 'Loss آموزش',
    'خطای ارزیابی': 'Loss ارزیابی',
    'خطای دستهٔ آموزش': 'Loss مربوط به Batch آموزش',
    'نمودار خطا': 'نمودار Loss',
    'منحنی خطا': 'منحنی Loss',
    'امتیاز همهٔ واژگان': 'Logits همهٔ Tokenها',
    'امتیاز واژگان': 'Logits',
    'تک‌پارامتری': 'با یک Parameter',
    'دوپارامتری': 'با دو Parameter',
    'غیرپارامتری': 'بدون Parameter',
    'پارامترند': 'Parameter هستند',
    'هیچ پارامتری': 'هیچ Parameter',
    'جدول پارامتری': 'جدول Parameterها',
    'تبدیل‌های پارامتری': 'تبدیل‌های دارای Parameter',
    'ضرب‌های ماتریسی': 'عملیات Matrix multiplication',
    'مبتنی بر توجهی': 'مبتنی بر Attention',
    'پرسش، کلید، مقدار': 'Query، Key، Value',
    'هر سطر یک پرسش و هر ستون یک کلید است': 'هر سطر یک Query و هر ستون یک Key است',
    'Q · پرسش': 'Q · Query',
    'K · کلید': 'K · Key',
    'V · مقدار': 'V · Value',
    'خطای نمایش‌داده‌شده': 'Loss نمایش‌داده‌شده',
    'کم‌شدن خطا با انتخاب هدف': 'کم‌شدن Loss با انتخاب هدف',
}
PHRASE_PATTERN = re.compile('|'.join(re.escape(k) for k in sorted(PHRASES, key=len, reverse=True)))


def protected(match, text):
    """Retain ordinary-language uses of an otherwise technical Persian noun."""
    word = match.group().casefold()
    tail = text[match.end():]
    before = text[max(0, match.start()-12):match.start()]
    if re.match(r'[A-Za-z]',word) and tail.startswith(('(', '[')):
        return True  # Function calls in prose still use their actual API spelling.
    if word in {'بهینه‌ساز','بهینه ساز'} and tail.startswith('ی'):
        return True  # Optimization is not an Optimizer object.
    if word == 'توجه' and (re.match(r'\s*(?:ن?کن|ن?کرد|ن?می[‌ ]?کن|ن?داشت|به\s+این)', tail) or before.endswith('با ')):
        return True
    if word == 'نشانه' and re.match(r'(?:ٔ|‌ای)?\s*(?:کیفیت|فهم|خرابی|سلامت|موفقیت|مشکل|خطا|صریح|آن|اینکه|′)', tail):
        return True
    return False


def canonical_label(match):
    # Canonical abbreviations remain recognizable alongside their full names.
    if re.fullmatch(r'[A-Z][A-Z0-9]{1,7}', match.group()):
        return match.group()
    return TERMS[ALIASES[match.group().casefold()]].name


def normalize_text(text):
    text = PHRASE_PATTERN.sub(lambda m: PHRASES[m.group()], text)
    text = re.sub(r'(?<!\w)توجهی(?!\w)', 'مبتنی بر Attention', text)
    text = PATTERN.sub(lambda m: m.group() if protected(m, text) else canonical_label(m), text)
    return re.sub(r'(?<=[A-Za-z])ٔ', '', text)


def clean_definition(match):
    text = normalize_text(match[1])
    # Remove only the now-redundant translated label, not its explanation.
    for term in TERMS.values():
        text = text.replace(f'{term.name} یا {term.name}', term.name)
        text = text.replace(f'{term.name} ({term.name})', term.name)
        text = text.replace(f'{term.name} ({term.name}؛ ', f'{term.name} (')
    return '<dfn>' + text + '</dfn>'


class ProseTerms(HTMLParser):
    VOID = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
    SKIP = {'pre','code','script','style','textarea','svg','kbd','samp'}
    NO_LINK = {'a','button','summary','label','option','select','title','h1','h2','h3','h4','nav'}

    def __init__(self, current, own_term=None):
        super().__init__(convert_charrefs=False)
        self.current, self.own_term = current, own_term
        self.parts, self.stack = [], []
        self.counts = {}

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        classes = data.get('class','').split()
        skip = tag in self.SKIP or 'math' in classes or 'no-terms' in classes
        self.parts.append(self.get_starttag_text())
        if tag not in self.VOID:
            self.stack.append((tag, skip, tag in self.NO_LINK))

    def handle_startendtag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        self.parts.append(f'</{tag}>')
        for index in range(len(self.stack)-1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, text):
        if any(item[1] for item in self.stack):
            self.parts.append(text)
            return
        text = normalize_text(text)
        if any(item[2] for item in self.stack):
            if not any(item[0] in {'title','bdi','select','option'} for item in self.stack):
                text = PATTERN.sub(lambda m: m.group() if protected(m,text) else
                                   '<bdi dir="ltr">'+html.escape(m.group())+'</bdi>', text)
            self.parts.append(text)
            return

        def annotate(match):
            slug = ALIASES[match.group().casefold()]
            if protected(match, text) or slug == self.own_term:
                return match.group()
            self.counts[slug] = self.counts.get(slug, 0) + 1
            anchor = f'term-ref-{slug}-{self.counts[slug]}'
            # One useful entry point per concept, plus its explicit definition.
            # Keep the old occurrence anchors so saved glossary returns survive.
            if self.counts[slug] > 1 and not any(item[0] == 'dfn' for item in self.stack):
                return f'<bdi dir="ltr" id="{anchor}">{html.escape(canonical_label(match))}</bdi>'
            target = f'glossary/{slug}.html'
            href = posixpath.relpath(target, posixpath.dirname(self.current) or '.')
            origin = posixpath.relpath(self.current, 'glossary') + '#' + anchor
            if self.current.startswith('glossary/'):
                query = ''  # Related concepts inherit the original reading context in JS.
            else:
                query = '?return=' + quote(origin, safe='')
            label = html.escape(canonical_label(match))
            return (f'<a class="term-link" id="{anchor}" data-term="{slug}" '
                    f'href="{href}{query}" title="توضیح {label}"><bdi dir="ltr">{label}</bdi></a>')

        self.parts.append(PATTERN.sub(annotate, text))

    def handle_entityref(self, name):
        self.parts.append('&'+name+';')

    def handle_charref(self, name):
        self.parts.append('&#'+name+';')

    def handle_decl(self, decl):
        self.parts.append('<!'+decl+'>')

    def handle_comment(self, data):
        self.parts.append('<!--'+data+'-->')


def annotate_html(source, current, own_term=None):
    # A definition's original gloss remains unless it is a duplicate label.
    source = re.sub(r'<dfn>([^<]+)</dfn>', clean_definition, source)
    parser = ProseTerms(current, own_term)
    parser.feed(source)
    parser.close()
    return ''.join(parser.parts)
