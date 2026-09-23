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

# These are established Persian mathematical/educational names, not awkward
# localizations. Keep them when an author deliberately uses them. English names
# remain the glossary identities and are retained when written in English.
CONVENTIONAL_PERSIAN = {
    'neural-network', 'vector', 'matrix', 'dot-product', 'matrix-multiplication',
    'transpose', 'derivative', 'partial-derivative', 'chain-rule',
    'computational-graph', 'corpus', 'subword', 'data-leakage', 'overfitting',
    'underfitting', 'distributed-training', 'character', 'language-model',
    'causal-language-model', 'generation',
}


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
    'توجه علّی چندسر': 'Causal Multi-Head Attention',
    'توجه چندسر': 'Multi-Head Attention',
    'Attention چندسر': 'Multi-Head Attention',
    'توجه تک‌سر': 'Single-Head Attention',
    'تک‌سر': 'Single-Head',
    'چندسر': 'Multi-Head',
    'همان سر': 'همان Head',
    'برای سرها': 'برای Headها',
    'سر خروجی': 'Language-model head',
    # Mask senses only: ordinary پوشش سؤال/متن/موضوع stays Persian.
    'این مثال هنوز پوشش ندارد': 'این مثال هنوز Mask ندارد',
    'وجود پوشش': 'وجود Mask',
    'خرابی پوشش': 'خرابی Mask',
    'شکل پوشش': 'شکل Mask',
    'پوشش ۴×۴': 'Mask ۴×۴',
    'با پوشش اجرا': 'با Mask اجرا',
    'پوشش را روشن': 'Mask را روشن',
    'پوشش را فعال': 'Mask را فعال',
    'پوشش روی صفحه': 'Mask روی صفحه',
    'با پوشش،': 'با Mask،',
    'بدون پوشش': 'بدون Mask',
    'حذف از حافظه و پوشش': 'حذف از حافظه و Mask',
    'نویسه‌ای': 'کاراکترمحور',
    'نویسه': 'کاراکتر',
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
    if word == 'نشانه' and re.match(r'(?:ٔ|‌ای)?\s*(?:کیفیت|فهم|خرابی|سلامت|موفقیت|مشکل|خطا|بهترشدن|صریح|آن|اینکه|′)', tail):
        return True
    return False


def canonical_label(match):
    slug = ALIASES[match.group().casefold()]
    if slug in CONVENTIONAL_PERSIAN and re.search(r'[\u0600-\u06ff]', match.group()):
        return match.group()
    # Canonical abbreviations remain recognizable alongside their full names.
    if re.fullmatch(r'[A-Z][A-Z0-9]{1,7}', match.group()):
        return match.group()
    return TERMS[ALIASES[match.group().casefold()]].name


def normalize_text(text, *, definition=False):
    text = PHRASE_PATTERN.sub(lambda m: PHRASES[m.group()], text)
    text = re.sub(r'(?<!\w)توجهی(?!\w)', 'مبتنی بر Attention', text)
    text = PATTERN.sub(lambda m: m.group() if protected(m, text) or
                      (definition and re.search(r'[\u0600-\u06ff]', m.group())) else canonical_label(m), text)
    return re.sub(r'(?<=[A-Za-z])ٔ', '', text)


def clean_definition(match):
    text = normalize_text(match[1], definition=True)
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
        definition = any(item[0] == 'dfn' for item in self.stack)
        text = normalize_text(text, definition=definition)
        if any(item[2] for item in self.stack):
            if not any(item[0] in {'title','bdi','select','option'} for item in self.stack):
                text = PATTERN.sub(lambda m: m.group() if protected(m,text) else
                                   '<bdi dir="'+('rtl' if re.search(r'[\u0600-\u06ff]',m.group()) else 'ltr')+'">'+html.escape(m.group())+'</bdi>', text)
            self.parts.append(text)
            return

        def annotate(match):
            slug = ALIASES[match.group().casefold()]
            if protected(match, text) or slug == self.own_term:
                return match.group()
            self.counts[slug] = self.counts.get(slug, 0) + 1
            anchor = f'term-ref-{slug}-{self.counts[slug]}'
            shown = match.group() if definition else canonical_label(match)
            direction = 'rtl' if re.search(r'[\u0600-\u06ff]',shown) else 'ltr'
            # One useful entry point per concept, plus its explicit definition.
            # Keep the old occurrence anchors so saved glossary returns survive.
            if self.counts[slug] > 1 and not any(item[0] == 'dfn' for item in self.stack):
                return f'<bdi dir="{direction}" id="{anchor}">{html.escape(shown)}</bdi>'
            target = f'glossary/{slug}.html'
            href = posixpath.relpath(target, posixpath.dirname(self.current) or '.')
            origin = posixpath.relpath(self.current, 'glossary') + '#' + anchor
            if self.current.startswith('glossary/'):
                query = ''  # Related concepts inherit the original reading context in JS.
            else:
                query = '?return=' + quote(origin, safe='')
            label = html.escape(shown)
            return (f'<a class="term-link" id="{anchor}" data-term="{slug}" '
                    f'href="{href}{query}" title="توضیح {label}"><bdi dir="{direction}">{label}</bdi></a>')

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


def normalize_html(source):
    """Use the same prose policy in notebook HTML, without site-relative links."""
    class NotebookProse(ProseTerms):
        def handle_data(self, text):
            if not any(item[1] for item in self.stack):
                text = normalize_text(text, definition=any(item[0] == 'dfn' for item in self.stack))
            self.parts.append(text)
    parser = NotebookProse('notebooks.html')
    parser.feed(source)
    parser.close()
    return ''.join(parser.parts)


def inline_code_html(source):
    """Protect inline code before HTML can interpret a comparison as a tag.

    Only prose backticks are markup: attributes, comments and protected subtrees
    remain byte-for-byte unchanged. A lone backtick is ordinary text.
    """
    tag_pattern = re.compile(r'''<!--.*?-->|<![^>]*>|</?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>])(?:[^<>"']|"[^"]*"|'[^']*')*>''', re.S)
    code_pattern = re.compile(r'`([^`\r\n]+)`')
    regions = ProseTerms('')
    parts, cursor = [], 0
    while cursor < len(source):
        if source[cursor] == '<':
            tag = tag_pattern.match(source, cursor)
            if tag:
                raw = tag[0]
                regions.feed(raw)
                parts.append(raw)
                cursor = tag.end()
                continue
        if source[cursor] == '`' and not any(item[1] for item in regions.stack):
            code = code_pattern.match(source, cursor)
            if code:
                parts.append('<code dir="ltr">'+html.escape(html.unescape(code[1]))+'</code>')
                cursor = code.end()
                continue
        parts.append(source[cursor])
        cursor += 1
    return ''.join(parts)


class ProseTypography(ProseTerms):
    """One protected-aware formatting policy for HTML and notebook prose.

    Notebook-only inline styles survive Jupyter's Markdown sanitizer and override
    its paragraph alignment. Never inject a global stylesheet or touch code cells.
    """
    BLOCKS = {'p','h1','h2','h3','h4','h5','h6','ul','ol','li','blockquote'}
    ARRAY_ATOM = r'(?:\[[A-Za-z0-9_.,\[\] +*/=-]+\]|\([A-Za-z0-9_=,/*× .-]+\))'
    ARRAYS = re.compile(r'(?<![A-Za-z0-9_.)\]])(?:\b[A-Za-z][A-Za-z0-9_]*=)?'+ARRAY_ATOM
                        +r'(?:\s*[·×+*/=−-]\s*(?:'+ARRAY_ATOM+r'|[−-]?\d+(?:\.\d+)?))*')
    LATIN = re.compile(r'(?<![\w-])[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*(?:[ \t]+[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*)*')
    LEAD = re.compile(r'^(\s*)((?:نکته مهم|نکته|پیش‌نیاز|تحویل|مفهومی|محاسباتی|پیاده‌سازی|پژوهشی):)\s+([\u0600-\u06ff][^\s<]{0,11})(?=\s)')

    def __init__(self, notebook=False):
        super().__init__('')
        self.notebook = notebook
        self.frames = []

    @staticmethod
    def styled(attrs, additions):
        attrs = dict(attrs)
        styles = dict(part.strip().split(':',1) for part in attrs.get('style','').split(';') if ':' in part)
        styles = {key.strip():value.strip() for key,value in styles.items()}
        styles.update(additions)
        attrs['style'] = ';'.join(f'{key}:{value}' for key,value in styles.items())
        return attrs

    @staticmethod
    def start_tag(tag, attrs):
        return '<'+tag+''.join(' '+key+(('="'+html.escape(value,quote=True)+'"') if value is not None else '')
                              for key,value in attrs.items())+'>'

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        classes = data.get('class','').split()
        special = tag in self.SKIP or any(c in classes for c in ('math','figure','no-terms'))
        opaque = any(frame['skip'] for frame in self.frames)
        inherited = self.frames[-1]['direction'] if self.frames else None
        direction = data.get('dir',inherited)
        styled = data
        if self.notebook and not opaque:
            if special and tag not in {'svg','script','style','textarea'}:
                styled = self.styled(styled, {'direction':'ltr','text-align':'left','unicode-bidi':'isolate'})
                if tag == 'pre' or 'math' in classes or 'figure' in classes:
                    styled = self.styled(styled, {'overflow-x':'auto','max-width':'100%'})
            elif not special and direction in {'rtl','ltr'} and (tag in self.BLOCKS or (tag == 'div' and 'dir' in data)):
                styled = self.styled(styled, {'text-align':'right' if direction == 'rtl' else 'left'})
                if direction == 'rtl' and tag in {'ul','ol'}:
                    # Jupyter strips logical padding/border properties. These
                    # physical equivalents are scoped to this known direction.
                    styled = self.styled(styled, {'padding-right':'1.5em','padding-left':'0'})
                if tag == 'blockquote':
                    start, end = ('right','left') if direction == 'rtl' else ('left','right')
                    styled = self.styled(styled, {'border-'+start:'3px solid #a3acbc','border-'+end:'0',
                                                 'padding-'+start:'1em','padding-'+end:'0'})
        index = len(self.parts)
        self.parts.append(self.get_starttag_text() if styled == data else self.start_tag(tag,styled))
        if tag not in self.VOID:
            self.frames.append(dict(tag=tag,skip=special,direction=direction,index=index,attrs=styled,
                                    text='', classify=self.notebook and not opaque and not special
                                    and tag in {'p','h1','h2','h3','h4','h5','h6','td','th'}))

    def handle_endtag(self, tag):
        for index in range(len(self.frames)-1,-1,-1):
            frame = self.frames[index]
            if frame['tag'] != tag:
                continue
            if frame['classify']:
                attrs, text = frame['attrs'], html.unescape(frame['text'])
                direction = attrs.get('dir')
                if direction is None and tag in {'td','th'}:
                    direction = 'rtl' if re.search(r'[\u0600-\u06ff]',text) else 'ltr'
                elif direction is None and re.search('[A-Za-z]',text) and not re.search(r'[\u0600-\u06ff]',text):
                    direction = 'ltr'
                if direction in {'rtl','ltr'}:
                    attrs = {**attrs, 'dir':direction}
                    attrs = self.styled(attrs, {'text-align':'right' if direction == 'rtl' else 'left'})
                    self.parts[frame['index']] = self.start_tag(tag,attrs)
            del self.frames[index:]
            break
        self.parts.append('</'+tag+'>')

    def handle_data(self, text):
        for frame in self.frames:
            frame['text'] += text
        if any(frame['skip'] or frame['tag'] in {'bdi','title','option','select'} for frame in self.frames):
            self.parts.append(text)
            return
        def latin(chunk):
            if not self.notebook:
                return chunk
            return self.LATIN.sub(lambda m:'<bdi dir="ltr">'+m[0]+'</bdi>',chunk)
        lead = self.LEAD.match(text) if self.frames and self.frames[-1]['tag'] in self.BLOCKS else None
        if lead:
            self.parts.append(lead[1]+'<span class="phrase-lead" style="white-space:nowrap">'+lead[2]+' '+lead[3]+'</span>')
            text = text[lead.end():]
        cursor = 0
        for match in self.ARRAYS.finditer(text):
            self.parts.append(latin(text[cursor:match.start()]))
            style = ' style="white-space:nowrap"' if len(match[0]) <= 28 else ''
            self.parts.append('<bdi dir="ltr" class="inline-math"'+style+'>'+match[0]+'</bdi>')
            cursor = match.end()
        self.parts.append(latin(text[cursor:]))

    def handle_entityref(self, name):
        for frame in self.frames:
            frame['text'] += html.unescape('&'+name+';')
        super().handle_entityref(name)

    def handle_charref(self, name):
        for frame in self.frames:
            frame['text'] += html.unescape('&#'+name+';')
        super().handle_charref(name)


def typography_html(source, *, notebook=False):
    """Idempotent formatting; protected subtrees and attributes stay opaque."""
    parser = ProseTypography(notebook)
    parser.feed(source)
    parser.close()
    return ''.join(parser.parts)
