"""Offline integrity checks for every generated page of the current edition."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import posixpath
from urllib.parse import unquote, urlsplit

from tools.build_book import CHAPTERS, LESSONS, PATHS, ROOT, UNITS


class BookParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.links, self.assets = [], [], []
        self.html = {}
        self.kind = ""
        self.pager_count = 0
        self.complete_ids = []
        self.breadcrumb = False
        self.main = False
        self.h1_count = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html": self.html = a
        if tag == "body": self.kind = a.get("data-page-kind")
        if a.get("id"): self.ids.append(a["id"])
        if tag == "a" and a.get("href"): self.links.append(a["href"])
        if tag in ("script", "img", "link"):
            url = a.get("src") or a.get("href")
            if url: self.assets.append(url)
        if tag == "main": self.main = True
        if tag == "h1": self.h1_count += 1
        if "pager" in a.get("class", "").split(): self.pager_count += 1
        if "breadcrumbs" in a.get("class", "").split(): self.breadcrumb = True
        if "data-complete" in a: self.complete_ids.append(a["data-complete"])


def validate(root=None):
    if not __debug__:
        raise RuntimeError('Validation requires assertions; do not use -O or PYTHONOPTIMIZE.')
    root = Path(root).resolve() if root is not None else ROOT / 'dist'
    pages = {}
    exact_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    link_count = 0
    for path in root.rglob("*.html"):
        source = path.read_text(encoding="utf-8")
        parser = BookParser()
        parser.feed(source)
        assert source.lower().startswith("<!doctype html>"), path
        assert parser.html.get("lang") == "fa" and parser.html.get("dir") == "rtl", path
        assert len(parser.ids) == len(set(parser.ids)), f"duplicate ID in {path}"
        assert parser.main and parser.h1_count == 1 and parser.breadcrumb, path
        assert not re.search(r'\[\[(?:[a-z][a-z0-9/_.-]+|[0-9]{2}-[a-z-]+)(?:\|[^\]]+)?\]\]', source), f"unresolved placeholder: {path}"
        for asset in parser.assets:
            assert not urlsplit(asset).scheme and not asset.startswith("//"), f"remote asset: {path} {asset}"
        if parser.kind in ("lesson", "checkpoint"):
            assert parser.pager_count == 1 and len(parser.complete_ids) == 1, path
        pages[path] = parser
    for path, parser in pages.items():
        for href in parser.links + parser.assets:
            link_count += 1
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            assert '\\' not in url.path, f"URL uses a Windows separator: {path} → {href}"
            relative = path.relative_to(root).as_posix()
            lexical = posixpath.normpath(posixpath.join(posixpath.dirname(relative), unquote(url.path))) if url.path else relative
            assert lexical in exact_files, f"missing or wrong-case URL: {path} → {href}"
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            assert target.is_relative_to(root), f"link escaped dist: {path} → {href}"
            assert target.is_file(), f"missing target: {path} → {href}"
            if url.fragment:
                assert target in pages and unquote(url.fragment) in pages[target].ids, f"missing anchor: {href}"
    for css in root.rglob('*.css'):
        for match in re.finditer(r'url\([\s\"\']*([^\s)\"\']+)', css.read_text(encoding='utf-8')):
            value = match[1]
            assert not urlsplit(value).scheme and not value.startswith('//'), f"remote CSS asset: {css} {value}"
            lexical = posixpath.normpath(posixpath.join(css.relative_to(root).parent.as_posix(), unquote(value)))
            assert lexical in exact_files, f"missing or wrong-case CSS asset: {css} {value}"
    for lesson in LESSONS:
        assert (root / PATHS[lesson.id]) in pages
        assert (root / "answers" / (lesson.id + ".html")) in pages
    progress_ids = [identifier for page in pages.values() for identifier in page.complete_ids]
    assert set(progress_ids) == {u[0] for u in UNITS}
    assert len(progress_ids) == len(UNITS)
    python_files = list(ROOT.glob('*.py'))
    for directory in ("mini_gpt", "book_src", "tests", "tools"):
        python_files.extend((ROOT / directory).rglob("*.py"))
    for path in python_files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"PASS: {len(pages)} HTML pages, {len(LESSONS)} lessons, {len(CHAPTERS)} chapters, "
          f"{len(UNITS)} progress units, {link_count} links/assets, {len(python_files)} Python files compile.")
    print("No broken local targets, duplicate IDs, missing navigation, or remote display dependencies.")
    return len(pages), link_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Generated site directory (default: dist).')
    validate(parser.parse_args().root)
