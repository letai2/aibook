"""Shared curriculum identity and routes; no rendering or notebook side effects."""
import importlib
from dataclasses import replace
from pathlib import PurePosixPath

LESSONS = [lesson for part in range(1, 11)
           for lesson in importlib.import_module(f'book_src.part{part:02}').LESSONS]
# Existing URLs and IDs are permanent, even when a lesson moves in the syllabus.
# In particular, old bookmarks for scale/cache/RAG remain valid without JS.
LEGACY_PATHS = {}
_chapters = {}
for lesson in LESSONS:
    key = (lesson.part, lesson.chapter)
    if key not in _chapters:
        _chapters[key] = sum(p == lesson.part for p, _ in _chapters) + 1
    LEGACY_PATHS[lesson.id] = f'part-{lesson.part:02}/chapter-{_chapters[key]:02}/{lesson.id}.html'

from .part11 import LESSONS as CONTEXT
from .part12 import LESSONS as MEMORY
from .part13 import LESSONS as TOOLS
from .system_lessons import LESSONS as SYSTEM
from .training_extension import LESSONS as EXTENSION

_all = {lesson.id: lesson for lesson in LESSONS + CONTEXT + MEMORY + TOOLS + SYSTEM + EXTENSION}
_tail = [
    (10, None, ['62b-lifecycle', '65-sft', '65a-sft-lab', '65b-lora', '66-preference']),
    (11, None, ['68-context', '67-rag', '69-chunks', '70-vectors', '71-grounding']),
    (12, None, ['72-history', '73-summary', '74-memory']),
    (13, None, ['75-tools', '76-reasoning', '77-candidates']),
    (14, None, ['80-controller', '81-system-eval']),
    (15, None, ['63-scale', '64-cache', '82-performance', '83-deployment', '84-capstone']),
]
LESSONS = [lesson for lesson in LESSONS if lesson.part < 10]
for part, _, identifiers in _tail:
    for identifier in identifiers:
        lesson = _all[identifier]
        chapter = ('یافتن شاهد' if identifier == '67-rag' else
                   'حافظه و محاسبه' if identifier in {'63-scale', '64-cache'} else
                   'مسیر تغییر رفتار' if identifier == '62b-lifecycle' else lesson.chapter)
        LESSONS.append(replace(lesson, part=part, chapter=chapter))
if len(LESSONS) != len(_all):
    raise ValueError('Every authored lesson must occur exactly once in the curriculum')
BY_ID = {lesson.id: lesson for lesson in LESSONS}
CHAPTERS, PATHS = {}, {}
for lesson in LESSONS:
    key = (lesson.part, lesson.chapter)
    if key not in CHAPTERS:
        number = sum(p == lesson.part for p, _ in CHAPTERS) + 1
        CHAPTERS[key] = (f'part-{lesson.part:02}/chapter-{number:02}/index.html', [])
    chapter_path, chapter_lessons = CHAPTERS[key]
    chapter_lessons.append(lesson)
    PATHS[lesson.id] = LEGACY_PATHS.get(lesson.id, str(PurePosixPath(chapter_path).with_name(lesson.id + '.html')))
