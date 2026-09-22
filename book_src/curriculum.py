"""Shared curriculum identity and routes; no rendering or notebook side effects."""
import importlib
from pathlib import PurePosixPath

LESSONS = [lesson for part in range(1, 11)
           for lesson in importlib.import_module(f'book_src.part{part:02}').LESSONS]
BY_ID = {lesson.id: lesson for lesson in LESSONS}
CHAPTERS, PATHS = {}, {}
for lesson in LESSONS:
    key = (lesson.part, lesson.chapter)
    if key not in CHAPTERS:
        number = sum(p == lesson.part for p, _ in CHAPTERS) + 1
        CHAPTERS[key] = (f'part-{lesson.part:02}/chapter-{number:02}/index.html', [])
    chapter_path, chapter_lessons = CHAPTERS[key]
    chapter_lessons.append(lesson)
    PATHS[lesson.id] = str(PurePosixPath(chapter_path).with_name(lesson.id + '.html'))

