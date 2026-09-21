# Connected reading experience — 2026-09-20

This extends the visual redesign documented in `VISUAL_REVIEW.md`. The active
book is generated into `dist`; edit `book_src`, not individual output pages.
The pre-change source/output snapshot is preserved in
`archive/pre-connected-glossary-20260920.zip`.

## Navigation

- All 72 lessons and ten checkpoints follow one 82-unit ordered journey.
  Every unit has a contextual next step, a previous step, a parent chapter or
  part, and a position indicator. The first links back to the guide; the final
  checkpoint continues to a personal project rather than ending at a dead end.
- All 35 chapter indexes, ten part indexes, answers, glossary pages, code
  references, guide, project, API reference, lab and journal offer a next step.
- Completion remains opt-in. Reading or opening a glossary does not mark a
  lesson complete or change the existing progress schema.

## Terminology and connected reference

- `glossary.py` contains 49 independently authored core explanations, covering
  every term explicitly requested, with motivation, intuition, technical detail,
  examples, Mini-GPT connections and related concepts.
- Another 164 entries reuse the actual definitions, worked answers and project
  connections in the existing lessons. Their provenance is visible. Definitions
  are extracted as complete sentences without splitting inline code or tags;
  lesson material is retained, not replaced by generic dictionary filler.
- The 213 concept pages have stable English slugs. The central glossary searches
  English names and accepted Persian aliases. Additional definitions sit in an
  expandable section so the initial page stays manageable.
- `terminology.py` normalizes explanatory prose at build time across the whole
  book. Long technical phrases take precedence. Ordinary Persian uses of words
  such as attention, sign and error are distinguished from technical meanings;
  compounds and grammatical contexts have explicit regression cases.
- Code, formulas, function calls, filenames, attributes, editable text and
  imported records are not rewritten. Actual API spelling remains authoritative.
  Bidirectional isolation helps English terminology sit naturally in Persian
  headings and navigation; plain-text-only HTML elements remain plain text.
- Terms inherit the paragraph color and use a quiet dotted rule, not button
  styling or bright navigation colors. Hover, keyboard focus and the return
  destination remain clearly visible.
- A lesson link carries its specific anchor through related glossary pages.
  Returns are limited to known pages of the same book, not arbitrary redirects.
  Collapsed ancestors reopen on return. Standalone entries link to their
  introductory lesson; native browser Back also works. Without JavaScript,
  explanations and normal navigation remain readable; exact context return uses
  browser Back.
- Authored lab readouts and captions use the same names and links. These links
  return to their experiment station. Arbitrary imported notes, tokens, model
  output and learner records stay literal data, never HTML or glossary markup.
  Transient experiment input/loaded files are not newly persisted by this work.

## Font and hierarchy

Vazirmatn v33.003 is bundled as an unmodified variable WOFF2, with its complete
SIL Open Font License. Sources: [official release font](https://github.com/rastikerdar/vazirmatn/blob/v33.003/fonts/webfonts/Vazirmatn%5Bwght%5D.woff2)
and [official license](https://github.com/rastikerdar/vazirmatn/blob/v33.003/OFL.txt).
The book makes no external font request. Preloading and `font-display: swap`
avoid delaying readable content.

The actual rendered Persian was evaluated on desktop and mobile: connected
glyphs, mixed Persian/English lines, long headings, captions, exercises, glossary
definitions and continuation navigation. Vazirmatn fits the existing restrained
visual design. This is an evaluation of the chosen face, not a claimed controlled
comparison of every candidate font.

Body text is 18 px with 2.1 line height, and 17 px on narrow screens. Titles,
section headings, objectives, notes, exercises, captions and navigation have
separate sizes/weights. Code retains Consolas / Courier New / monospace and is
not forced into the prose font. Persian diagram labels use the reading font.

## Verification

- Whole-build checks cover every generated page, all local references, the
  exact ordered navigation, every glossary destination/return anchor, related
  concepts, and all 60 executable lesson examples. Code text remains identical.
- Eight editorial regression tests join the existing 15 release/design tests.
  One new JavaScript test verifies full-term matching and literal text
  preservation; the existing math, inspection and journal tests remain active.
- Browser layout checks traversed all 82 learning units at 320 px and 768 px,
  plus all 213 glossary pages at 320 px. No document-width overflow was found.
  Selected screenshots were inspected at desktop size and 320/390 px, including
  long English titles, Persian paragraphs, diagrams and the next-step section.
- Browser checks verified Query → Attention → exact Query return; static and
  dynamic lab returns; reopening a collapsed experiment; invalid external return
  fallback; English/Persian glossary search and zero results; final-checkpoint
  continuation; locally loaded font; separate code font; QKV inspection and
  training metrics. A plain-text select-label regression found during QA was
  corrected. The review tab reported no browser warnings or errors.
- This combines whole-book structural checks and selected visual/editorial
  inspections. It does not claim a fresh line-by-line human copyedit or a
  screenshot of every scroll position on every operating system.

Run `python -B prepare_release.py` for the final gate and current checksums.
`release/manifest.json` and `release/SHA256SUMS.txt` identify the exact bundle.
The Python model and lesson executable code were not changed; prior CPU training
verification is historical evidence, not a new training run in this pass.

## Deployment boundary

The existing Sites project lookup still returned `NOT_FOUND`. Its configuration
and audience were not changed, and no replacement project, upload, publication,
commit or push was performed. Local release preparation is complete separately
from remote publishing; resolve access to the configured project before using
that publication workflow. Live HTTPS and hosting-header checks remain pending
publication.
