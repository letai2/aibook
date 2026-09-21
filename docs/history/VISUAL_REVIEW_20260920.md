# Visual refactor review — 2026-09-20

The subsequent glossary/navigation/font extension is documented in
`EDITORIAL_REVIEW.md`. Counts and hashes below describe this earlier visual pass;
current release evidence is at the top of `PROGRESS.md`.

## Design and preservation

The visual layer was rebuilt around a quiet reading surface, a compact header,
and subject-specific introductions. Every one of the 72 lessons has an explicit
brief in `book_src/visuals.py`: twelve subject palettes and six composition
families (statement, question, diagram, workbench, case, editorial). Assignment
is authored, not random. Captions distinguish schematic illustrations from the
unchanged numerical labs and genuine model inspection sample.

Navigation, project context, self-checks, model map, progress controls, and lab
stations use progressive disclosure. They have not been removed. Code retains
its exact text, with restrained Python syntax highlighting and a copy control.
Exercises use a prediction / run / observe / explain sequence. On phones the
residual architecture diagram becomes a readable ordered process rather than a
scaled-down desktop SVG. Reduced-motion preferences disable animations and
transitions; there are no looping animations or hidden-content entry effects.

The pre-refactor version is preserved in
`archive/pre-visual-refactor-20260920.zip`. All ten lesson source files,
checkpoints, and reference content were compared byte-for-byte with that
snapshot and remain unchanged. No Mini-GPT model code was edited in this pass.
All 72 lessons, 35 chapters, ten checkpoints, 82 progress units, answers, code
pages, labs, and journal remain present.

## Browser review scope

- Every one of the 72 lesson openings was rendered and visually inspected on
  desktop, in curriculum order. This was not just a shared stylesheet review.
- All 72 lessons received document-width and opening-boundary checks at 320 px
  and 768 px viewports. No page-level horizontal overflow or opening/context
  collision remained. A prior 390 px pass identified the checkpoint lesson's
  long inline code; wrapping was fixed and covered by the stricter 320 px pass.
- Selected phone/tablet screenshots additionally covered attention masking,
  Transformer architecture, retrieval, exercise composition, the alternative
  residual process diagram, and journal fields.
- Lower-page desktop inspection covered highlighted Python and its copy
  feedback, exercises and expanded self-checks, attention matrices, the model
  inspector, and the notebook. Existing journal records were preserved.
- Search returned the two block lessons for `بلوک`. Header menus were verified
  to exclude each other; Escape closes the active menu and restores focus.
  The search hint now uses terms that actually appear in titles or identifiers.
- Attention mask/query controls updated the numeric result. A direct lab hash
  opened the requested station. The real bundled inspection sample loaded,
  and all seven inspection views rendered without page-level horizontal
  overflow at a 390 px viewport; wide tables scroll within their own region.
- The notebook retained all seven fields and its three existing QA records.
  This review did not clear browser storage or retest every import conflict
  scenario; the existing JavaScript regression suite covers those contracts.

These checks do not claim exhaustive screenshots of every scroll position of
all 250 generated pages or testing every browser/operating system. Static link
validation and code-preservation tests complement the visual review.

## Lesson-specific corrections found during review

- Tensor shape: replaced a decorative grid with an actual 2-by-3 example.
- Matrix multiplication: replaced misleading sequential arrows with compatible
  matrix dimensions.
- Probability/logits: bar heights now match their stated example values.
- Data contracts: removed a constrained-height editorial stack that collided
  with following content.
- Self-attention: made the shared input branch to Q, K, and V.
- Training curves: made schematic status and line meanings explicit.
- KV cache: showed concatenation of saved and new values, not an ambiguous chain.
- Checkpoints: allowed long inline code to wrap on narrow screens while keeping
  executable preformatted blocks scrollable and copyable.
- Persian labels: corrected typography in diagrams and matrix captions rather
  than inheriting a code font for prose.
- Entry motion: removed the initial opacity delay so useful content is always
  immediately visible.

## Release and limits

Run `python -B prepare_release.py` for the complete static release gate: 15
release/design tests, 28 JavaScript tests, syntax checks for all five generated
scripts, and validation of all generated HTML/local links/Python sources.
Five new design tests cover complete brief assignment, controlled variety,
preserved titles/objectives, and exact code text for examples and model files.

The final gate passed all of those checks: 250 HTML pages, 19,390 local
references and 57 compiling Python files. All 257 public archive members were
independently verified by size and SHA-256 against `release/manifest.json`.
The release ZIP SHA-256 is
`9023eacaa553d0766d4874c5941b893f6c878c221c8e0251b45c219867966637`.

The model and executable lesson sources were not changed, so the large PyTorch
environment was not reinstalled and the model training suite was not rerun in
this visual pass. Earlier verified CPU results remain historical evidence, not
new runs. GPU behavior remains unverified.

The configured Sites project returned `NOT_FOUND`. The existing hosting
configuration has been preserved; no new project, upload, audience change,
commit, push, or remote deployment was performed. The local static release is
ready for the existing publication workflow once project access is resolved.
