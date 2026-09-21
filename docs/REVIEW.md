# Curriculum, editorial and laboratory review — 2026-09-21

The active course has 76 lessons, 36 chapters, 10 parts and 10 checkpoints:
86 progress units. The HTML book remains the primary path. Twelve independent
Jupyter notebooks provide experimental stops; they do not replace the chapters.

## Review and corrections

All lessons, exercises, answers and project connections received a fresh review
across foundations, language/Attention/architecture, and training/post-training.
The main editor integrated the results and reread the complete
`persian-educational-writer` skill for the final pass. Its guidance led to concrete
changes: examples before API details, explicit prerequisites, selective glossary
links, fewer forced metaphors, precise axes, and less repeated explanation.
Strong existing passages were retained rather than rewritten for difference.
This is an expert editorial review, not a learner usability study.

Material changes:

- Clear Regression/Classification and numeric-ID distinctions; improved basic
  probability, axis/index, and numerical-stability explanations.
- New `26a-sequence-models`: token-only v1 versus fixed-window MLP versus RNN;
  a real v1 comparison and a tiny order-sensitive projection. It does not claim
  that all MLPs are incapable of sequence processing.
- Cross-attention now has a translation example, distinct Q/K/V sources and
  rectangular output/source dimensions. Decoder architecture is distinguished
  from tokenizer decoding.
- New required `62b-lifecycle`: GPT/model versus assistant/product, Base versus
  Foundation, Alignment, and pretraining/SFT/preference stages. RLHF and DPO are alternatives,
  not mandatory successive stages; LoRA and RAG belong to different decisions.
- Mean Loss no longer implies every target probability improves. SFT explains
  automatic pretraining targets, target-aligned masking, selected-target means,
  and prompt gradient paths; no unsupported SFT API is advertised.
- Repaired stale lesson-number references, version/milestone confusion, setup
  paths and changed-title references. Existing lesson URLs and previous/next
  navigation are preserved.
- Repeated glossary occurrences retain return anchors without all becoming
  links. Jupyter's kernel is not incorrectly linked to the hardware Kernel entry.

The [lesson ledger](COVERAGE.md) covers every lesson in curriculum order.

## Curriculum benchmarks

The [Howsam deep-learning syllabus](https://howsam.org/downloads/deep-learning-course/)
supports a progression through prerequisites, neurons/MLP, training diagnostics,
sequence models and Attention. It informed the targeted prerequisite bridges;
unrelated vision, GAN and graph-network material was not imported into this book.

The [Howsam ChatGPT-from-scratch page](https://howsam.org/downloads/implementing-chatgpt-from-scratch-with-pytorch/)
places implementation/training before instruction and preference tuning. Its
entry assumptions are more advanced than this book's, so they are not a reason
to omit beginner foundations. Some advanced advertised extensions are conditional;
this review does not claim every advertised item is already published.

The requested [Maktabkhooneh course](https://maktabkhooneh.org/course/یادگیری-ماشین-پایتون-mk1318/)
was checked, but a reliable lesson outline was not available in the retrieved
content. No detailed syllabus coverage is attributed to it.

Technical distinctions were checked against primary sources, including the
[Foundation Models report](https://arxiv.org/abs/2108.07258),
[InstructGPT](https://arxiv.org/abs/2203.02155),
[DPO](https://arxiv.org/abs/2305.18290), and
[LoRA](https://arxiv.org/abs/2106.09685). These are conceptual references, not
claims that Mini-GPT implements their complete training systems.

## Laboratory integration

The [notebook guide](NOTEBOOKS.md) lists every notebook and connected lesson.
Coverage: matrix products; probability/Loss; tensor shapes; gradients/manual
descent; first network; tokenization/Embedding/position; explicit Attention math;
causal masking; actual multi-head merge/projection; actual Pre-Norm block;
Mini-GPT training/inspection; generation and Sampling.

All 114 cells were read for educational and technical consistency. The 47 code
cells execute independently by notebook. Exercises require predictions and
controlled changes; expected failures are caught explicitly. The real model is
imported, not copied into a second implementation. Training curves and samples
come from that execution; no invented output or quality guarantee is presented.

Only JupyterLab and Matplotlib are added as direct optional requirements. The
notebook requirements include the existing base requirements. First-cell root
and interpreter diagnostics, clean-kernel instructions, Windows activation
fallback, kernel-selection recovery, hygiene and localhost book links are
documented. Earlier lesson callouts explicitly defer execution until the named
final prerequisite; main lesson navigation stays intact.

## Structure and cleanup

Build/validation/release commands moved into `tools/`; release tests moved into
`tests/site/` and browser tests into `tests/browser/`. Static source assets and
fonts are grouped in `book_src/assets/`. Current guides are in `docs/`, older
reports in `docs/history/`; the root README is now a short entry point.

The old empty deployment-test folder was removed after its files were moved.
Personal `runs/`, source-control state and useful recovery snapshots were
preserved. A pre-review snapshot is retained in
`archive/pre-curriculum-review-20260921.zip`. Temporary execution/rendering
artifacts are not release inputs. The public inventory rejects model checkpoints,
private files, unlisted notebooks and executed notebook outputs.

## Verification evidence and limits

- All 12 notebooks executed top-to-bottom in fresh kernels in a newly created
  Windows CPU environment and again from the independent download extracted into
  a path containing spaces and Persian text. Interpreter and root are asserted.
- Measured notebook outputs include 10 figures. Source notebooks retain no
  outputs or execution counts; intentional exceptions do not become uncaught
  notebook errors. Explicit kernel shutdown avoids leaving child kernels alive.
  Verification uses a temporary IPython profile rather than the learner's own
  profile or command-history database.
- All 39 model/teaching tests pass, including 65 standalone lesson programs, all 23
  milestones, causal-prefix checks, sampling, actual trace equivalence and exact
  CPU resume. The download intentionally skips the book-source-only test.
- All 36 site tests and 29 JavaScript tests pass. Site tests cover all previous/next/parent links, glossary returns, source
  preservation, laboratory mappings and notebook package hygiene. Browser logic
  tests cover numeric demonstrations, inspection schema and journal boundaries.
- New/modified pages were checked at 320 and 768 pixel viewports. Notebook HTML
  previews revealed Jupyter's left-aligned paragraph override and ambiguous
  mixed-direction identifiers; explicit paragraph alignment and isolated LTR
  inline code/lesson IDs correct those issues.
- Deterministic asset-version queries prevent old cached manifests from showing
  the previous 84-unit total. Existing browser progress is preserved.

Tested versions: Python 3.11.0, PyTorch 2.14.0+cpu, JupyterLab 4.6.3,
Matplotlib 3.11.2, nbclient 0.11.0 and nbformat 5.11.1. A fresh virtual environment
is not a fresh Windows VM. Other version combinations, GPU, ARM and other OSes
were not exhaustively tested. Execution was verified with real Jupyter kernels;
the graphical JupyterLab menu workflow was documented, not automated end-to-end.
No notebook remains unverified in the tested configuration.

The configured Sites project still returned `NOT_FOUND`. Its configuration was
preserved; no replacement project, audience change or publication was attempted.
The current build validates 486 HTML pages, 76 lessons, 224 glossary entries and
507 public files. The local static release is ready for a separately authorized deployment after
access to that existing project is resolved. Current hashes and file inventory
are generated in `release/manifest.json` and `release/SHA256SUMS.txt`.
