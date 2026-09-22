# Current learning and terminology audit — 2026-09-22

This pass continued the existing project. It did not restart the book or replace
its Mini-GPT implementation. All 76 lessons were reviewed. Thirty-six lesson
records changed; 40 already-strong records were retained. The comparison against
the pre-pass snapshot preserves every lesson ID/order/path, all 76 executable
lesson examples, all 34 formula blocks, and every original answer/self-check.

## What learners now receive

- 76 dedicated lesson notebooks, from article reading time through RAG.
- All 12 earlier notebooks retained and extended as optional synthesis reviews.
  Their 47 original code cells remain verbatim.
- 88 notebooks, 1,698 cells: 751 code and 947 Markdown.
- Two learner tasks per notebook (implementation and repair), predictions,
  an independent one-factor experiment, a deliberate failure, checks, reflection,
  and an explicit connection to the corresponding Mini-GPT stage.
- Student stubs are valid Python and report INCOMPLETE. Reference definitions
  are on separate HTML answer pages and in author-only exercise specifications;
  no notebook silently fills in a learner answer.
- Forty primary notebooks directly import actual mini_gpt modules; earlier
  prerequisites use small standalone Python/PyTorch examples. SFT, LoRA, DPO,
  RAG and cache examples are explicitly bounded teaching models, not advertised
  as production capabilities of the training CLI.
- Review caught near-complete answers in neighboring demonstrations. Eight
  notebooks were adjusted and re-executed after the full execution pass.

The original registry is extended, not duplicated: notebook metadata and
book_src/laboratories.py feed the lesson callouts, full notebooks.html index,
launcher routing, and window.BOOK.laboratories in dist/assets/manifest.js.
A primary entry records stable lesson ID, HTML path, notebook path, learning goal
and project stage. Build-time checks reject missing or duplicate primary mappings.

The dedicated callout appears immediately after the lesson explanation. The
first lesson now starts with a concrete article-card problem, a three-row
dataset and coefficients 1/2/3 before naming Model, Parameter, Target and Loss.

## One environment and one daily command

Install requirements-notebooks.txt in the project .venv once. Then run
`python run.py` in that environment (or
`.\\.venv\\Scripts\\python.exe run.py` without activation).

- Book: http://127.0.0.1:8000/
- Learning desk: http://127.0.0.1:8000/start.html
- Laboratory index: http://127.0.0.1:8000/notebooks.html
- Jupyter: http://127.0.0.1:8888/lab, authenticated by a new private token per run.

The launcher builds missing/stale HTML, starts both local-only services with its
exact Python, registers only the project kernel for that session, and opens the
desk. It does not regenerate notebooks or overwrite learner implementations.
It refuses occupied ports instead of killing existing processes. Ctrl+C has an
owned-process cleanup path; authenticated Jupyter shutdown closes its kernels,
and an unexpected Jupyter exit also closes the book server.

The learning-project ZIP now includes source/build files, model, data, all
notebooks and setup documentation. Windows 10/11 instructions describe one
installation and one launch, with explicit-interpreter commands that do not
require changing PowerShell execution policy. Private directories and personal
unmapped notebooks are not public course assets.

## Terminology: 224 concepts audited

The inventory is in the existing glossary system:
`terminology_inventory()`, `USAGE_DECISIONS`, `USAGE_SOURCES` and
`CORRECTED_CONCEPTS` in book_src/glossary.py. It records the current English
identity, aliases, usage context, A–G category, chosen form, confidence and
evidence where a usage decision was uncertain.

There are **20 concept-level corrections** (not a count of repeated string
substitutions), plus a renderer policy preserving **16 conventional Persian
mathematical/educational names**. First-introduction Persian glosses now survive
normalization. Code identifiers, formulas, URLs and API spellings are protected.

| Previous wording / ambiguity | Decision |
| --- | --- |
| یادگیری با هدف مرجع | Supervised learning (یادگیری نظارت‌شده) |
| descriptive self-supervision label | Self-supervised learning (یادگیری خودنظارتی) |
| تنظیم تکمیلی | Fine-Tuning (تنظیم دقیق); SFT uses تنظیم دقیق نظارت‌شده; PEFT explanation aligned |
| Inference as prediction alone | Use of a fixed-parameter model; preserve Prediction as a separate term |
| One-hot as تک‌روشن | Retain One-hot and explain one component is 1, all others 0 |
| Gradient as generic derivative/error | Keep Gradient/گرادیان; distinguish scalar derivatives and Gradient flow |
| Gradient clipping gloss | Describe norm clipping accurately; do not imply component-wise clipping |
| Autograd recording gradients | Recording operations/the graph; backward computes gradients |
| Q/K/V as learned transformations | Computed outputs of learned Projection parameters |
| position ID as added scalar | Learned position vector; ID sequences already have order |
| buffer as immutable data | Non-parameter Tensor/state; buffers need not be immutable |
| independent FFN per position | The same shared-weight FFN applied separately |
| bare Head as output layer | Language-model head versus Attention Head |
| generic error as Loss | Name Cross-Entropy/Loss where that actual metric is meant |
| Softmax subtraction wording | Subtract the maximum from scores, not scores from the maximum |
| literal Pre-Norm captions | Pre-Norm plus the explanation of where normalization occurs |

The main English-first vocabulary remains Tensor, Token, Tokenizer, Embedding,
Attention, Transformer, Gradient, Loss, Optimizer, Logits, Softmax, Checkpoint and
Context Window where it helps readers connect to code and outside courses.
Conventional Persian terms such as شبکهٔ عصبی، بردار، ماتریس، مشتق، ضرب داخلی
and قاعدهٔ زنجیره‌ای are not forcibly Anglicized. English and Persian remain
paired when useful: Attention/توجه, Encoder/رمزگذار, Decoder/رمزگشا and
Residual Connection/اتصال باقی‌مانده. Descriptions are not presented as invented
formal Persian terminology.

Usage evidence includes independent authored educational material, not search
result counts: [university ML/NLP assignment](https://dsp-lab.ir/wp-content/uploads/2025/11/ML4NLP-HW1-1404-1.pdf),
[Datayad's supervised-learning lesson](https://datayad.com/supervised-machine-learning/),
[Howsam's LLM course](https://howsam.org/downloads/implementing-chatgpt-from-scratch-with-pytorch/),
[Tehran Data's LLM curriculum](https://tehrandata.org/courses/llm/) and
[AvalAI's fine-tuning documentation](https://docs.avalai.org/fa/guides/fine-tuning).
These establish usage; unrelated technical claims on those pages were not copied.

Ambiguous cases were not overclaimed:

- Context Window: پنجرهٔ زمینه / پنجرهٔ بافت vary; evidence for a dominant Persian
  name is limited. Keep English plus a functional explanation.
- Pre-Norm: پیش‌نرمال / نرمال‌سازی پیشین have sparse independent usage. Keep
  Pre-Norm consistently; this is a clarity decision, not a frequency finding.
- Gate: university دروازه and Howsam گیت/دریچه coexist. Keep Gate and the
  explanatory دریچه; do not call the existing explanation incorrect.
- Embedding: تعبیه and vector-oriented descriptions both occur. Keep Embedding
  and explain the operation; do not equate every Representation with it.

The same prose-only policy applies to notebook Markdown. Legacy glossary aliases
remain for lookup compatibility, not as recommended new labels. Source review
also covered all parts, reference pages, visual/UI captions and learner docs.
Random rereads sampled foundations, Attention, Transformer, Mini-GPT and
post-training instead of treating grep replacements as an educational review.

## Verification and limits

- All 88 student notebooks executed in fresh kernels; both stubs remained
  incomplete as intended. All 88 reference-solution executions passed.
- Eight revised notebooks were then rerun in both modes: 16 further fresh
  kernels, 144 executed cells including verification assertions, no errors.
- All 39 model tests, 58 site/launcher/editorial tests and 29 browser-math/journal checks passed.
- Final reproducible static release: 597 public files in release/book-site.zip.
  SHA-256: 5b32ecd27cf90526e1631da1c3d66d495f2e41e94764eb95a2aa0e96f7f9a714.
  Release checks passed; nothing was published.
- HTML validation: 500 pages, 76 lessons, 36 chapters, 86 progress units,
  47,273 local links/assets and 80 Python files; no broken targets or duplicate IDs.
- Live smoke test verified book HTTP, authenticated Jupyter, and exact project
  interpreter. Browser inspection verified direct lesson-to-notebook opening,
  rendered Persian text, TODOs and project-kernel selection.
- Live authenticated shutdown with an active notebook kernel returned 200 and
  freed both listening ports. The terminal automation did not reliably deliver
  Ctrl+C, so a physical PowerShell Ctrl+C interaction is not claimed as tested.
  Lifecycle and failure cleanup also have automated tests.
- Learner ZIP was extracted under a path with spaces and Persian characters:
  environment check, actual imported module paths, full build, 88 byte-identical
  copied notebooks and unchanged source notebooks all passed.
- Environment: Python 3.11.0, CPU PyTorch 2.14.0+cpu, JupyterLab 4.6.4,
  matplotlib 3.11.2. No notebook was left unexecuted.

This is an expert educational review, not a beginner usability study. Assertions
cover representative cases, not every possible learner implementation. A fresh
Windows 10/11 VM install and GPU execution were not performed; extraction reused
the verified local .venv. Jupyter return links have valid exact lesson targets. The in-app browser opened
the returned lesson in a separate book tab; that destination was subsequently
verified. It did not replace the notebook tab.
Public hosting was not changed; Jupyter must never be deployed as a public book
service.

---

# Previous recorded review — 2026-09-21 (historical)

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
