# Complete model-to-system learning journey — 2026-09-23

This is the current report. Reports below the historical divider describe earlier
editions, not the current totals. The repository's `persian-educational-writer`
skill guided the new Persian explanations: concrete failure first, a small
experiment next, and an explicit boundary between a model and its surrounding
software. The existing foundation was reviewed and preserved, not restarted.
Pre-change evidence is retained in `.verification/before-system-journey.zip`.

## Requested 24-point report

### 1. Original curriculum structure

76 lessons, 36 chapters, 10 parts, 10 checkpoints and 86 progress units. There
were 88 notebooks: 76 primary labs and 12 optional reviews. The first nine parts
already supplied a strong Python-to-Mini-GPT path. The final part introduced the
training lifecycle, scale, cache, SFT, LoRA, preferences and a small RAG example,
but did not provide a continuous, executable path through the surrounding system.

### 2. Final curriculum structure

92 lessons, 49 chapters, 15 parts, 15 checkpoints and 107 progress units;
104 notebooks. Parts 1–9 remain in their existing order. Parts 10–15 now cover
instruction-oriented training; context and grounded retrieval; conversation and
external memory; tools and verification; bounded orchestration and evaluation;
then performance, local serving and an integrated project. See
[the complete inventory](JOURNEY_INVENTORY.md) and [prerequisite map](COVERAGE.md).
All 76 previous lesson IDs and published URLs remain valid, including moved
lessons whose legacy URL still contains `part-10`. Navigation, breadcrumbs and
the teaching order use the new conceptual grouping.

### 3. Lessons added

| ID | Concrete educational purpose |
| --- | --- |
| `65a-sft-lab` | Actually pretrain and response-only fine-tune Mini-GPT; compare measured outcomes. |
| `68-context` | Pack a complete serialized request within a model-token budget, with output reserve. |
| `69-chunks` | Split documents with source IDs/offsets; inspect overlap and boundary failures. |
| `70-vectors` | Build count vectors/cosine ranking, then train a tiny embedding example. |
| `71-grounding` | Separate retrieved evidence, citations, supported answers and abstention. |
| `72-history` | Serialize conversational roles and observe history growth. |
| `73-summary` | Compare history with a lossy extractive summary and trace omissions. |
| `74-memory` | Explicitly select, persist, retrieve and forget external records. |
| `75-tools` | Validate structured proposals and dispatch only allowed local functions. |
| `76-reasoning` | Decompose a task and distinguish calculation checks from solving the right problem. |
| `77-candidates` | Compare candidates, voting, verification and inference budgets. |
| `80-controller` | Connect model proposals, observations and tools in a bounded loop. |
| `81-system-eval` | Evaluate components and full-system regressions separately. |
| `82-performance` | Measure latency/throughput and inspect quantization error without invented speedups. |
| `83-deployment` | Reuse a loaded model and enforce request/output limits locally. |
| `84-capstone` | Integrate the components and deliver evidence, including real-model failures. |

### 4. Lessons removed

None. Obsolete claims that the project ends at the old RAG lesson or that no
instruction-training implementation exists were removed, not entire lessons.

### 5. Lessons merged

None. Existing focused lessons did not need forced consolidation.

### 6. Lessons split

No existing lesson was mechanically split. The formerly broad final part was
reorganized into six stages, using the existing lessons plus justified bridges.

### 7. Existing lessons substantially improved

The lifecycle (`62b-lifecycle`), SFT (`65-sft`), preferences (`66-preference`) and
RAG (`67-rag`) endings/project connections now lead to executable follow-on work.
The scale/cache lessons are taught after system integration. Shared introduction,
roadmap, laboratory directory, project map and final checkpoints were updated.
All 92 lessons gained workload metadata. The first 69 lesson records and their
order are otherwise exactly unchanged; all 31 original model Python files are
byte-identical to this pass's backup. Original lesson code was not rewritten.

### 8. Reasons for major curriculum changes

Instruction behavior needs a measured training example, not only a loss mask.
Retrieval needs a context budget before a larger index. Persistent memory needs
history and an honest account of summary loss first. A controller needs validated
tools, observable failures and verification before it deserves an agent-like
loop. Performance claims need a defined workload after that system exists.
These dependencies determine the order; fashionable frameworks do not.

### 9. Final Mini-GPT/project evolution

The existing 23 runnable core milestones remain unchanged. Later lessons are
labelled project extensions, not fictitious additional milestone commands.
The same Mini-GPT architecture continues through response-only training and a
real generation adapter. Nine small runtime modules add instruction training,
context, retrieval, memory, tools, reasoning, orchestration, evaluation and
performance around it. Nothing silently replaces a failed model with an answer.

The default SFT experiment uses 40 pretraining and 160 fine-tuning steps. In the
verified CPU run, training response loss went from 2.196958 to 0.002498 and exact
answer accuracy from 0.5 to 1.0; two held-out *rewritings of the same facts* went
from loss 2.241845 to 0.005010 and accuracy 0.5 to 1.0. This measures a tiny
yes/no task, not new-fact generalization or general instruction following.

The 64-position demo cannot hold the controller protocol. The capstone therefore
also provides a separately configured 2048-position miniature, allocated before
training with an explicit vocabulary. Its short 2+4-step experiment actually
passes a 675-token prompt through generation, but the observed response
`ببنببببن` fails the action schema. Its longest training sequence is only 42
positions. Larger allocated context is not evidence of long-context competence.
The failed result is kept; it is never replaced by a scripted success.

### 10. RAG coverage

External question/document scenario → existing keyword retrieval → stable chunks
and overlap → count-vector cosine ranking → a tiny learned embedding contrast →
context packing → source citations/grounding → abstention and evaluation.
No vector database or service is required. The learned four-term example has no
held-out semantic benchmark; its decreasing loss is not proof of useful semantic
retrieval. The extractive baseline and scripted integration fixtures are clearly
identified as non-generative teaching aids.

### 11. Memory coverage

History, summary, persistent records and retrieval-based selection are distinct
from learned weights and KV cache. Summary provenance/omitted turns expose loss.
The controller sees only explicitly selected memory keys; default selection is
empty. Writes and forgetting are explicit. The caller must provide an authorized
user's store: this is not a production identity/access-control implementation.

### 12. Reasoning coverage

Manual decomposition, candidate plans, executable checking, incorrect-premise
failures, disagreement, voting/self-consistency and bounded inference spending.
Training changes weights; extra inference calls do not. Fluent intermediate
text is not a faithful trace of hidden computation. Majority agreement is not
correctness, especially with correlated candidates. Small deterministic examples
isolate these ideas without claiming to train a frontier reasoning model.

### 13. Tool/function calling coverage

The model proposes data; Python validates and executes. Strict JSON/schema and
duplicate-key/nonfinite rejection precede a three-function arithmetic allowlist.
Arguments and calls are bounded; errors and actual results return as observations.
No `eval`, shell, network, arbitrary filesystem access or framework dependency is
used by the teaching tools. The model itself does not execute a function.

### 14. Agent/system architecture coverage

`run_assistant` connects explicit context/retrieval/memory, a backend proposal,
validation, dispatch, observations, verification and termination. Step/tool limits,
repeated-action detection, missing-context and invalid-action stops are observable.
Required protocol and prior tool results cannot silently disappear when packing.
`MiniGPTBackend` uses the real tokenizer/model/generation; `ScriptedFixture`
tests controller behavior only. “Agent” is presented as variable ecosystem usage,
not a universal definition. Prompt delimiters are not a security boundary.

### 15. Evaluation coverage

Existing held-out loss/perplexity and generation experiments continue. New small
datasets distinguish raw retrieval recall from evidence actually retained in the
final context, exact answer checks, allowed citations, tool behavior, verification
and full-system outcomes. A finish event without an independent verifier remains
unverified. Regression cases include malformed actions, exhausted budgets,
unsupported answers and memory/context omissions. Evidence is reported per layer.

### 16. Performance/deployment coverage

Scale and KV-cache teaching precede measurement, quantization error and local
request handling. CPU median latency and throughput have stated boundaries;
model loading, HTTP/network time and queueing are excluded. The toy quantizer
stores `int8` even when simulating four-bit levels: it is not packed four-bit
storage or a demonstrated acceleration. No real KV cache was added to the core.
`InferenceSession` reuses a loaded model and rejects invalid/oversize inputs; it
is not an authenticated concurrent public inference server. The book remains a
static deployment artifact with local-only Jupyter. No public deployment was made.

### 17. Notebook changes

16 new independent primary notebooks retain the established prediction, student
TODO, explicit incomplete checks, reference implementation, comparison, variation,
break/fix and project-connection contract. Existing 88 notebooks retain all 751
code cells; 749 are identical, and only the first two optional review setup cells
drop premature PyTorch imports. IDs and lesson/HTML mapping metadata are preserved.
Current totals: 104 notebooks, 879 code cells and 1107 Markdown cells. Lesson time
estimates are shared with notebook headers. All 104 pass independently in student
and solution mode, including 11 generated figures in each mode. Student execution
does not fill in the learner's TODOs or disguise unfinished work as success.

### 18. Terminology changes

The generated glossary grows from 231 to 268 entries. New definitions cover
context budgets, chunks, cosine ranking, grounding, abstention, provenance,
history/summary/persistent memory, tools, verification and controller concepts.
Existing conventional Persian choices and searchable aliases are retained.
English technical identities are paired with functional Persian explanations;
API names stay code. The SFT/RAG glossary project links now point to actual
extensions. No claim is made that a newly invented Persian translation is the
community standard. Earlier usage evidence remains in the historical report and
the source glossary's explicit usage-decision ledger.

### 19. RTL/LTR and visual fixes

The shared protected-aware formatter remains in use for HTML and all notebooks.
Persian remains RTL; code, shapes and short math remain LTR-isolated. New time
headers reuse the restrained page design, with a collapsed optional breakdown
and responsive rows. The homepage's ambiguous “remove the cover” wording now
explicitly names Mask, avoiding an accidental Vector glossary link. Browser
checks cover early foundations, Attention, Mini-GPT, context/grounding, memory,
reasoning, controller and capstone pages at desktop/narrow widths. No global
nowrap, blanket nonbreaking spaces or new display dependency was introduced.

### 20. Reading/learning-time methodology

One source module assigns every lesson an explicit activity profile. It combines
prose reading, difficulty-adjusted concept/manual work, code tracing, notebook
experimentation/debugging, exercises/self-testing and prerequisite recall.
Profile selection accounts for mathematical novelty and integration burden;
reading alone is a small part of most labs. Checkpoints add separate synthesis
time. Exact minute endpoints aggregate; displayed large ranges round outward.
[The method and assumptions](LEARNING_TIME.md) are public in the book and docs.
These are editorial planning ranges, not measurements of Persian learners.

### 21. Total estimated book learning time

Core path including all 15 checkpoints: **5960–10460 minutes**, or
**99 hours 20 minutes–174 hours 20 minutes**, displayed as **99–174.5 hours**.
This excludes optional review notebooks (45–90 minutes each), initial setup,
long training runs, papers, open-ended projects and breaks. Setup may require
roughly 1–3 hours but troubleshooting can exceed that. No completion guarantee
or fixed calendar-week promise is made.

### 22. Per-Part estimated time

| Part | Scope | Display range, hours |
| --- | --- | --- |
| 1 | Python problem/model foundations | 4–8 |
| 2 | Numbers, probability and change | 8.5–15.5 |
| 3 | Computation tools and first network | 8–14.5 |
| 4 | Text and learned representations | 6–11 |
| 5 | Single-Head Attention | 9–16.5 |
| 6 | Multiple heads and Transformer block | 8–14.5 |
| 7 | Complete language model | 5–9 |
| 8 | Training practice | 10–18 |
| 9 | Generation, failures and experiments | 9.5–17.5 |
| 10 | Instruction-oriented training | 5–9.5 |
| 11 | Context and grounded retrieval | 5.5–10 |
| 12 | Conversation and external memory | 4.5–8 |
| 13 | Tools, reasoning and verification | 4–7.5 |
| 14 | Model inside a system | 4–7 |
| 15 | Cost, local execution and capstone | 6.5–12 |

Each includes its checkpoint. Rounded display ranges should not be re-summed;
the book total uses exact minute endpoints, which are tested and recorded in
[JOURNEY_INVENTORY.md](JOURNEY_INVENTORY.md).

### 23. Validation/build results

The current staging build passes: 614 HTML pages, 92 lessons, 49 chapters,
107 progress units, 72115 local links/assets and 103 compiled Python files.
89 model/system/example tests and the Mini-GPT smoke test pass. All 92 lesson
examples are covered by the shared registry. 70 site/launcher/release/editorial
tests pass. The 104 student and 104 solution runs have no uncaught errors; every
recorded source hash matches its current notebook. Reports are retained locally
as `.verification/journey-student.json` and `journey-solutions.json`.

The downloadable project inventory includes all 40 runtime modules, all 9 model
test files and all 104 notebooks. Final release preparation reran all 70 site
tests, the generated-source consistency checks, full HTML validation, JavaScript
syntax checks and all 29 JavaScript tests successfully. The public inventory is
727 files. The release archive SHA-256 is
`f843ac50c45d54e47f0b56eae6cd1935d8db10e6df3b6351a4f9a740cccd6908`;
`release/manifest.json` records every exact public-file hash. No files were
silently deleted and nothing was published.

The actual SFT and capstone notebooks were opened in authenticated local
JupyterLab. At 1280 and 625 CSS-pixel widths, the capstone's visible Persian
paragraphs are RTL/right-aligned, code editors are LTR, and the document has no
horizontal overflow. The 90–150-minute lesson/lab estimate is visible and says
not to count the notebook twice. All QA-opened documents were closed, incidental
unsaved metadata discarded, and generated-source/hash checks passed again. The
owned local launcher was restarted with the expanded catalog; the released
capstone's laboratory link opened the matching notebook successfully. The local
book and authenticated Jupyter remain available on ports 8000 and 8888; the
temporary staging preview was stopped.
Checks use Python 3.11.0, PyTorch 2.14.0+cpu and JupyterLab 4.6.4.

### 24. Remaining weaknesses or ambiguities

This is an educational miniature, not a reliable general assistant. Tiny SFT
accuracy measures only the declared toy task; real-model controller failures
remain visible. The embedding exercise has no semantic generalization benchmark.
The memory store has no production authorization/encryption; the controller has
no complete prompt-injection defense; local inference has no HTTP authentication,
queue/concurrency or hard per-request timeout. KV cache, packed quantization,
distributed inference, GPU behavior and real-world throughput are not claimed
implemented or validated. Learning times need calibration with actual learners.
Representative browser inspection is not a claim that every frontend/device was
visually tested. A native Persian teaching/editorial review can still improve
pacing and terminology; no automatic check proves educational effectiveness.

## Research sources and why they were used

- [Rice workload-estimation resources](https://cte.rice.edu/resources/workload-estimator):
  supports separating task type, reading purpose and difficulty. It does not
  validate our Persian reading rates or exact lesson durations.
- [InstructGPT](https://arxiv.org/abs/2203.02155),
  [LoRA](https://arxiv.org/abs/2106.09685) and
  [DPO](https://arxiv.org/abs/2305.18290): maintain the distinction between
  instruction supervision, parameter-efficient adaptation and preference data.
  The course's tiny experiments are not replications of these large-scale results.
- [RAG](https://arxiv.org/abs/2005.11401) and
  [Sentence-BERT](https://arxiv.org/abs/1908.10084): ground the external-retrieval
  and learned shared-vector explanations; random vectors are not semantic search.
- [Chain-of-thought prompting](https://arxiv.org/abs/2201.11903),
  [self-consistency](https://arxiv.org/abs/2203.11171) and
  [ReAct](https://arxiv.org/abs/2210.03629): motivate decomposition, candidate
  aggregation and interleaved observations/actions, without importing a framework.
- [Unfaithful explanations](https://arxiv.org/abs/2305.04388): motivates the
  explicit warning that visible reasoning text is not guaranteed faithful.

These sources support technical boundaries, not a claim that the English-first
new terms have one universally accepted Persian translation. Existing Persian
usage sources and deliberate retained spellings remain recorded below.

---

# Historical report: final Persian reader, terminology and typography audit — 2026-09-23

This is a refinement of the current book, not a restart. The repository's
persian-educational-writer skill guided paragraph-level motivation, natural prose
and restraint: good explanations and the existing Mini-GPT progression were kept.
The source snapshot for this pass is .verification/before-final-reader-pass.zip;
older audit counts below describe earlier work.

## Concrete final report

| Item | Result |
| --- | --- |
| 1. Lessons audited | All 76, in continuous context: 43 records refined, 33 retained. IDs, sequence, chapters and routes preserved. |
| 2. Notebooks audited | All 88: 76 one-to-one lesson labs plus 12 optional reviews. All 947 Markdown cells reviewed/formatted; all 751 code cells preserved. Seven original review Markdown cells received additional factual clarifications. |
| 3. Important terminology decisions | 60 explicit decisions, including deliberate keeps, in FINAL_TERM_REVIEW. The full inventory covers 231 concepts; 21 research-backed/uncertainty decisions reference 20 recorded sources. These are not substitution counts or frequency estimates. |
| 4. Major terminology changes | کاراکتر in Python prose, with Character, Unicode Code point, Token and Token ID explicitly distinguished. Consistent Attention Head, Mask/Causal Mask, Embedding and recognizable Transformer terminology. |
| 5. English intentionally retained | Token, Tokenizer, Tokenization, Embedding, Q/K/V, Attention, Transformer, Loss, Optimizer, Checkpoint, Inference, Logits, Softmax, Dropout, Batch, Epoch, SFT, LoRA, Quantization and Perplexity remain recognizable English identities. Literal API spelling remains code, not a term to capitalize. |
| 6. Persianized forms retained | کاراکتر is the prose choice. توکن, توکن‌سازی, امبدینگ, ترنسفورمر, گرادیان, ماسک and چک‌پوینت remain accepted introductory/search forms where applicable, not competing names randomly alternated in paragraphs. Established mathematical Persian and Encoder/رمزگذار, Decoder/رمزگشا glosses stay. |
| 7. Literal/uncommon wording replaced | نویسه/نویسه‌ای becomes کاراکتر/کاراکترمحور in explanatory prose. Unnecessary تبدیل‌گر glosses are removed. Vague پوشش and سر references in lessons and shared educational UI/docs are clarified as Mask and Head. Ordinary خطا and نشانه are not automatically Loss and Token; an explicit regression protects نشانهٔ بهترشدن. |
| 8. Reader-level fixes | Distinguish all parameters from trainable parameters; explain Batch-weighted Loss correctly; keep Vocabulary inside the Checkpoint; separate Causal Mask from SFT target selection. Remove misleading “first PyTorch notebook”; compare embeddings with W_V using real variable names; explain that changing Value projection can leave Attention coefficients fixed. Glossary summaries no longer depend on orphaned “this/that” referents. |
| 9. RTL alignment | Shared notebook renderer explicitly right-aligns Persian headings, paragraphs, lists and callouts instead of merely setting direction. No blanket right-alignment or global injected stylesheet. |
| 10. Mixed directions | Code, formulas, short arrays/shapes and complete short dot-product expressions have LTR isolation. Backticks are protected before HTML parsing/term normalization, including 0<p<=1 and unknown-token spellings. English-only paragraphs and mixed table cells are classified separately. |
| 11. Line breaks | Only short Persian label-plus-first-word groups are kept together. English compounds are not split by that rule. Emphasis remains inline; long prose and code retain responsive wrapping. No global nowrap or blanket nonbreaking spaces. |
| 12. HTML/CSS | One protected-aware typography policy serves HTML and notebooks. Existing attributes/styles are merged, not duplicated; code/math/script/style subtrees remain opaque. Browser QA caught Jupyter stripping logical padding/border styles: scoped physical RTL equivalents now handle lists and quote borders. |
| 13. Glossary | 224 → 231 entries: Character, Code point, Hyperparameter, Language model, Causal language model, Generation and Mask added. Clearer Positional Embedding, Weight Decay, Resume, Perplexity, Teacher forcing and related standalone definitions; old search aliases retained without making them canonical prose. |
| 14. Validation | Full student and solution execution: 88 + 88 fresh kernels, 11 figures per pass, no uncaught errors. Student TODOs correctly remain INCOMPLETE; reference exercises pass. 39 model tests and smoke test pass. Final static release checks and exact inventory are recorded below. |
| 15. Remaining ambiguity | Do not force a single Persian name for Perplexity, Quantization, Regularization, Context Window, Gate or Pre-Norm without human editorial review. Independent sources disagree or are sparse; keep English and a precise functional explanation. |

All 66 nonempty lesson code examples and 34 formula blocks are unchanged.
All 31 Mini-GPT Python source files are byte-identical to the pre-pass snapshot.
No new dependency, training algorithm, exercise answer or notebook fallback was
introduced. All 32 API table names and the 76 diagram payloads are preserved.
Notebook header prose may change, but primary lesson/HTML/path identities do not.

## Evidence and terminology reasoning

Usage evidence is not an endorsement of every technical claim on a source page,
nor a statistical claim about the whole Persian AI community. The complete URL,
confidence and decision ledger is in book_src/glossary.py: USAGE_SOURCES,
USAGE_DECISIONS and terminology_inventory(). That inventory also records
definitions, aliases, lesson IDs and notebook paths.

- Character: the authored Persian Python text by Javad Vahidi and Ramazan
  Abbasnejad repeatedly uses کاراکتر in its string chapter
  ([faculty-hosted PDF](https://professor.masoudkargar.ir/ProfessorFile/-647a8664163678388262411520611003714.pdf)).
  The independent [Bardia AI course](https://bardia.ai/ai-course/) also uses it.
  [Python's Unicode HOWTO](https://docs.python.org/3/howto/unicode.html) supplies
  the technical boundary between code points and visible characters.
- Token, Tokenization, Embedding and Transformer: the
  [Howsam authored Transformer article](https://howsam.org/transformer/comment-page-2/)
  and [Bardia course](https://bardia.ai/ai-course/) support recognizable
  transliterations. Howsam evidence was accessible through indexed article text;
  direct retrieval of that page was intermittent. Keep English-first names plus
  Persian explanations rather than infer a universally dominant spelling.
- Checkpoint: an [author's Persian model card](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m/blob/main/README.fa.md)
  uses چک‌پوینت; the independent
  [Yaadestan reproducibility lesson](https://yaadestan.com/courses/zharfa/lessons/t2/s06-reproducibility)
  uses checkpoint. Explain saved state, not a literal photograph; keep English
  first. The latter was checked through indexed lesson text.
- Perplexity: [Tehran computational-linguistics coursework](https://dsp-lab.ir/wp-content/uploads/2025/05/CL-HW3-1403-2.pdf)
  uses سرگشتی, while the independent Bardia course retains English. That variation
  argues against replacing every occurrence with one claimed conventional gloss.
- Quantization: [Sharif MLSD coursework](https://sharifmlsd.github.io/assets/MLSD_HW2.pdf)
  retains English; Bardia uses a transliteration. Explain reduced-precision
  representation; do not declare کم‌بیت‌سازی the field's canonical term.
- Regularization: [Sharif's authored lesson 13](https://www.youtube.com/watch?v=38Ih1rLG_sw)
  uses تنظیم مدل; Khayyam Salehi's independently authored course announcement
  (URL recorded in USAGE_SOURCES) uses منظم‌سازی. Keep Regularization and explain
  its role, without turning the explanatory phrase محدودسازی into a formal name.

## Visual verification and limits

Actual JupyterLab and book pages were inspected at 1280, 900 and 625 CSS-pixel
viewport widths. In the token notebook, computed Persian paragraph/heading
alignment changed from left/start to right, while code stayed LTR/left.
The scores lesson keeps complete dot-product expressions together and in the
correct order. The opening lesson's Persian table and inline definitions were
inspected; the SFT lesson supplies advanced mixed-direction content.

A disposable, non-curriculum Jupyter fixture exercises lists, blockquotes,
English-only text, mixed table cells, code, math and emphasis. This matters
because those element types are not all present in the authored Markdown cells.
At narrow widths the Jupyter file sidebar may leave too little reading room;
the fixture is also checked with that sidebar collapsed. This pass does not
redesign Jupyter's own application chrome. Browser styling/geometry checks and
repeat-render tests complement, rather than replace, human reading.

Fresh-kernel reports are retained locally in
.verification/final-reader-student.json and
.verification/final-reader-solutions.json. Rendering uses JupyterLab 4.6.4;
execution used Python 3.11.0 / PyTorch 2.14.0+cpu. GPU behavior and other notebook
frontends are not claimed verified. No public deployment was performed.

## Final static release checks

The complete release gate passes: 64 site/launcher/release/editorial tests,
29 JavaScript tests, all five JavaScript syntax checks, 507 HTML pages,
48,073 local links/assets, and 80 compiled Python files. No broken local targets,
duplicate IDs, missing navigation or remote display dependencies were found.
The release contains 604 public files; release/manifest.json and SHA256SUMS.txt
record the exact archive identity. The regenerated 88-notebook source check also
passes after closing Jupyter and removing its incidental metadata edits.

The disposable visual fixture and its checkpoint were removed after inspection;
neither is curriculum content or included in the release. Authored notebooks
remain output-free. The browser check confirmed a 3px right quote border and no
left border after the sanitizer-compatible fix, with no prose overflow at the
tested reading widths. This is local verification, not public hosting validation.

## Earlier audit record

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
