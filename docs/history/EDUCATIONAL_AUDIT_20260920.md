# Final educational audit — 2026-09-20

The active book has **74 lessons, 36 chapters, 10 parts, 10 checkpoints and 84
progress units**. The 221 connected concept pages retain English technical names
with Persian explanations. The existing lesson URLs, learner progress/journal,
model architecture and historical editions are preserved.

## Educational outcome

The full source/rendered lesson and answer sequence, titles, exercises, visual
briefs, checkpoints, glossary, references and Mini-GPT implementation were
reviewed for a Python programmer with no assumed DL/PyTorch/math background.
[EDUCATIONAL_COVERAGE.md](EDUCATIONAL_COVERAGE.md) records all 74 lesson decisions
and their prerequisite contracts. This is a rigorous editorial/code audit, not a
claim that every learner will succeed without practice or instructor feedback.

Added two lessons where a short insertion would not close the gap:

- `05a-vector-operations`: Scalar/vector operations, subscripts, summation,
  element-wise versus Dot product, Weighted sum and Python-list traps.
- `12b-neuron`: a complete manual Neuron/Layer example, ReLU, prediction, Loss,
  branching sensitivities, Parameter update and a zero-Gradient counterexample.

Significant expansions cover probability/logarithms, NLL derivation, Loss surfaces,
branching Backpropagation, silent Broadcasting mistakes, Linear/Module and XOR
transitions, numerical Q/K/V Projection, Variance/scaling, explicit causal masking,
FFN construction, scalar-before-Jacobian Residual reasoning, constant LayerNorm,
independent block stacking, class-axis flattening, Parameter counts and detach
debugging. Every standalone reference program is executed, not merely compiled.

The complete Attention lab now follows scaling/Value/masking instruction. The
first causal test uses the already-taught SingleHead instead of a premature
finished Transformer. Full-model causality returns after blocks are introduced.
Math/PyTorch project labels explicitly identify prerequisites; the Tokenizer
lesson points to its actual first executable stage.

Real Training commands now begin at lesson 47, before LR/Checkpoint/Resume tasks.
Checkpoint round-trip and 6-versus-4+2 Resume have usable code and commands.
Sampling examples, unknown/context diagnostics, a real failing/fixed shape case,
capacity arithmetic, cache lengths and shifted SFT Loss-mask alignment were
strengthened. Advanced topics remain explicitly optional.

## Terminology and implementation corrections

- Preserved API calls/indexing and BPE/RNN/LSTM/GRU/SGD abbreviations in prose.
- Fixed ordinary-language Token substitutions and contextual representation
  incorrectly labeled Positional Embedding; distinguished output head from
  Attention Head, and LayerNorm from Vector Norm.
- Corrected `CharacterTokenizer`, `context_length` and the actual Batch/DataLoader
  contract in the glossary. Added focused examples for ambiguous/foundational
  entries; source-derived references include the exercise question and optional
  complete lesson context instead of disconnected answer fragments.
- Added explicit `evaluate --test-text` for separate UTF-8 Test documents, using
  only saved Vocabulary. Original Validation mode still enforces corpus identity.
  Different file content is explicitly not proof of independent data.
- Added stdlib-only UTF-8 console configuration at CLI boundaries, including v0,
  milestones and stage wrappers. Labeled milestone 0's unsmoothed frequencies.
  No neural architecture or training algorithm was changed.

## Windows process

The complete copyable procedure is [WINDOWS_SETUP.md](WINDOWS_SETUP.md), also
included in the model ZIP. Its Persian counterpart is `dist/windows.html`.
The standalone download has its own README with valid in-package links.

From the extracted model/project root, in PowerShell or CMD:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -B -m mini_gpt.smoke_test
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Only readers need a browser. Local serving uses Python with no packages:
`python -m http.server 8000 --bind 127.0.0.1 --directory dist` from the full checkout;
use `--directory .` from the extracted static site. Open `http://127.0.0.1:8000/`
and stop with Ctrl+C. Node 24 LTS is author-only for `prepare_release.py` checks.
No npm install, backend, database, CUDA toolkit or global execution-policy change.

## Verification evidence

- Fresh isolated Windows venv: Python 3.11.0 x64, pip upgraded from 22.3 to 26.2.1,
  official CPU PyTorch 2.14.0+cpu. The initial old-pip dependency-metadata failure
  was reproduced; upgrading pip fixed installation. No global packages changed.
- **38 model/example tests passed** in the full checkout, including all 64
  standalone lesson programs, all 23 milestones, Attention/gradient contracts,
  checkpoint integrity, mode restoration, Sampling and exact CPU Resume.
- Extracted `mini-gpt-project.zip` into `.verification/بسته آزمون Windows`:
  smoke test passed; model suite passed (37 tests plus the intentionally skipped
  book-source-only example test). Counting v0/milestone 0 ran with base Python
  without PyTorch. PowerShell and CMD activation selected the isolated interpreter.
- The documented tiny run used 4,720 Parameters, Vocabulary 40, six CPU steps:
  Validation Loss 3.684826 and unknown rate 0.0145. Generation and inspection
  succeeded; Resume to total eight steps gave Validation Loss about 3.6818.
  These are workflow checks, not claims of language quality.
- The exact authored six-step versus four-plus-two recipe produced identical
  saved model tensors. Explicit independent-Test mode ran on a **synthetic tool
  fixture**, not a quality benchmark. Tests cover Vocabulary nonmutation,
  unknown rate, corpus identity, short/empty input and mode exclusivity.
- Browser: all 84 learning units checked at 320 and 768 px; all 221 glossary
  routes at 320 px, including all 165 expanded source-context panels; no
  document-width overflow. Windows guide checked at 320/768 px. New lesson
  layouts, exact glossary return anchors and lesson-to-checkpoint navigation
  inspected. Existing user records were not changed.
- Final static ZIP extracted, checked against every manifest/hash and validated
  again; its root-directory local server served pages, scripts and the bundled
  font. Final model ZIP retested in `.verification/بسته نهایی Windows` with the
  same 37 passes and one intentional book-source skip.
- Final release gate and archive counts/checksum are recorded in `PROGRESS.md`
  and `release/manifest.json`; all local HTML/anchor/CSS references are checked,
  including URL case on this case-insensitive Windows filesystem.

## Cleanup and remaining limits

Removed six verified-unused legacy `.glossary` CSS rules. The tool-created
`.verification` tree (679,931,783 bytes) was removed after evidence collection:
temporary venv, extracted test copies, synthetic fixtures, generated bytecode
and diagnostic scripts. None belongs in the public build. Temporary preview
servers were stopped and the temporary browser viewport/tab were restored/closed.
The pre-audit backup is `archive/pre-educational-audit-20260920.zip`.
Existing educational sources, author tests, model runs, data, useful documentation
and historical backups are deliberately retained. They are excluded from static
deployment; the intentionally downloadable learner code/tests are inside one ZIP.
No previous user experiment or learner browser record was deleted.

No clean Windows VM or OS/runtime installer was available. This verifies an
isolated environment on the existing Windows host, not a fresh Windows 10/11
installation. GPU/CUDA and Windows ARM are untested. Optional NumPy was absent;
its PyTorch warning did not prevent tests. Direct `file://` browser testing was
blocked by the browser tool's URL policy and was not bypassed; localhost works.
File-mode support is based on static relative paths and classic scripts, while
storage/download behavior remains browser-dependent.

The configured Sites project returned `NOT_FOUND`. Its metadata was preserved;
no replacement project, audience change or remote publication occurred. The
validated static archive is the deployment-ready deliverable.
