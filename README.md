# From Python to Mini-GPT

A Persian, beginner-oriented HTML textbook with a transparent PyTorch Mini-GPT
with 92 dedicated learner laboratories and 12 optional review notebooks. Each HTML
lesson leads directly to its coding laboratory. The 15-part path now continues
through response-only SFT, context/retrieval, memory, tools, verification and a
bounded offline model-plus-system project.

- [Open the built book](dist/index.html)
- [Windows installation](docs/WINDOWS_SETUP.md)
- [Jupyter setup and lesson-to-notebook map](docs/NOTEBOOKS.md)
- [Current review and verification](docs/REVIEW.md)
- [Learning-time assumptions](docs/LEARNING_TIME.md)
- [Curriculum](docs/CURRICULUM.md) and [review coverage](docs/COVERAGE.md)
- [Model commands](mini_gpt/README.md)
- [Deployment](docs/DEPLOYMENT.md)

## Work from the project root

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe run.py
```

Reuse an existing environment instead of recreating it. After activation, daily use
is **`python run.py`**. It builds HTML when needed and starts both the book
(http://127.0.0.1:8000/) and authenticated JupyterLab (http://127.0.0.1:8888/lab).
The learning desk opens automatically; each lesson button opens its exact notebook.
Ctrl+C stops both owned services. The static book remains readable without Python. To build a verified
release, install the model dependencies, ensure Node.js is available, then run:

```powershell
python -B -m unittest discover -s tests -v
python -B -m tools.prepare_release
```

The release command builds in staging, checks site/editorial rules and browser
logic, validates links, and packages only public files. To execute the Jupyter
labs as well, install `requirements-notebooks.txt` in the same environment and
run `python -B -m tools.verify_notebooks --mode student` and then
`python -B -m tools.verify_notebooks --mode solutions`. Student checks report
unfinished TODOs explicitly; separate reference checks must pass.

## Project map

| Directory | Purpose |
| --- | --- |
| `book_src/` | Authored lessons, shared educational surfaces, glossary and local assets |
| `mini_gpt/` | The actual model, training/evaluation/generation commands and learning stages |
| `notebooks/` | Independent experiments linked to the lessons |
| `data/` | Small sample corpus and real inspection fixture |
| `tools/` | Build, validation, release and notebook-verification commands |
| `tests/` | Model tests; `site/` for publishing/editorial tests; `browser/` for JavaScript |
| `docs/` | Current setup/curriculum/review guides; older reports under `history/` |
| `dist/`, `release/` | Generated static site and deployment package |
| `runs/` | Ignored personal experiments; not included in public releases |
| `archive/` | Local recovery snapshots; not published |

No cloud account, GPU, remote dataset or downloaded model is needed for the
learning path. Existing run data and historical recovery snapshots are retained.
Deployment publication is separate from producing a verified local release.
The system's scripted fixtures test component integration, not learned model
capability. Real Mini-GPT generation is available separately and can fail its
output contract; it never silently falls back to a canned answer.
