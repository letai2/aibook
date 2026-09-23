# Lesson laboratories

The 92 lessons each have a dedicated learner notebook. The original 12 notebooks
remain as optional, broader review laboratories. HTML explains the idea; the
notebook is where you predict, write code, run, inspect, change one factor,
diagnose a deliberate bug, and write a repair.

The complete 104-notebook set now includes response-only SFT, context budgeting,
retrieval, memory, tool validation, reasoning/verification, bounded control,
system evaluation and deployment-cost experiments. These remain small offline
CPU laboratories. No framework, remote model or API key is required. Existing
notebook paths and all previous lesson URLs remain stable. Learning-time ranges
in the book already include the corresponding lab; do not add them twice.

## Install once; launch with one command

Use one Python 3.11+ environment for the book, notebooks, and Mini-GPT.
From the complete project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe run.py
```

If the environment already exists, reuse it. If activated, the daily command is:

```powershell
python run.py
```

The launcher builds stale/missing HTML without regenerating notebooks, starts
both services, and opens the learning desk:

- Book: http://127.0.0.1:8000/
- Learning desk: http://127.0.0.1:8000/start.html
- JupyterLab: http://127.0.0.1:8888/lab — use the private token URL printed by
  the launcher, or the lesson's laboratory button.
- All lesson and review labs: http://127.0.0.1:8000/notebooks.html

Keep the terminal open. Ctrl+C shuts down both owned services and Jupyter
kernels; it does not delete saved learner work. Save notebook changes first.
If either port is occupied, the launcher explains the conflict and stops without
terminating any existing process. Stop your previous server yourself, then retry.
Jupyter may open a return link in a separate browser tab; its destination is the exact lesson, not the homepage. Save work before leaving a notebook.
The fixed book port keeps notebook return links and browser progress consistent.

Both services bind only to 127.0.0.1. Jupyter authentication remains enabled.
Tokens are generated per launch, are never written into static book files, and
must not be shared. The local book redirects only known lesson/review mappings,
not arbitrary paths. This is a personal local environment, not a multi-user or
production host. Notebook code has the same local permissions as your Python.

## Finding the right lab

Click **آزمایشگاه این جلسه** directly after a lesson's explanation. With the
launcher running, this opens that exact notebook in Jupyter, using the same
Python interpreter that launched the book. No second terminal or kernel
registration is required. The first cell prints the interpreter and project root.

The metadata in each notebook extends the existing laboratory catalog:
`kind=lesson`, `primary_lesson`, `lesson_number`, `html`, `goal`, and
`stage` identify its role. A stable lesson ID determines its path:

`01-model → notebooks/lessons/01-model/lab.ipynb`

`65b-lora → notebooks/lessons/65b-lora/lab.ipynb`

The lesson number is displayed separately, so adding a lesson does not rename
old notebook paths. The build checks a one-to-one mapping. The original notebooks
keep their existing paths and IDs, with `kind=review`. They do not replace any
dedicated lesson lab. The generated laboratory index is the complete live map.
The same records are exported as `window.BOOK.laboratories` in
`dist/assets/manifest.js`; each primary record carries its stable ID, HTML path,
notebook path, goal, and Mini-GPT stage. Personal notebooks without course
metadata are not added to the curriculum or downloadable learning package.

## Student work, examples, and separate answers

Each lesson lab contains two unfinished functions: the main exercise and a repair.
Both are valid Python returning `None`. A fresh **Restart Kernel and Run All
Cells** executes the demonstrations and reports **INCOMPLETE** at these checks.
That is expected and explicitly does not mean the exercise passed.

Replace each TODO with your implementation. Rerun the function cell, then its
check. Assertions test the function on concrete cases; **PASS** means those
checks passed, not that arbitrary implementations have been formally verified.
Variation/debug cells run independently of unfinished learner code.

No learner cell silently inserts a reference solution. Optional answers live on
the separate HTML answer page, linked only after the exercises. Authoring sources
contain the reference implementations for maintainer tests, not hidden notebook
metadata or a student fallback. Do not look there before trying.

Each notebook has a prediction note and a final explanation area. Record what
you expected, what actually happened, and why the repair changes the result.
Advanced SFT, LoRA, DPO, RAG and KV-cache experiments are explicitly small teaching
implementations; they are not claims that the production Mini-GPT implements
these systems.

## Portability and source hygiene

Extract the **whole** learning-project ZIP. Keep `run.py`, `tools`,
`book_src`, `mini_gpt`, `notebooks`, `data`, and `docs` together. Unlike
the older model-only archive, the current download includes book source/build
files, so the unified launcher also works after extraction. A standalone
`.ipynb` download is a replacement within that tree, not a complete environment.

Root discovery works from the project root and nested notebook directories,
including paths containing spaces and Persian characters. Experiments default
to tiny CPU data and one PyTorch thread. They require no external data/model
downloads after installation. Temporary checkpoint experiments use private
temporary directories rather than overwriting `runs`.

Authored notebooks have no saved outputs or execution counts. Keep your personal
work; do not regenerate authored notebooks over it. The launcher never invokes
the notebook generator. Maintainers can regenerate reviewed specifications with
`python -B -m tools.build_notebooks`; this explicit author command **overwrites
lesson notebooks**, so use it only with intentional source edits and backups.

## Maintainer verification

Persian Markdown is formatted centrally by `render_markdown` in
`tools/build_notebooks.py`, using the shared protected-aware rules in
`book_src/terminology.py`. Persian headings, paragraphs and lists receive RTL
alignment; code and formulas retain LTR layout. English-only paragraphs and
table cells have their own direction. No global notebook stylesheet is injected.
Jupyter strips logical border/padding properties, so the formatter uses
direction-scoped physical equivalents for lists and blockquotes.

In prose specifications, put literal Python identifiers, signatures, comparisons,
shapes and paths in backticks. The formatter protects these before terminology
normalization, including `<` comparisons and escaped entities. Do not rely on
bare function text being recognized as code. Existing `<code>`, `<pre>`, formula,
script and style subtrees are protected. Short Persian labels may stay with one
following word; long sentences and inline code can still wrap naturally.

The author checks cover all 1107 Markdown cells, repeat-render stability, API
case preservation, short math groups, mixed tables and protected content. Browser
QA should also inspect actual Jupyter output: successful source CSS assertions
do not prove that its sanitizer retained a property.

```powershell
python -B -m tools.build_notebooks --check
python -B -m tools.verify_notebooks --mode student
python -B -m tools.verify_notebooks --mode solutions
python -B -m unittest discover -s tests -v
python -B -m tools.prepare_release
```

Student verification executes authored notebooks unchanged in fresh kernels and
requires explicit incomplete exercise status. Solution verification replaces only
tagged TODO cells **in memory**, executes each notebook in another fresh kernel,
and requires the exercise/repair checks to pass. Neither mode saves output into
authored notebooks. The report records execution mode, path, cell counts, and
timings. An uncaught red error is not an expected student experience.

See [Windows setup](WINDOWS_SETUP.md) for Python selection and activation
troubleshooting. Implementation references:
[Jupyter authentication](https://jupyter-server.readthedocs.io/en/latest/operators/security.html),
[kernel specifications](https://jupyter-client.readthedocs.io/en/stable/kernels.html),
and [nbclient execution](https://nbclient.readthedocs.io/en/latest/client.html).
