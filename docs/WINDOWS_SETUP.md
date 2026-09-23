# Windows setup — Persian Mini-GPT book

For Windows 10/11, 64-bit x86. Use the unified learning environment below for
both the book and Jupyter. Reading the static book alone remains possible.
Commands below are run from the extracted project folder, not from inside a ZIP.

## Recommended first run: book + every notebook

Install Python 3.11+ from the official source described below, extract the whole
project, and open PowerShell in the folder containing `run.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe run.py
```

Reuse `.venv` if it already exists. This is one dependency installation for
Mini-GPT, Jupyter, and plots; the book builder itself uses the standard library.
Activation is optional. With the environment activated, daily use is simply
`python run.py`.

The learning desk opens at http://127.0.0.1:8000/start.html. It links to the book,
lesson list, laboratories, and setup help. Jupyter runs at
http://127.0.0.1:8888/lab; use the private tokenized URL printed in the terminal
or click a lesson's lab button. The launcher chooses its own exact Python as
the notebook kernel. You do not register kernels or start another process.

Keep the terminal open. Save your notebook, then press Ctrl+C to stop both
services. The launcher stops only processes it created. If port 8000 or 8888 is
occupied, stop your older instance and retry; it does not silently switch ports.
Do not expose either service to a network or share the Jupyter token.

A fresh lab reports INCOMPLETE for its two TODO functions. Implement them,
rerun their cells and checks, and obtain PASS. This is different from a missing
package or uncaught exception. See [notebook workflow](NOTEBOOKS.md).

The remaining sections also document optional reading-only and model-only use.

## 1. Choose what you need

| Use | Required software | Folder |
| --- | --- | --- |
| Read the static book | Current Edge, Chrome or Firefox | Extracted `book-site.zip`, or full checkout's `dist` |
| Serve the book locally (recommended for shared progress) | Python 3.11+; no packages | Same static files |
| Run the counting model v0 | Python 3.11+; no packages | Full checkout or extracted `mini-gpt-project.zip` |
| Train/run neural Mini-GPT | Python 3.11, venv, CPU PyTorch 2.14.0 | Full checkout or model ZIP |
| Edit/build/release the book | Python 3.11+ and Node.js 24 LTS | Full source checkout only |

No backend, database, npm install, account, GPU, CUDA toolkit, or external font
service is required. Node.js is only for author JavaScript checks, not reading or
training. Installing PyTorch needs internet and additional disk space; afterward
the bundled examples run locally. The browser does not run PyTorch or load `.pt`
files: the model inspector imports JSON exported by the Python command.

## 2. Install prerequisites only if needed

If only reading, skip this section. For Python, use the official
[Windows instructions](https://docs.python.org/3/using/windows.html) and
[Python downloads](https://www.python.org/downloads/windows/). The current Python
Install Manager supports Windows 10/11; choose its Python 3.11 runtime for the
tested model path. After installing the manager, `py install 3.11` installs that
runtime. Close and reopen the terminal. `py -3.11 --version` selects it explicitly.
Manager installation itself was source-checked, not performed during this audit.

In the remaining commands, `python` must be the chosen interpreter. Check:

```powershell
python --version
python -c "import sys, struct; print(sys.executable); print(struct.calcsize('P')*8)"
```

Expect Python 3.11.x and `64` for the tested model setup. If `python` is absent,
opens the Store, or selects another version, use `py -3.11` instead of `python`
when creating the virtual environment. An older standalone Python install may
work through `python` even if `py` reports no registered runtimes. Do not install
packages into an unrelated interpreter. Once the venv exists, use its explicit
path below. Do not change global execution policy or run the terminal as admin.

For authors only, install **Node.js 24 LTS** using the official Windows installer
from [Node.js downloads](https://nodejs.org/en/download). Reopen the terminal and
check `node --version`. No `package.json` or npm dependency installation is used.

## 3. Extract and open the right folder

Use Explorer → Extract All. Spaces and Persian characters in the folder name are
supported. Examples below use `C:\Books\MiniGPT`; replace it with your location.

PowerShell:

```powershell
Set-Location -LiteralPath 'C:\Books\MiniGPT'
```

Command Prompt (CMD; `/d` also changes drive):

```bat
cd /d "C:\Books\MiniGPT"
```

The full checkout contains `tools/build_book.py`, `book_src`, `mini_gpt`, `data`, and
`dist`. The learning-project ZIP also contains `run.py`, `book_src`, `tools`,
`notebooks`, `mini_gpt`, model tests, data, and these guides. It builds the HTML
on first launch; a separate book download is no longer required. Download it
from the book's project page. The static site ZIP has `index.html` at its root;
do not expect a nested `dist` directory in that ZIP.

## 4. Read or serve the book

Double-click the extracted site's `index.html` (full checkout: `dist/index.html`).
All pages, CSS, JavaScript, diagrams and Vazirmatn fonts are bundled. Reading and
inline calculators need no server. `file://` storage and downloads vary between
browsers; export progress and the experiment journal as JSON. A stable localhost
origin is recommended for progress shared across pages.

From the **full checkout**, PowerShell or CMD:

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory dist
```

From the **extracted static site ZIP** folder:

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory .
```

Open [the local book](http://127.0.0.1:8000/). Keep the terminal open. Stop with
Ctrl+C. Binding to `127.0.0.1` exposes it only to your own computer; this server is
not a public production host. Use the same URL and port on later visits so the
browser uses the same storage origin. Export before changing browser or origin.

## 5. Install and run Mini-GPT

Run from the folder containing `mini_gpt` and `data`. The following explicit-path
commands work in both PowerShell and CMD and need **no activation**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__)"
.\.venv\Scripts\python.exe -B -m mini_gpt.smoke_test
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Upgrading pip is important: the pip bundled with this machine's Python 3.11.0
misread dependency metadata during the first install. The upgraded pip succeeded.
The pinned CPU command is the tested path and satisfies `requirements.txt`
(`torch>=2.6,<3`). Installing the broad requirements range is not a reproducibility
pin. No NumPy API is used; PyTorch may warn that optional NumPy is absent. This
warning did not prevent any model/example test from passing.

To use the shorter `python` commands shown in lessons, activate first:

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

CMD:

```bat
.venv\Scripts\activate.bat
python -c "import sys; print(sys.executable)"
```

If PowerShell blocks activation scripts, keep using the explicit venv Python path
instead. Activation is a convenience, not a requirement. `deactivate` leaves an
activated environment. Never infer activation merely from the terminal title.

After activation, this small end-to-end run works in both shells. Use a new output
folder; an existing nonempty run is deliberately not overwritten.

```powershell
python -m mini_gpt.milestones --stage 0
python -m mini_gpt.train --text data/sample.txt --output runs/windows-check --steps 6 --eval-every 2 --context-length 8 --embedding-dim 16 --num-heads 2 --num-layers 1 --batch-size 2 --device cpu --threads 1
python -m mini_gpt.evaluate --checkpoint runs/windows-check/last.pt --text data/sample.txt --device cpu --threads 1
python -m mini_gpt.generate --checkpoint runs/windows-check/last.pt --prompt "مدل " --tokens 12 --greedy --device cpu --threads 1
python -m mini_gpt.inspect --checkpoint runs/windows-check/last.pt --prompt "مدل " --output runs/windows-check/inspection.json --max-tokens 8 --generate-tokens 2 --metrics runs/windows-check/metrics.csv --device cpu
python -m mini_gpt.train --output runs/windows-check --resume runs/windows-check/last.pt --steps 8 --eval-every 2 --device cpu --threads 1
```

`--steps 8` on resume means a total of eight, not eight extra. Resume restores
saved architecture, Batch, Optimizer and random state; do not repeat architecture
flags. Preserve the entire run folder when moving it. Load only your own trusted
checkpoints. Six steps check the workflow; they do not produce a capable chatbot.

Independent Test evaluation is optional and needs a **real separately held-out**
UTF-8 text file longer than the model's context length:

```powershell
python -m mini_gpt.evaluate --checkpoint runs/windows-check/last.pt --test-text data/test.txt --device cpu
```

`data/test.txt` is not supplied or manufactured from the Training sample. This
mode uses the saved Vocabulary and reports unknown rate. It cannot certify that
your documents are independent. Default `--text` instead checks the fingerprint
of the original corpus and evaluates only its Validation tail.

## 6. Author build and static deployment

Full checkout only; no PyTorch is needed for these commands:

```powershell
python -B -m tools.build_book
python -B -m tools.validate_book
python -B -m tools.prepare_release
```

The release command requires Node and runs HTML, Python, JavaScript, navigation
and release-boundary checks. It writes `release/book-site.zip`, `manifest.json`
and `SHA256SUMS.txt`. Publish only the archive's contents or `dist`, not the
repository, tests, model runs, venv or archive history. The public download ZIP
intentionally includes model source/tests for learners; it is not a server runtime.
Remote Sites registration was unavailable during this audit; no remote deployment
was performed. Existing hosting metadata is preserved.

## 7. Troubleshooting

- `ModuleNotFoundError: torch`: use `.\.venv\Scripts\python.exe`; compare the
  printed interpreter path with the installation command.
- `No module named mini_gpt` or missing `data/sample.txt`: move to the extracted
  model/project root. Do not run from inside the `mini_gpt` directory.
- Port 8000 occupied: stop your old server, or use port 8001 and open the matching
  URL. A new port has separate browser storage; import your exported JSON.
- Persian output: project CLIs configure UTF-8, including redirected output.
  For your own lesson scripts, use `python -X utf8 lesson.py`. Save `.py` and text
  data as UTF-8. In Windows PowerShell 5, `>` may transcode output; prefer the
  command's `--output` file option for JSON instead of shell redirection.
- Nonempty output folder: select a new `--output`, or resume its own `last.pt`.
  Do not delete previous experiments merely to make a command succeed.
- Model too large/slow: use the tiny CPU settings above first. CUDA and Windows
  ARM were not tested. Do not install a GPU wheel/toolkit for this CPU workflow.
- `file://` progress missing: serve on localhost and export/import records.
  Private-browsing mode or blocked storage may prevent persistence.
- Missing fonts/styles: extract the whole ZIP; do not copy `index.html` alone.

## Verification scope

The audit used an isolated venv on this existing Windows host, Python 3.11.0
64-bit and official CPU PyTorch 2.14.0+cpu; no global package installation.
The install, model tests, lesson examples, training/evaluation/generation/inspection
and resume are exercised as recorded in `REVIEW.md` in the full checkout.
The extracted distribution is also checked under a path with spaces and Persian
characters. This is **not** a clean Windows VM or a Windows 10/11 installer test.
Browser checks ran over localhost. Direct file:// browser testing was blocked by
the browser tool's URL policy; it was not bypassed. File-mode support is based on
the static relative-path/classic-script design, not a completed browser test.
No claim is made for every browser or hardware combination.


## Lesson laboratories

The 92 dedicated lesson notebooks and 12 review labs share this one environment.
See [Jupyter laboratories](NOTEBOOKS.md) for mapping, exercises and verification.
