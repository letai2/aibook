# Jupyter laboratories

The HTML book is the primary course. These 12 focused notebooks are optional
experimental stops, not replacements for lessons. Read the linked prerequisites,
predict the result, run, inspect, change one factor, explain, and return to the book.

## Windows: one environment for the model and notebooks

Use 64-bit Python 3.11 for the tested CPU path. Open PowerShell in the **project
root** (the directory containing `mini_gpt`, `data`, and `requirements.txt`).
If `.venv` already exists for this project, reuse it and skip its creation.
Do not recreate an existing environment just to add Jupyter.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-notebooks.txt
python -m jupyterlab --notebook-dir=. --ip=127.0.0.1
```

The Torch command is the tested Windows CPU installation. If the project
environment already has a supported Torch installation, keep it; the notebook
requirements reuse `requirements.txt` rather than replacing it. See
[Windows setup](WINDOWS_SETUP.md) for installation, Python selection, CMD, and
platform-specific details.

If PowerShell blocks activation, **do not change machine execution policy**.
Use the interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe -m jupyterlab --notebook-dir=. --ip=127.0.0.1
```

Open the authenticated local URL Jupyter prints. Keep its token private. Do not
disable authentication or bind to a public interface. In the file browser, open
`notebooks/` and choose a notebook. Select the Python kernel belonging to this
environment; its first cell prints `sys.executable` so you can verify that choice.
If the matching kernel is missing, stop the incorrectly launched server and launch
Jupyter with the explicit environment interpreter. If needed, register a distinctly
named kernel **inside this project environment**, then select it:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name mini-gpt-book --display-name "Mini-GPT (.venv)"
```

Stop the server with Ctrl+C when finished, then confirm shutdown if prompted.

The only direct additions are **JupyterLab** (with its normal kernel/client
dependencies) and **Matplotlib**, which renders actual matrices, gradients,
Attention heatmaps, and measured training curves. No GPU, external dataset,
downloaded checkpoint, network call, or notebook-specific model implementation
is required after installation.

## Book links and downloaded projects

Each notebook links back to the HTML book at `http://127.0.0.1:8000/`.
In another PowerShell window, from the full checkout run:

```powershell
python -B -m http.server 8000 --bind 127.0.0.1 --directory dist
```

If you extracted `book-site.zip`, run the same server **from its extracted
directory**, without `--directory dist`. The independent
`mini-gpt-project.zip` contains the model, sample data, notebooks and setup
guides, but not the HTML book. Obtain the book separately or use the full
checkout; that distinction is intentional. If port 8000 is already serving
this book, reuse it.

Always extract the **whole** Mini-GPT ZIP. Preserve `mini_gpt/`, `data/`,
`notebooks/`, and `docs/` as siblings. Notebook root discovery works from
their nested directories, including paths containing spaces and Persian text.
A notebook downloaded individually must replace the corresponding file in that
structure; it is not a standalone copy of all dependencies.

## Learning map

All listed lessons are links from HTML to the appropriate notebook guide.
Finish the listed prerequisites before running the notebook. The final listed lesson is the ready-to-run boundary; earlier callouts explicitly defer execution.

| Lab | Notebook | Connected lesson IDs |
| --- | --- | --- |
| 01 | [از سطر و ستون تا ضرب ماتریسی](../notebooks/mathematics/01_matrix_products.ipynb) | `06-dot`, `07-matmul` |
| 02 | [از امتیاز تا احتمال و Loss](../notebooks/mathematics/02_probability_loss.ipynb) | `08-probability`, `09-softmax`, `10-entropy` |
| 03 | [شکل درست، محور درست؟](../notebooks/pytorch/03_tensor_shapes.ipynb) | `13-torch`, `14-index-device`, `15-broadcast`, `16-reshape` |
| 04 | [Gradient جهت را نشان می‌دهد؛ اندازهٔ گام چه می‌کند؟](../notebooks/pytorch/04_gradients_steps.ipynb) | `11-derivative`, `12-chain`, `12-sgd`, `17-autograd` |
| 05 | [یک شبکهٔ کوچک واقعاً چه چیزی یاد می‌گیرد؟](../notebooks/pytorch/05_first_network.ipynb) | `18-module`, `19-network` |
| 06 | [از نویسه تا نمایش یادگرفتنی](../notebooks/nlp/06_tokens_embeddings.ipynb) | `21-tokenizer`, `23-shift`, `25-embedding`, `26-positions` |
| 07 | [Attention را خانه‌به‌خانه باز کنیم](../notebooks/attention/07_attention_math.ipynb) | `28-qkv`, `29-scores`, `30-scaling`, `31-values`, `33-mask` |
| 08 | [اگر آینده را باز بگذاریم چه می‌شود؟](../notebooks/attention/08_causal_mask.ipynb) | `33-mask`, `34-causal-test` |
| 09 | [چند Head، یک خروجی](../notebooks/transformer/09_multi_head.ipynb) | `35-split-heads`, `36-merge-heads` |
| 10 | [دو جمع در یک بلوک واقعی](../notebooks/transformer/10_block_trace.ipynb) | `37-ffn`, `38-residual`, `39-layernorm`, `40-block`, `41-stack`, `45-trace` |
| 11 | [یک آموزش کوچک، با شاهد قابل دیدن](../notebooks/mini_gpt/11_train_inspect.ipynb) | `43-lm-head`, `45-trace`, `46-gradient-path`, `47-loop`, `48-evaluate`, `52-first-run`, `53-curves` |
| 12 | [وزن ثابت، انتخاب متفاوت](../notebooks/mini_gpt/12_sampling.ipynb) | `54-generate`, `55-temperature`, `56-topkp`, `57-prompts` |

## Execution and experiments

Use **Kernel → Restart Kernel and Run All Cells** for a clean run. Every
notebook is independent, uses fixed CPU seeds, and defines its own state from
top to bottom. Labs 11 and 12 each train their own small model; neither reads a
checkpoint from another notebook. Training takes longer than the small arithmetic
labs. Exact floating-point values and timings can vary across versions/devices.

Expected failures are deliberately caught and explained: mismatched matrix
dimensions, silent broadcasting, wrong target dtype, missing Autograd graph,
out-of-range IDs, incompatible C/H, excessive context and invalid temperature.
An uncaught red error is **not** expected. Read the exception and check the
environment and folder structure before proceeding.

The Attention sequence exposes embeddings → Q/K/V → raw and scaled scores →
mask → Softmax → weights → weighted Values. Later notebooks verify the real
project's multi-head merge/projection, Pre-Norm residual block, complete logits,
Loss, gradients, parameter updates, train/validation curves and generated text.
Their explicit miniature arithmetic is labeled separately from the actual
`mini_gpt` imports.

Plots use English axis labels to keep tensor coordinates unambiguous; explanations
and exercises are Persian. Attention heatmaps use Query rows and Key columns; other matrix plots label their own axes.
The small corpus's held-out tail is not an independent real-world benchmark.
Neither falling Loss nor attractive samples establish an intelligent or reliable
assistant.

## Keep source notebooks clean

Authored notebooks have empty outputs and null execution counts. Running them
does not write model checkpoints, datasets or images into the project. Jupyter
may create `.ipynb_checkpoints`; Git ignores these. Save useful observations in
your own working copy or learning log. Before contributing source notebooks,
restart the kernel, clear all outputs and save. Do not delete useful personal
experiments merely to make Git quiet.

From the full checkout, maintainers can verify all notebooks in fresh kernels:

```powershell
.\.venv\Scripts\python.exe -B -m tools.verify_notebooks
```

This executes each notebook in its own nested directory with the exact invoked
Python interpreter; it never saves executed outputs into authored files.
Maintainer tooling is not included in the independent model ZIP.

Implementation references: [JupyterLab installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html),
[nbclient execution](https://nbclient.readthedocs.io/en/latest/client.html), and
[Matplotlib installation](https://matplotlib.org/stable/install/index.html).
