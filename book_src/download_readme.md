# Persian AI Book — complete local learning project

This download contains the book source, 92 dedicated lesson notebooks, 12 optional
review notebooks, the actual Mini-GPT and offline system modules, tiny sample data,
model/system tests, and setup guides.
Extract the whole ZIP before running anything.

## Windows: install once

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe run.py
```

Daily use, with the environment activated: `python run.py`.

Book: http://127.0.0.1:8000/
Learning desk: http://127.0.0.1:8000/start.html
Jupyter: http://127.0.0.1:8888/lab (use the private token URL or a book lab button).

The launcher builds HTML locally, starts both servers on loopback, and stops its
children on Ctrl+C. Save notebook work first. It never fills in your TODOs or
regenerates your notebooks. Read docs/WINDOWS_SETUP.md and docs/NOTEBOOKS.md.

No GPU, account, external dataset, or downloaded checkpoint is needed. Internet
is required for initial package installation. Do not expose Jupyter publicly.
The model is educational, not a capable general-purpose assistant.

The author-only release/test infrastructure is not part of this learner ZIP.
Model tests and the local book builder are included.
