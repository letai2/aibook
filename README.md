# From Python to Mini-GPT

A Persian, beginner-oriented HTML textbook with a transparent PyTorch Mini-GPT
and 12 focused Jupyter laboratories. The HTML book remains the main learning path.

- [Open the built book](dist/index.html)
- [Windows installation](docs/WINDOWS_SETUP.md)
- [Jupyter setup and lesson-to-notebook map](docs/NOTEBOOKS.md)
- [Current review and verification](docs/REVIEW.md)
- [Curriculum](docs/CURRICULUM.md) and [review coverage](docs/COVERAGE.md)
- [Model commands](mini_gpt/README.md)
- [Deployment](docs/DEPLOYMENT.md)

## Work from the project root

```powershell
python -B -m tools.build_book
python -B -m tools.validate_book
python -B -m http.server 8000 --bind 127.0.0.1 --directory dist
```

The static book needs no Python server after deployment. The local server is
recommended for browser testing and notebook return links. To build a verified
release, install the model dependencies, ensure Node.js is available, then run:

```powershell
python -B -m unittest discover -s tests -v
python -B -m tools.prepare_release
```

The release command builds in staging, checks site/editorial rules and browser
logic, validates links, and packages only public files. To execute the Jupyter
labs as well, install `requirements-notebooks.txt` in the same environment and
run `python -B -m tools.verify_notebooks`.

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
