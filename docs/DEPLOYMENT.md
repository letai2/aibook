# Deployment guide

First-time Windows instructions: [WINDOWS_SETUP.md](WINDOWS_SETUP.md).
Current educational and verification status: [REVIEW.md](REVIEW.md).

This is a static Persian textbook with a downloadable Python project. Hosting
does **not** require Python, PyTorch, Node.js, a database, API keys, or a backend.
The browser never executes model training or uploads journal/inspection data.

## Prepare a release

From the project directory, with Python 3.11+ and Node.js installed:

```sh
python -B -m tools.prepare_release
```

This builds into a new temporary directory, validates every page/link, checks
JavaScript syntax, runs 29 JavaScript and 36 release/design/editorial tests, refreshes `dist`, and creates:

- `release/book-site.zip`: only the public site, with `index.html` at its root.
- `release/manifest.json`: exact public file list, sizes and SHA-256 hashes.
- `release/SHA256SUMS.txt`: the deployment ZIP checksum.

The archive is deterministic for unchanged source in the same build environment
(including Python/zlib and platform newline conventions). Existing unexpected files in
`dist` stop preparation; inspect and move them out deliberately before retrying.
No files are silently deleted, no Git commit is made, and nothing is published.
The release and temporary directories are ignored by Git. Do not run concurrent
builds against the same checkout.

For a local preview:

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:8000/`. Stop with Ctrl+C. This development server is not a
production server.

## Existing deployment target

`.openai/hosting.json` already identifies the existing Sites project and selects
`dist` as the static directory. Preserve that project and its audience. Preparing
a release does not grant permission to publish or change access.

When publication is requested, use the Sites hosting workflow for the existing
project. Its packaging tool creates the platform-specific deployment archive;
`release/book-site.zip` is a portable static-site bundle, **not** a replacement
for that platform-specific archive or source revision workflow.

The 2026-09-21 checks could not resolve the configured Sites
project (`NOT_FOUND`). No replacement project was created and no audience was
changed. Resolve access to the existing project before publishing there.

## Hosting contract

- Serve **only `dist`**, never the repository root. Keep checkpoints, `runs`,
  edition archives, QA backups, environments, source-control and release metadata
  private. The source-code download already intentionally lives in `dist/downloads`.
- Use HTTPS and a stable origin. Journal and progress live in the browser, not on
  the server. Moving domains, changing HTTP to HTTPS, or clearing browser data
  requires export/import to retain learner records. Back up before migration.
- Preserve nested paths and `.html` filenames, including direct lesson links.
  This is not a single-page app; do not rewrite missing routes to `index.html`.
- Use UTF-8 HTML and normal JavaScript/CSS/ZIP/WOFF2 content types. Serve notebook
  downloads as `application/x-ipynb+json` or an attachment; static hosting does not
  execute their Python cells.
- JavaScript and CSS URLs carry a deterministic source-version query so new
  pages do not load an older progress manifest. Revalidate HTML and unversioned
  resources on updates (`Cache-Control: no-cache` or equivalent); do not mark
  HTML immutable. Preserve query strings in cache keys.
- Where the host supports them, use `X-Content-Type-Options: nosniff` and
  `Referrer-Policy: strict-origin-when-cross-origin`. Host headers are not configured
  by this repository's static Sites manifest; verify them after publishing.
- The book bundles Vazirmatn and its SIL Open Font License locally. There are no
  external font requests, analytics or CDN assets. No secrets are needed.
  Do not invent a license or make the existing site public without authorization.

## Verification gates

The deployment preparation command covers the static site, not the PyTorch model.
Before a release with model or lesson-code changes, use the project's virtual
  environment (reuse it if it already exists):

```sh
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.venv\Scripts\python.exe -B -m mini_gpt.smoke_test
python -B -m unittest discover -s tests/site -v
python -B -m tools.prepare_release
```

On macOS/Linux use `.venv/bin/python`. The temporary dependency installations used
during authoring are not part of the product and are not needed for deployment.
The latest model verification used Python 3.11 / PyTorch 2.14.0+cpu; GPU behavior
has not been verified. Install CUDA-specific builds only for a matching device.

After publishing, verify the homepage, a direct nested lesson URL, lab, journal,
downloadable project, browser console and mobile layout on the actual HTTPS URL.
Also check a glossary round-trip, glossary search, a lesson/checkpoint transition,
and the locally served Persian font. Editorial review details are in `EDITORIAL_REVIEW.md`.
Verify that a genuinely missing path returns 404. Export/import a disposable
learner record before changing the origin of an existing site.

## Recovery

The previous textbook editions and training evidence remain in `archive` and
`runs`. Cleanup preserves QA browser fixture JSON in an archive. Installed test
libraries are reproducible dependencies, not backups. Browser learner records
are separate and are never deleted by the release command.

Keep the previous deployed version available through the hosting service for
rollback. Restore a saved source revision and rerun release preparation when
rebuilding an older release; do not mix files from different releases.


For daily local learning, install `requirements-notebooks.txt` in that same environment
and run `python run.py`; this starts the book and authenticated Jupyter together.
Use `python -B -m tools.verify_notebooks --mode student` and `--mode solutions`
for author verification. Jupyter is local-only and is never included as a public server. The release inventory rejects
unlisted notebooks and notebooks containing executed outputs. Detailed evidence
and platform limits are in [the current review](REVIEW.md).
