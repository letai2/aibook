"""Build, validate and package the static book without publishing it.

Requires Python 3.11+ and Node.js; no Python third-party packages are needed.
Unexpected existing dist files are reported, never silently deleted.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from book_src.laboratories import catalog as notebook_catalog


ROOT = Path(__file__).resolve().parents[1]
STAMP = (2026, 9, 19, 0, 0, 0)
ALLOWED_SUFFIXES = {'.html', '.css', '.js', '.zip', '.woff2', '.txt', '.ipynb'}
RELEASE_FILES = ('book-site.zip', 'manifest.json', 'SHA256SUMS.txt')
NOTEBOOK_FILES = {lab['path'] for lab in notebook_catalog(ROOT)}


def inventory(directory):
    """Return exact public file hashes, rejecting links and private artifacts."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir() or (
            getattr(directory.lstat(), 'st_file_attributes', 0) & 1024):
        raise ValueError(f'Expected a real directory: {directory}')
    result = {}
    for path in sorted(directory.rglob('*')):
        # is_junction is not available on Python 3.11; reparse points cover it.
        if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 1024:
            raise ValueError(f'Links/reparse points are not release files: {path}')
        relative = path.relative_to(directory)
        if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
            raise ValueError(f'Private/cache path in public output: {relative}')
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f'Unexpected public file: {relative}')
        if path.suffix == '.zip' and relative.as_posix() != 'downloads/mini-gpt-project.zip':
            raise ValueError(f'Unexpected downloadable archive: {relative}')
        if path.suffix == '.ipynb':
            if relative.as_posix() not in NOTEBOOK_FILES:
                raise ValueError(f'Unexpected public notebook: {relative}')
            notebook = json.loads(path.read_text(encoding='utf-8'))
            if any(cell.get('outputs') or cell.get('execution_count') is not None
                   for cell in notebook['cells'] if cell['cell_type'] == 'code'):
                raise ValueError(f'Executed notebook must not be published: {relative}')
        if path.suffix in ('.woff2', '.txt') and relative.as_posix() not in {
                'assets/fonts/Vazirmatn.woff2', 'assets/fonts/OFL-Vazirmatn.txt'}:
            raise ValueError(f'Unexpected font/license file: {relative}')
        data = path.read_bytes()
        result[relative.as_posix()] = {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
        }
    if 'index.html' not in result:
        raise ValueError('Public output has no index.html')
    return result


def check_existing_output(current, expected):
    if not current.exists():
        return
    existing = inventory(current)
    unexpected = set(existing) - set(expected)
    if unexpected:
        raise ValueError('Unexpected files in dist; review/move them before release: '
                         + ', '.join(sorted(unexpected)))


def write_archive(public, target, files):
    """Store exactly the verified public files, with reproducible ZIP metadata."""
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(files):
            data = (public / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != files[relative]['sha256']:
                raise ValueError(f'File changed during release: {relative}')
            info = zipfile.ZipInfo(relative, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)


def real_directory(path):
    if path.is_symlink() or (path.exists() and (
            not path.is_dir() or getattr(path.lstat(), 'st_file_attributes', 0) & 1024)):
        raise ValueError(f'Refusing a linked or non-directory output: {path}')
    path.mkdir(exist_ok=True)


def check_release_destinations(directory):
    for name in RELEASE_FILES:
        path = directory / name
        if path.is_symlink() or (path.exists() and (
                not path.is_file() or getattr(path.lstat(), 'st_file_attributes', 0) & 1024)):
            raise ValueError(f'Refusing a linked or non-file release target: {path}')


def run(*command):
    print('Running:', ' '.join(str(part) for part in command), flush=True)
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    # The HTML validator uses assertions; never silently disable them.
    env.pop('PYTHONOPTIMIZE', None)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11 or newer is required.')
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js is required for JavaScript release checks.')
    config = json.loads((ROOT / '.openai/hosting.json').read_text(encoding='utf-8'))
    if config.get('static', {}).get('directory') != 'dist' or not config.get('project_id'):
        raise ValueError('Expected the existing Sites project with static.directory=dist.')

    releases = ROOT / 'release'
    temporary = ROOT / '.release-tmp'
    real_directory(releases)
    check_release_destinations(releases)
    real_directory(temporary)
    try:
        with tempfile.TemporaryDirectory(prefix='build-', dir=temporary) as work:
            work = Path(work)
            public = work / 'public'
            run(sys.executable, '-B', '-m', 'tools.build_notebooks', '--check')
            run(sys.executable, '-B', '-m', 'tools.journey_report', '--check')
            run(sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests/site', '-v')
            run(sys.executable, '-B', '-m', 'tools.build_book', '--output', str(public))
            run(sys.executable, '-B', '-m', 'tools.validate_book', '--root', str(public))
            for name in ('book.js', 'experience.js', 'journal.js', 'manifest.js', 'inspection-demo.js'):
                run(node, '--check', str(public / 'assets' / name))
            for script in ('test_browser_math.js', 'test_journal.js'):
                run(node, str(ROOT / 'tests' / 'browser' / script))

            files = inventory(public)
            current = ROOT / 'dist'
            check_existing_output(current, files)
            archive = work / 'book-site.zip'
            write_archive(public, archive, files)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest = {
                'format': 'book-static-release-v1', 'public_directory': 'dist',
                'archive': 'book-site.zip', 'archive_sha256': digest,
                'file_count': len(files), 'files': files,
            }
            (work / 'manifest.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            (work / 'SHA256SUMS.txt').write_text(
                f'{digest}  book-site.zip\n', encoding='ascii')

            # There are no extra files to delete. All published files come from
            # the freshly built, validated staging directory, not the old dist.
            shutil.copytree(public, current, dirs_exist_ok=True)
            if inventory(current) != files:
                raise ValueError('dist changed during release preparation; rerun after review.')
            check_release_destinations(releases)
            for name in RELEASE_FILES:
                destination = releases / name
                os.replace(work / name, destination)
            print(f'Ready: {len(files)} public files; {digest}')
            print('Deployment archive: release/book-site.zip')
            print('Nothing was published. Model training tests are a separate release gate.')
    finally:
        # Only remove this known empty helper folder, never a recursive root.
        try:
            temporary.rmdir()
        except OSError:
            pass


if __name__ == '__main__':
    main()
