"""Execute every lab in a fresh kernel using this exact Python interpreter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def verify(root: Path, report: Path | None = None, html_output: Path | None = None):
    notebooks = sorted((root / 'notebooks').rglob('*.ipynb'))
    if not notebooks:
        raise ValueError('No notebooks found')
    results = []
    with tempfile.TemporaryDirectory(prefix='book-kernel-') as temporary:
        kernel_root = Path(temporary)
        specification = kernel_root / 'book-python'
        specification.mkdir()
        (specification / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Book verification', 'language': 'python',
            'env': {'MPLBACKEND': 'module://matplotlib_inline.backend_inline',
                    # Do not read or write the learner's personal IPython profile.
                    'IPYTHONDIR': str(kernel_root / 'ipython'),
                    'PYTHONDONTWRITEBYTECODE': '1'},
        }), encoding='utf-8')
        for path in notebooks:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == 'code' and (cell.outputs or cell.execution_count is not None):
                    raise ValueError(f'Clear authored outputs before verification: {path}')
            started = time.monotonic()
            manager = KernelManager(kernel_name='book-python',
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
            client = NotebookClient(notebook, km=manager, timeout=180,
                allow_errors=False, resources={'metadata': {'path': str(path.parent)}})
            # A new manager and kernel for every notebook; execute in its nested directory.
            try:
                client.execute()
            finally:
                # nbclient does not own an explicitly supplied KernelManager.
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
            outputs = [output for cell in notebook.cells if cell.cell_type == 'code'
                       for output in cell.outputs]
            if any(output.output_type == 'error' for output in outputs):
                raise AssertionError(f'Unexpected error output in {path}')
            streams = ''.join(output.get('text','') for output in outputs
                              if output.output_type == 'stream')
            if str(root) not in streams or sys.executable not in streams:
                raise AssertionError(f'Wrong project root or kernel interpreter: {path}')
            result = {'path': path.relative_to(root).as_posix(),
                      'code_cells': sum(cell.cell_type == 'code' for cell in notebook.cells),
                      'figures': sum('image/png' in output.get('data', {}) for output in outputs),
                      'seconds': round(time.monotonic()-started, 2)}
            results.append(result)
            if html_output:
                from nbconvert import HTMLExporter
                rendered, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
                html_output.mkdir(parents=True, exist_ok=True)
                (html_output/(path.stem+'.html')).write_text(rendered, encoding='utf-8')
            print(json.dumps(result), flush=True)
    payload = {'python': sys.version.split()[0], 'interpreter': sys.executable,
               'root': str(root), 'notebooks': results}
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path)
    parser.add_argument('--html-output', type=Path, help='Optional disposable rendered QA directory.')
    args = parser.parse_args()
    verify(args.root.resolve(), args.report, args.html_output)
