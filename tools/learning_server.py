"""A local-only, single-interpreter learning environment with owned lifetimes."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import URLError
from urllib.parse import parse_qs, quote, urlsplit
from urllib.request import Request, urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
BOOK_URL = 'http://127.0.0.1:8000'
JUPYTER_URL = 'http://127.0.0.1:8888'

def environment_errors():
    errors = []
    if sys.version_info < (3, 11):
        errors.append('Python 3.11 or newer is required.')
    for package in ('torch', 'jupyterlab', 'ipykernel', 'matplotlib', 'nbclient'):
        if importlib.util.find_spec(package) is None:
            errors.append(f'Missing package: {package}')
    for relative in ('mini_gpt', 'book_src', 'notebooks', 'data/sample.txt'):
        if not (ROOT / relative).exists():
            errors.append(f'Missing project content: {relative}; extract the complete project.')
    return errors

def needs_build(root=ROOT):
    marker = root / 'dist/start.html'
    if any(not (root/'dist'/name).is_file() for name in
           ('start.html', 'index.html', 'notebooks.html', 'assets/book.css', 'assets/manifest.js')):
        return True
    stamp = marker.stat().st_mtime_ns
    sources = [root/'run.py', root/'requirements-notebooks.txt', root/'requirements.txt']
    for directory in ('book_src', 'mini_gpt', 'tools', 'notebooks', 'data', 'docs'):
        sources.extend(p for p in (root/directory).rglob('*') if p.is_file()
                       and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts)
    return any(p.stat().st_mtime_ns > stamp for p in sources)

def port_available(port):
    with socket.socket() as probe:
        if os.name == 'nt':
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(('127.0.0.1', port))
        except OSError:
            return False
    return True

def notebook_target(query, labs):
    """Only mapped paths are accepted; never redirect to a user-provided URL."""
    args = parse_qs(query, keep_blank_values=True)
    if not args:
        return ''
    if set(args) == {'lesson'} and len(args['lesson']) == 1:
        matches = [lab for lab in labs if lab.get('primary_lesson') == args['lesson'][0]]
    elif set(args) == {'review'} and len(args['review']) == 1:
        matches = [lab for lab in labs if lab['kind'] == 'review' and lab['id'] == args['review'][0]]
    else:
        raise ValueError('Unknown laboratory request')
    if len(matches) != 1:
        raise ValueError('Unknown laboratory')
    return matches[0]['path']

class LearningHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, token, labs, **kwargs):
        self.token, self.labs = token, labs
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        # Avoid retaining private tokens or learner URL contents in request logs.
        pass

    def end_headers(self):
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def list_directory(self, path):
        self.send_error(403, 'Directory listing is disabled')
        return None

    def valid_local_request(self):
        if self.headers.get('Host') not in {'127.0.0.1:8000', 'localhost:8000'}:
            self.send_error(403, 'Local host required')
            return False
        translated = Path(self.translate_path(urlsplit(self.path).path)).resolve()
        if not translated.is_relative_to(Path(self.directory).resolve()):
            self.send_error(403, 'Path outside public book')
            return False
        return True

    def do_HEAD(self):
        if self.valid_local_request():
            super().do_HEAD()

    def do_GET(self):
        if not self.valid_local_request():
            return
        request = urlsplit(self.path)
        if request.path == '/launch.html':
            # A foreign webpage must not use our local launcher as a token oracle.
            origin = self.headers.get('Origin')
            referrer = self.headers.get('Referer')
            allowed = {BOOK_URL, 'http://localhost:8000'}
            referer_origin = (f'{urlsplit(referrer).scheme}://{urlsplit(referrer).netloc}'
                              if referrer else None)
            if (self.headers.get('Sec-Fetch-Site') == 'cross-site'
                    or (origin and origin not in allowed)
                    or (referer_origin and referer_origin not in allowed)):
                self.send_error(403, 'Open the laboratory from the local book')
                return
            try:
                target = notebook_target(request.query, self.labs)
            except ValueError as error:
                self.send_error(404, str(error))
                return
            destination = JUPYTER_URL + '/lab' + ('/tree/'+quote(target, safe='/') if target else '')
            self.send_response(303)
            self.send_header('Location', destination+'?token='+self.token)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        super().do_GET()

def jupyter_request(path, token, method='GET'):
    request = Request(JUPYTER_URL+path, method=method,
                      headers={'Authorization': 'token '+token})
    with urlopen(request, timeout=2) as response:
        return response.read()

def kernel_configuration(directory, root=ROOT):
    kernel = directory/'data/kernels/aibook'
    kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'AI Book (project Python)', 'language': 'python',
        'env': {'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'MPLBACKEND': 'module://matplotlib_inline.backend_inline'}
    }), encoding='utf-8')
    config = {'ServerApp': {'ip': '127.0.0.1', 'port': 8888, 'port_retries': 0,
              'open_browser': False, 'root_dir': str(root), 'allow_remote_access': False,
              'trust_xheaders': False},
              'KernelSpecManager': {'allowed_kernelspecs': ['aibook']}}
    path = directory/'jupyter_config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    return path

def stop_jupyter(process, token):
    if process is None or process.poll() is not None:
        return
    try:
        jupyter_request('/api/shutdown', token, 'POST')
    except (URLError, OSError, TimeoutError):
        pass
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check this interpreter without starting servers')
    parser.add_argument('--no-browser', action='store_true', help='Print URLs without opening a browser')
    parser.add_argument('--smoke-test', action='store_true', help='Start, verify both services, then stop')
    args = parser.parse_args(argv)
    errors = environment_errors()
    if errors:
        print('\n'.join(errors))
        print(f'Install once into this environment:\n"{sys.executable}" -m pip install -r "{ROOT / "requirements-notebooks.txt"}"')
        return 1
    print('Python:', sys.executable, flush=True)
    if args.check:
        print('Environment ready. Run: python run.py')
        return 0
    for port in (8000, 8888):
        if not port_available(port):
            print(f'Port {port} is occupied. Stop your previous book/Jupyter server and retry. No existing process was stopped.')
            return 1
    if needs_build():
        print('Building the book (your notebook edits are preserved)...', flush=True)
        subprocess.run([sys.executable, '-B', '-m', 'tools.build_book'], cwd=ROOT, check=True)
    from book_src.laboratories import catalog
    labs, token = catalog(ROOT), secrets.token_urlsafe(32)
    server, process, thread = None, None, None
    try:
        with tempfile.TemporaryDirectory(prefix='aibook-session-') as temporary:
            directory = Path(temporary)
            config = kernel_configuration(directory)
            env = {**os.environ, 'JUPYTER_TOKEN': token, 'JUPYTER_CONFIG_DIR': str(directory),
                   'JUPYTER_DATA_DIR': str(directory/'data'), 'JUPYTER_RUNTIME_DIR': str(directory/'runtime'),
                   'JUPYTER_PATH': str(directory/'data'), 'IPYTHONDIR': str(directory/'ipython'),
                   'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
            for variable in ('JUPYTER_TOKEN_FILE', 'JUPYTER_CONFIG_PATH'):
                env.pop(variable, None)
            handler = partial(LearningHandler, directory=str(ROOT/'dist'), token=token, labs=labs)
            server = ThreadingHTTPServer(('127.0.0.1', 8000), handler)
            with (directory/'jupyter.log').open('w+', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-m', 'jupyterlab', '--config='+str(config)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                try:
                    deadline = time.monotonic()+60
                    while True:
                        if process.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            detail = log.read()[-4000:].replace(token, '[private token]')
                            raise RuntimeError('Jupyter did not start.\n'+detail)
                        try:
                            jupyter_request('/api/status', token)
                            break
                        except (URLError, OSError, TimeoutError):
                            time.sleep(0.2)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    print('\nAI Book — ready\nBook: '+BOOK_URL+'/\nLearning desk: '+BOOK_URL+'/start.html\nJupyter (private URL): '+JUPYTER_URL+'/lab?token='+token+
                          '\nStart: Lesson 01 → its laboratory.\nKeep this terminal open. Ctrl+C stops BOTH services.', flush=True)
                    if args.smoke_test:
                        with urlopen(BOOK_URL+'/start.html', timeout=5) as response:
                            assert response.status == 200
                        kernels = json.loads(jupyter_request('/api/kernelspecs', token))
                        assert kernels['kernelspecs']['aibook']['spec']['argv'][0] == sys.executable
                        print('PASS: book, authenticated Jupyter, and project interpreter.', flush=True)
                    else:
                        if not args.no_browser:
                            webbrowser.open(BOOK_URL+'/start.html')
                        while process.poll() is None:
                            time.sleep(0.3)
                        raise RuntimeError('Jupyter stopped; the book server is also stopping.')
                finally:
                    if thread:
                        server.shutdown()
                    server.server_close()
                    stop_jupyter(process, token)
    except KeyboardInterrupt:
        print('\nStopped book and Jupyter. Saved notebooks are unchanged.')
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error))
        return 1
    finally:
        if server:
            server.server_close()
        stop_jupyter(process, token)
    return 0
