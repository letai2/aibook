"""Local launcher routing, lifecycle, and single-interpreter safety contracts."""
from contextlib import redirect_stdout
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, patch
from urllib.error import URLError
from urllib.parse import quote

from tools.learning_server import (
    LearningHandler, kernel_configuration, main, needs_build, notebook_target,
    stop_jupyter,
)


class LauncherTests(unittest.TestCase):
    def test_only_known_notebooks_can_be_opened(self):
        labs = [{'primary_lesson':'01-model','kind':'lesson','path':'notebooks/lessons/01-model/lab.ipynb'},
                {'id':'review','kind':'review','path':'notebooks/review.ipynb'}]
        self.assertEqual(notebook_target('lesson=01-model',labs),labs[0]['path'])
        self.assertEqual(notebook_target('review=review',labs),labs[1]['path'])
        self.assertEqual(notebook_target('',labs),'')
        for query in ('lesson=../../private','lesson=https://evil.test','path=secret',
                      'lesson=01-model&lesson=01-model','lesson=01-model&token=x','lesson='):
            with self.subTest(query=query), self.assertRaises(ValueError):
                notebook_target(query,labs)

    def test_kernel_uses_exact_interpreter_and_private_local_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / 'project with spaces' / 'کتاب'
            config = json.loads(kernel_configuration(root, root=project).read_text())
            kernel = json.loads((root/'data/kernels/aibook/kernel.json').read_text())
            self.assertEqual(kernel['argv'][0],sys.executable)
            self.assertEqual(kernel['argv'][1:],
                             ['-m', 'ipykernel_launcher', '-f', '{connection_file}'])
            self.assertEqual(config['ServerApp']['root_dir'], str(project))
            self.assertEqual(config['ServerApp']['ip'],'127.0.0.1')
            self.assertFalse(config['ServerApp']['allow_remote_access'])
            self.assertFalse(config['ServerApp']['trust_xheaders'])
            self.assertEqual(config['ServerApp']['port_retries'],0)
            self.assertEqual(config['KernelSpecManager']['allowed_kernelspecs'],['aibook'])
            self.assertNotIn('token',config.get('IdentityProvider',{}))

    def test_no_build_required_when_sources_are_older(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('run.py','requirements.txt','requirements-notebooks.txt'):
                (root/name).write_text('')
            self.assertTrue(needs_build(root))
            (root/'dist').mkdir()
            (root/'dist/assets').mkdir()
            for name in ('index.html','notebooks.html','assets/book.css','assets/manifest.js','start.html'):
                (root/'dist'/name).write_text('ready')
            self.assertFalse(needs_build(root))
            (root/'dist/index.html').unlink()
            self.assertTrue(needs_build(root))

    def test_edited_notebook_requires_build_but_checkpoint_cache_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('run.py', 'requirements.txt', 'requirements-notebooks.txt',
                         'notebooks/lesson.ipynb', 'dist/index.html', 'dist/notebooks.html',
                         'dist/assets/book.css', 'dist/assets/manifest.js', 'dist/start.html'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original', encoding='utf-8')
                os.utime(path, ns=(1_000_000_000, 1_000_000_000))
            marker = root / 'dist/start.html'
            os.utime(marker, ns=(2_000_000_000, 2_000_000_000))
            for directory in ('notebooks/.ipynb_checkpoints', 'book_src/__pycache__'):
                path = root / directory / 'ignored'
                path.parent.mkdir(parents=True)
                path.write_text('cache', encoding='utf-8')
                os.utime(path, ns=(3_000_000_000, 3_000_000_000))
            self.assertFalse(needs_build(root))
            os.utime(root / 'notebooks/lesson.ipynb', ns=(3_000_000_000, 3_000_000_000))
            self.assertTrue(needs_build(root))

    def test_shutdown_targets_only_owned_authenticated_server(self):
        process = Mock()
        process.poll.return_value = None
        with patch('tools.learning_server.jupyter_request') as request:
            stop_jupyter(process,'test-secret')
        request.assert_called_once_with('/api/shutdown','test-secret','POST')
        process.wait.assert_called_once_with(timeout=12)
        process.terminate.assert_not_called()

    def test_shutdown_does_nothing_without_a_running_owned_process(self):
        process = Mock()
        process.poll.return_value = 0
        with patch('tools.learning_server.jupyter_request') as request:
            stop_jupyter(None, 'test-secret')
            stop_jupyter(process, 'test-secret')
        request.assert_not_called()
        process.wait.assert_not_called()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_shutdown_escalates_only_on_owned_process_after_timeouts(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('owned-jupyter', 12),
                                    subprocess.TimeoutExpired('owned-jupyter', 5), 0]
        with patch('tools.learning_server.jupyter_request', side_effect=URLError('offline')) as request:
            stop_jupyter(process, 'test-secret')
        request.assert_called_once_with('/api/shutdown', 'test-secret', 'POST')
        self.assertEqual(process.method_calls, [call.poll(), call.wait(timeout=12),
                         call.terminate(), call.wait(timeout=5), call.kill(), call.wait(timeout=5)])

    def test_check_never_builds_or_starts_services(self):
        with (redirect_stdout(io.StringIO()),
              patch('tools.learning_server.environment_errors', return_value=[]),
              patch('tools.learning_server.port_available') as available,
              patch('tools.learning_server.subprocess.run') as build,
              patch('tools.learning_server.subprocess.Popen') as launch,
              patch('tools.learning_server.ThreadingHTTPServer') as server):
            self.assertEqual(main(['--check']), 0)
        available.assert_not_called()
        build.assert_not_called()
        launch.assert_not_called()
        server.assert_not_called()

    def test_occupied_ports_do_not_stop_existing_processes(self):
        for results, occupied in (([False], 8000), ([True, False], 8888)):
            with (self.subTest(port=occupied), redirect_stdout(io.StringIO()) as output,
                  patch('tools.learning_server.environment_errors', return_value=[]),
                  patch('tools.learning_server.port_available', side_effect=results),
                  patch('tools.learning_server.subprocess.run') as build,
                  patch('tools.learning_server.subprocess.Popen') as launch,
                  patch('tools.learning_server.ThreadingHTTPServer') as server,
                  patch('tools.learning_server.stop_jupyter') as stop):
                self.assertEqual(main(['--no-browser']), 1)
                self.assertIn(f'Port {occupied} is occupied', output.getvalue())
                self.assertIn('No existing process was stopped', output.getvalue())
                build.assert_not_called()
                launch.assert_not_called()
                server.assert_not_called()
                stop.assert_not_called()

    def test_failed_child_closes_book_socket_and_redacts_log_token(self):
        process = Mock()
        process.poll.return_value = 1
        server = Mock()

        def failed_child(*args, **kwargs):
            kwargs['stdout'].write('Could not start: test-private-token\n')
            kwargs['stdout'].flush()
            return process

        with (redirect_stdout(io.StringIO()) as output,
              patch('tools.learning_server.environment_errors', return_value=[]),
              patch('tools.learning_server.port_available', return_value=True),
              patch('tools.learning_server.needs_build', return_value=False),
              patch('tools.learning_server.secrets.token_urlsafe', return_value='test-private-token'),
              patch('tools.learning_server.ThreadingHTTPServer', return_value=server),
              patch('tools.learning_server.subprocess.Popen', side_effect=failed_child) as launch,
              patch('tools.learning_server.jupyter_request') as request,
              patch('tools.learning_server.webbrowser.open') as browser):
            self.assertEqual(main(['--no-browser']), 1)
        self.assertIn('[private token]', output.getvalue())
        self.assertNotIn('test-private-token', output.getvalue())
        server.server_close.assert_called()
        server.serve_forever.assert_not_called()
        request.assert_not_called()
        browser.assert_not_called()
        process.terminate.assert_not_called()
        self.assertEqual(launch.call_args.args[0][:3], [sys.executable, '-m', 'jupyterlab'])
        environment = launch.call_args.kwargs['env']
        self.assertEqual(environment['JUPYTER_TOKEN'], 'test-private-token')
        self.assertNotIn('JUPYTER_TOKEN_FILE', environment)
        self.assertNotIn('JUPYTER_CONFIG_PATH', environment)


class LauncherHTTPTests(unittest.TestCase):
    """Real requests, on an ephemeral loopback port; never start Jupyter."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='aibook-handler-test-')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        public = root / 'public'
        public.mkdir()
        (public / 'index.html').write_text('public book', encoding='utf-8')
        (public / 'launch.html').write_text('static launch fallback', encoding='utf-8')
        (public / 'empty').mkdir()
        private = root / 'private.txt'
        private.write_text('private file must stay private', encoding='utf-8')
        self.labs = [
            {'primary_lesson': '01-model', 'kind': 'lesson',
             'path': 'notebooks/lessons/01-model/lab.ipynb'},
            {'id': 'review', 'kind': 'review', 'path': 'notebooks/مرور درس.ipynb'},
        ]

        class BoundaryHandler(LearningHandler):
            def translate_path(self, path):
                # Exercise resolved-path containment without requiring Windows
                # symlink privileges. All other paths use the real translator.
                if path == '/outside-public':
                    return str(private)
                return super().translate_path(path)

        handler = partial(BoundaryHandler, directory=str(public),
                          token='test-private-token', labs=self.labs)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive(), 'The owned test server did not stop')

    def request(self, path, *, method='GET', headers=None, host='127.0.0.1:8000'):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            # Production validates its public origin, not this test's ephemeral
            # TCP port. Suppress http.client's automatic Host to test it exactly.
            connection.putrequest(method, path, skip_host=True)
            if host is not None:
                connection.putheader('Host', host)
            for name, value in (headers or {}).items():
                connection.putheader(name, value)
            connection.endheaders()
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def assert_private_response_absent(self, headers, body):
        self.assertNotIn('Location', headers)
        self.assertNotIn('test-private-token', str(headers))
        self.assertNotIn(b'test-private-token', body)

    def test_static_get_and_head_have_security_headers(self):
        for host in ('127.0.0.1:8000', 'localhost:8000'):
            for method in ('GET', 'HEAD'):
                with self.subTest(host=host, method=method):
                    status, headers, body = self.request('/index.html', method=method, host=host)
                    self.assertEqual(status, 200)
                    self.assertEqual(headers['Referrer-Policy'], 'no-referrer')
                    self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
                    self.assertEqual(body, b'public book' if method == 'GET' else b'')
                    self.assert_private_response_absent(headers, body)

    def test_missing_or_foreign_host_is_rejected_for_static_and_launcher(self):
        for host in (None, 'evil.test', '127.0.0.1:8888', 'localhost:8000.evil.test',
                     '127.0.0.1:8000@evil.test', '127.0.0.1'):
            for path, method in (('/index.html', 'GET'), ('/index.html', 'HEAD'),
                                 ('/launch.html?lesson=01-model', 'GET')):
                with self.subTest(host=host, path=path, method=method):
                    status, headers, body = self.request(path, method=method, host=host)
                    self.assertEqual(status, 403)
                    self.assert_private_response_absent(headers, body)

    def test_known_lesson_and_unicode_review_redirect_without_caching(self):
        for query, target in (('', ''), ('?lesson=01-model', self.labs[0]['path']),
                              ('?review=review', self.labs[1]['path'])):
            for origin in ('http://127.0.0.1:8000', 'http://localhost:8000'):
                with self.subTest(query=query, origin=origin):
                    status, headers, body = self.request('/launch.html' + query, headers={
                        'Origin': origin, 'Referer': origin + '/part-01/index.html',
                        'Sec-Fetch-Site': 'same-origin',
                    })
                    self.assertEqual(status, 303)
                    destination = 'http://127.0.0.1:8888/lab'
                    if target:
                        destination += '/tree/' + quote(target, safe='/')
                    self.assertEqual(headers['Location'], destination + '?token=test-private-token')
                    self.assertEqual(headers['Cache-Control'], 'no-store')
                    self.assertEqual(headers['Content-Length'], '0')
                    self.assertEqual(headers['Referrer-Policy'], 'no-referrer')
                    self.assertEqual(body, b'')

    def test_cross_site_launcher_requests_cannot_obtain_token(self):
        attempts = (
            {'Sec-Fetch-Site': 'cross-site'},
            {'Origin': 'https://evil.test'},
            {'Origin': 'null'},
            {'Origin': 'http://127.0.0.1:8000.evil.test'},
            {'Referer': 'https://evil.test/article'},
            {'Referer': 'http://127.0.0.1:8000@evil.test/article'},
            {'Origin': 'http://127.0.0.1:8000', 'Referer': 'https://evil.test/article'},
            {'Origin': 'http://127.0.0.1:8000', 'Sec-Fetch-Site': 'cross-site'},
        )
        for supplied in attempts:
            with self.subTest(headers=supplied):
                status, headers, body = self.request('/launch.html?lesson=01-model', headers=supplied)
                self.assertEqual(status, 403)
                self.assert_private_response_absent(headers, body)

    def test_launcher_rejects_non_whitelisted_and_ambiguous_targets(self):
        for query in ('lesson=missing', 'lesson=', 'lesson=..%2F..%2Fprivate',
                      'lesson=https%3A%2F%2Fevil.test', 'path=private.txt',
                      'lesson=01-model&lesson=01-model', 'lesson=01-model&token=other',
                      'lesson=01-model&review=review', 'review=missing', 'review=review&extra=1'):
            with self.subTest(query=query):
                status, headers, body = self.request('/launch.html?' + query)
                self.assertEqual(status, 404)
                self.assert_private_response_absent(headers, body)

    def test_head_launcher_does_not_disclose_token(self):
        status, headers, body = self.request('/launch.html?lesson=01-model', method='HEAD')
        self.assertEqual(status, 200)
        self.assertEqual(body, b'')
        self.assert_private_response_absent(headers, body)

    def test_directory_listing_and_outside_public_files_are_not_served(self):
        for path, expected in (('/empty/', 403), ('/outside-public', 403),
                               ('/../private.txt', 404), ('/%2e%2e/private.txt', 404)):
            for method in ('GET', 'HEAD'):
                with self.subTest(path=path, method=method):
                    status, headers, body = self.request(path, method=method)
                    self.assertEqual(status, expected)
                    self.assertNotIn(b'private file must stay private', body)
                    self.assert_private_response_absent(headers, body)

if __name__ == '__main__':
    unittest.main()
