"""Deployment-only tests; not part of the downloadable Mini-GPT project."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.prepare_release import check_existing_output, check_release_destinations, inventory, run, write_archive


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.public = self.root / 'public'
        self.public.mkdir()
        (self.public / 'index.html').write_text('<!doctype html>آزمون', encoding='utf-8')

    def test_hashes_and_sizes_are_exact(self):
        data = (self.public / 'index.html').read_bytes()
        self.assertEqual(inventory(self.public)['index.html'], {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
        })

    def test_private_paths_are_rejected(self):
        (self.public / '.env').write_text('not-a-secret', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Private/cache'):
            inventory(self.public)

    def test_training_files_are_rejected(self):
        (self.public / 'last.pt').write_bytes(b'test')
        with self.assertRaisesRegex(ValueError, 'Unexpected public'):
            inventory(self.public)

    def test_unexpected_archive_is_rejected(self):
        (self.public / 'backup.zip').write_bytes(b'test')
        with self.assertRaisesRegex(ValueError, 'Unexpected downloadable'):
            inventory(self.public)

    def test_stale_files_fail_without_deletion(self):
        expected = inventory(self.public)
        stale = self.public / 'obsolete.html'
        stale.write_text('old page', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unexpected files in dist'):
            check_existing_output(self.public, expected)
        self.assertTrue(stale.is_file())

    def test_reproducible_archive_has_no_source_prefix(self):
        files = inventory(self.public)
        first, second = self.root / 'one.zip', self.root / 'two.zip'
        write_archive(self.public, first, files)
        write_archive(self.public, second, files)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(archive.namelist(), ['index.html'])
            self.assertIsNone(archive.testzip())

    def test_changed_input_is_rejected(self):
        files = inventory(self.public)
        (self.public / 'index.html').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed during release'):
            write_archive(self.public, self.root / 'site.zip', files)

    def test_missing_entry_point_is_rejected(self):
        (self.public / 'index.html').unlink()
        with self.assertRaisesRegex(ValueError, 'no index.html'):
            inventory(self.public)

    def test_optimized_environment_cannot_disable_release_checks(self):
        with patch.dict(os.environ, {'PYTHONOPTIMIZE': '2'}):
            with patch('tools.prepare_release.subprocess.run') as process:
                run('python', '-B', '-m', 'tools.validate_book')
        self.assertNotIn('PYTHONOPTIMIZE', process.call_args.kwargs['env'])
        self.assertTrue(process.call_args.kwargs['check'])

    def test_all_destination_types_are_preflighted_without_replacing_files(self):
        archive = self.root / 'book-site.zip'
        archive.write_bytes(b'previous release')
        (self.root / 'manifest.json').mkdir()
        with self.assertRaisesRegex(ValueError, 'non-file release target'):
            check_release_destinations(self.root)
        self.assertEqual(archive.read_bytes(), b'previous release')


if __name__ == '__main__':
    unittest.main()
