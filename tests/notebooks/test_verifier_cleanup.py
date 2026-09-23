"""Optional notebook-tooling tests; requires the Jupyter development dependencies."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, call, patch

import nbformat

from tools import verify_notebooks


ROOT = Path(__file__).resolve().parents[2]
SELECTED = "lessons/26b-sequence-memory/lab.ipynb"


class VerifierCleanupTests(unittest.TestCase):
    def setUp(self):
        self.events = Mock()
        self.manager = Mock()
        self.manager.has_kernel = True
        self.client = Mock()
        self.client.execute.side_effect = self.execute
        for name, operation in (
            ("execute", self.client.execute),
            ("stop_channels", self.client.kc.stop_channels),
            ("shutdown_kernel", self.manager.shutdown_kernel),
            ("cleanup_resources", self.manager.cleanup_resources),
        ):
            self.events.attach_mock(operation, name)

    def make_client(self, notebook, **kwargs):
        self.assertIs(kwargs["km"], self.manager)
        self.assertFalse(kwargs["allow_errors"])
        self.notebook = notebook
        return self.client

    def execute(self):
        # Real, validated source notebook; simulate only its captured output.
        first_code = next(cell for cell in self.notebook.cells if cell.cell_type == "code")
        first_code.outputs = [nbformat.v4.new_output(
            "stream", name="stdout", text=f"Project: {ROOT}\nPython: {sys.executable}\n")]

    def verify_once(self):
        with patch.object(verify_notebooks, "KernelManager", return_value=self.manager), \
                patch.object(verify_notebooks, "NotebookClient", side_effect=self.make_client), \
                redirect_stdout(StringIO()):
            return verify_notebooks.verify(ROOT, selected=SELECTED)

    def assert_full_cleanup_order(self):
        self.assertEqual(self.events.mock_calls, [
            call.execute(), call.stop_channels(),
            call.shutdown_kernel(now=True), call.cleanup_resources(),
        ])

    def test_success_stops_channels_before_shutdown_and_resource_cleanup(self):
        result = self.verify_once()
        self.assertEqual(len(result["notebooks"]), 1)
        self.assertEqual(result["notebooks"][0]["path"], "notebooks/" + SELECTED)
        self.assert_full_cleanup_order()

    def test_execution_failure_still_closes_client_and_manager(self):
        error = RuntimeError("execution failed")
        self.client.execute.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.verify_once()
        self.assertIs(caught.exception, error)
        self.assert_full_cleanup_order()

    def test_shutdown_failure_still_cleans_manager_resources(self):
        error = RuntimeError("shutdown failed")
        self.manager.shutdown_kernel.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.verify_once()
        self.assertIs(caught.exception, error)
        self.assert_full_cleanup_order()

    def test_channel_failure_still_shuts_down_and_cleans_manager(self):
        error = RuntimeError("channel stop failed")
        self.client.kc.stop_channels.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.verify_once()
        self.assertIs(caught.exception, error)
        self.assert_full_cleanup_order()

    def test_startup_failure_without_client_still_cleans_resources(self):
        self.client.kc = None
        self.manager.has_kernel = False
        error = RuntimeError("startup failed")
        self.client.execute.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.verify_once()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.events.mock_calls, [call.execute(), call.cleanup_resources()])


if __name__ == "__main__":
    unittest.main()
