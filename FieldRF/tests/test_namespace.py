"""Installed FieldRF imports coexist with unrelated top-level utils modules."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NamespaceTests(unittest.TestCase):
    def test_imports_preserve_unrelated_utils(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "utils.py").write_text("sentinel = 'external utils'\n")
            env = dict(os.environ, PYTHONPATH=directory)
            source = "import utils; from fieldrf.models import FieldRFModel; import fieldrf.training.vae; assert utils.sentinel == 'external utils'"
            result = subprocess.run([sys.executable, "-c", source], cwd=directory,
                                    env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
