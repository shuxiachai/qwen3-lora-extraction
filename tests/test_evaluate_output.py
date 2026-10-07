"""Existing evaluation output must survive; dry-run remains a read-only plan."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate


class EvaluationOutputTests(unittest.TestCase):
    def setUp(self):
        self.config = {"model": {"id": "dummy"}, "evaluation": {"do_sample": False}}

    def test_run_refuses_existing_output_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("keep me", encoding="utf-8")
            with patch.object(sys, "argv", ["evaluate.py", "--run", "--output", str(output)]), \
                 patch.object(evaluate, "load_config", return_value=self.config), \
                 patch.object(evaluate, "local_cache_environment"), \
                 patch("src.modeling.load_base_model") as loader:
                with self.assertRaises(FileExistsError):
                    evaluate.main()
                loader.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

    def test_dry_run_does_not_overwrite_or_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("keep me", encoding="utf-8")
            stream = io.StringIO()
            with patch.object(sys, "argv", ["evaluate.py", "--output", str(output)]), \
                 patch.object(evaluate, "load_config", return_value=self.config), \
                 patch.object(evaluate, "local_cache_environment"), \
                 patch("src.modeling.load_base_model") as loader, \
                 contextlib.redirect_stdout(stream):
                evaluate.main()
                loader.assert_not_called()
            self.assertIn('"mode": "dry_run"', stream.getvalue())
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")
