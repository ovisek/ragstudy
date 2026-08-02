from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts" / "graphrag_bridge.py"
SPEC = importlib.util.spec_from_file_location("graphrag_bridge", SCRIPT)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class GraphRagBridgeCliTest(unittest.TestCase):
    def test_cli_is_resolved_beside_active_python_without_path_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir) / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            graphrag = bin_dir / "graphrag"
            python.touch()
            graphrag.touch()

            with patch.object(sys, "executable", str(python)):
                self.assertEqual(bridge.graphrag_cli(), str(graphrag))

    def test_missing_cli_reports_active_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            python = Path(temp_dir) / "bin" / "python"

            with patch.object(sys, "executable", str(python)):
                with self.assertRaisesRegex(RuntimeError, "not installed beside the active Python"):
                    bridge.graphrag_cli()

    def test_child_environment_enables_python_safe_path(self) -> None:
        study = {"provider": {"api_key_env": "TEST_GRAPHRAG_KEY"}}
        with patch.dict(os.environ, {"TEST_GRAPHRAG_KEY": "secret"}):
            environment = bridge.environment(study)

        self.assertEqual(environment["PYTHONSAFEPATH"], "1")

    def test_settings_limit_concurrency_and_requests_per_minute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "settings.yaml").write_text("completion_models: {}\nembedding_models: {}\n", encoding="utf-8")
            study = {
                "provider": {
                    "generator_model": "generator",
                    "embedding_model": "embedding",
                    "base_url": "https://example.test/v1",
                    "requests_per_minute": 30,
                },
                "common": {"chunk_tokens": 600, "chunk_overlap_tokens": 100, "concurrent_workers": 1},
                "systems": {"graphrag": {}},
            }

            bridge.configure_settings(workspace, study)
            settings = yaml.safe_load((workspace / "settings.yaml").read_text(encoding="utf-8"))

            self.assertEqual(settings["concurrent_requests"], 1)
            self.assertEqual(settings["completion_models"]["default_completion_model"]["rate_limit"]["requests_per_period"], 30)
            self.assertEqual(settings["embedding_models"]["default_embedding_model"]["rate_limit"]["requests_per_period"], 30)


if __name__ == "__main__":
    unittest.main()
