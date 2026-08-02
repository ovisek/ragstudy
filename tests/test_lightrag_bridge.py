from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "lightrag_bridge.py"
SPEC = importlib.util.spec_from_file_location("lightrag_bridge", SCRIPT)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class LightRagBridgeConfigurationTest(unittest.TestCase):
    def test_chunk_concurrency_defaults_to_serial(self) -> None:
        study = {"systems": {"lightrag": {}}}
        self.assertEqual(bridge.chunk_llm_concurrency(study), 1)

    def test_chunk_concurrency_accepts_positive_integer(self) -> None:
        study = {"systems": {"lightrag": {"chunk_llm_concurrency": 4}}}
        self.assertEqual(bridge.chunk_llm_concurrency(study), 4)

    def test_chunk_concurrency_rejects_invalid_values(self) -> None:
        for value in (0, -1, True, 1.5, "4"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                bridge.chunk_llm_concurrency({"systems": {"lightrag": {"chunk_llm_concurrency": value}}})


if __name__ == "__main__":
    unittest.main()
