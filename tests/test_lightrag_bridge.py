from __future__ import annotations

import importlib.util
import asyncio
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "lightrag_bridge.py"
SPEC = importlib.util.spec_from_file_location("lightrag_bridge", SCRIPT)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class LightRagBridgeConfigurationTest(unittest.TestCase):
    def test_concurrent_workers_defaults_to_serial(self) -> None:
        study = {"common": {}}
        self.assertEqual(bridge.concurrent_workers(study), 1)

    def test_concurrent_workers_accepts_positive_integer(self) -> None:
        study = {"common": {"concurrent_workers": 4}}
        self.assertEqual(bridge.concurrent_workers(study), 4)

    def test_concurrent_workers_rejects_invalid_values(self) -> None:
        for value in (0, -1, True, 1.5, "4"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                bridge.concurrent_workers({"common": {"concurrent_workers": value}})

    def test_embedding_dimension_is_explicit_and_positive(self) -> None:
        self.assertEqual(bridge.embedding_dimension({"provider": {"embedding_dimension": 2048}}), 2048)
        for value in (None, 0, -1, True, 1536.0, "2048"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                bridge.embedding_dimension({"provider": {"embedding_dimension": value}})

    def test_request_pacer_uses_configured_rpm(self) -> None:
        pacer = bridge.AsyncRequestPacer(30)
        self.assertEqual(pacer.minimum_interval, 2.0)

        async def exercise() -> None:
            await pacer.wait()

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
