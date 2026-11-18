from __future__ import annotations

import importlib.util
import asyncio
import unittest
from unittest.mock import AsyncMock, patch
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

    def test_embedding_timeout_defaults_to_ten_minutes(self) -> None:
        self.assertEqual(bridge.embedding_timeout_seconds({"provider": {}}), 600)

    def test_embedding_timeout_accepts_positive_integer(self) -> None:
        self.assertEqual(
            bridge.embedding_timeout_seconds(
                {"provider": {"embedding_timeout_seconds": 900}}
            ),
            900,
        )

    def test_embedding_timeout_rejects_invalid_values(self) -> None:
        for value in (0, -1, True, 600.0, "600"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                bridge.embedding_timeout_seconds(
                    {"provider": {"embedding_timeout_seconds": value}}
                )

    def test_generator_timeouts_have_finite_defaults(self) -> None:
        study = {"provider": {}}
        self.assertEqual(bridge.generator_request_timeout_seconds(study), 300)
        self.assertEqual(bridge.generator_worker_timeout_seconds(study), 600)

    def test_generator_timeouts_accept_positive_integers(self) -> None:
        study = {
            "provider": {
                "generator_request_timeout_seconds": 240,
                "generator_worker_timeout_seconds": 720,
            }
        }
        self.assertEqual(bridge.generator_request_timeout_seconds(study), 240)
        self.assertEqual(bridge.generator_worker_timeout_seconds(study), 720)

    def test_generator_timeouts_reject_invalid_values(self) -> None:
        for key, func in (
            ("generator_request_timeout_seconds", bridge.generator_request_timeout_seconds),
            ("generator_worker_timeout_seconds", bridge.generator_worker_timeout_seconds),
        ):
            for value in (0, -1, True, 600.0, "600"):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, "positive integer"
                ):
                    func({"provider": {key: value}})

    def test_request_max_attempts_is_configurable(self) -> None:
        self.assertEqual(bridge.request_max_attempts({"provider": {}}), 3)
        self.assertEqual(bridge.request_max_attempts({"provider": {"request_max_attempts": 6}}), 6)
        for value in (0, -1, True, 3.0, "3"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                bridge.request_max_attempts({"provider": {"request_max_attempts": value}})

    def test_retry_delay_is_bounded_exponential_with_jitter(self) -> None:
        study = {"provider": {"retry_backoff_seconds": 4.0, "retry_max_delay_seconds": 10.0}}
        with patch.object(bridge.random, "uniform", return_value=0.5):
            self.assertEqual(bridge.retry_delays(study, 4), [4.5, 8.5, 10.0, 10.0])

    def test_only_retryable_status_codes_are_retried(self) -> None:
        class ProviderError(Exception):
            def __init__(self, status_code=None):
                self.status_code = status_code

        retryable = (ProviderError,)
        for status in (None, 408, 409, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(bridge.is_retryable_provider_error(ProviderError(status), retryable))
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertFalse(bridge.is_retryable_provider_error(ProviderError(status), retryable))

    def test_cache_settings_are_explicit(self) -> None:
        study = {"systems": {"lightrag": {
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
            "enable_embedding_cache": True,
        }}}
        self.assertEqual(bridge.lightrag_cache_settings(study), {
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
            "enable_embedding_cache": True,
        })

    def test_query_cache_can_be_bypassed_without_disabling_extraction_cache(self) -> None:
        study = {"systems": {"lightrag": {
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
            "enable_embedding_cache": True,
        }}}
        with patch.dict(bridge.os.environ, {"RAGSTUDY_BYPASS_QUERY_CACHE": "1"}):
            self.assertEqual(bridge.lightrag_cache_settings(study, "query"), {
                "enable_llm_cache": False,
                "enable_llm_cache_for_entity_extract": True,
                "enable_embedding_cache": True,
            })
            self.assertTrue(bridge.environment_flag("RAGSTUDY_BYPASS_QUERY_CACHE"))
        self.assertTrue(bridge.lightrag_cache_settings(study, "index")["enable_llm_cache"])

    def test_request_pacer_uses_configured_rpm(self) -> None:
        pacer = bridge.AsyncRequestPacer(30)
        self.assertEqual(pacer.minimum_interval, 2.0)

        async def exercise() -> None:
            await pacer.wait()

        asyncio.run(exercise())

    def test_finalize_rag_stops_background_workers(self) -> None:
        class Worker:
            def __init__(self) -> None:
                self.shutdown = AsyncMock()

        class Embedding:
            func = Worker()

        class Rag:
            def __init__(self) -> None:
                self.finalize_storages = AsyncMock()
                self.llm_model_func = Worker()
                self.embedding_func = Embedding()

        rag = Rag()
        asyncio.run(bridge.finalize_rag(rag))
        rag.finalize_storages.assert_awaited_once()
        rag.llm_model_func.shutdown.assert_awaited_once()
        rag.embedding_func.func.shutdown.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
