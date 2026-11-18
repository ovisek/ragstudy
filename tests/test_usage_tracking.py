from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ragstudy.adapters import NaiveRagAdapter
from ragstudy.io import read_jsonl
from ragstudy.usage_tracking import ModelUsageTracker, UsageRecorder, summarize_usage, usage_cost_usd


class UsageTrackingTest(unittest.TestCase):
    def test_journal_preserves_exact_and_partial_estimated_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = UsageRecorder(Path(temp_dir) / "usage_events.jsonl", system="lightrag", operation="query")
            tracker = ModelUsageTracker(recorder, model="generator", model_kind="generator")
            tracker.add_usage({"prompt_tokens": 12, "completion_tokens": 5})
            recorder.record(
                model="embedding", model_kind="embedding", input_tokens=7,
                output_tokens=None, usage_source="partial_estimate", status="failed",
            )

            events = recorder.events_since(0)
            summary = summarize_usage(events)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["usage_source"], "provider")
        self.assertEqual(summary["input_tokens"], 19)
        self.assertEqual(summary["output_tokens"], 5)
        self.assertEqual(summary["api_calls"], 2)
        self.assertEqual(summary["failed_calls"], 1)
        self.assertEqual(summary["unknown_output_calls"], 1)
        self.assertEqual(len(summary["by_operation_model"]), 2)

    def test_cost_uses_separate_generator_and_embedding_prices(self) -> None:
        summary = {
            "by_operation_model": [
                {"model_kind": "generator", "input_tokens": 1_000_000, "output_tokens": 500_000},
                {"model_kind": "embedding", "input_tokens": 2_000_000, "output_tokens": 0},
            ]
        }
        provider = {
            "generator_input_per_million": 1,
            "generator_output_per_million": 2,
            "embedding_input_per_million": 0.1,
            "embedding_output_per_million": 0,
        }
        self.assertAlmostEqual(usage_cost_usd(summary, provider), 2.2)

    def test_naive_adapter_journals_failed_retry_and_provider_usage(self) -> None:
        class RetryableError(Exception):
            status_code = None

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = NaiveRagAdapter.__new__(NaiveRagAdapter)
            adapter.workspace = Path(temp_dir)
            adapter.system = "naive"
            adapter.phase = "pilot"
            adapter.retryable_errors = (RetryableError,)
            calls = 0

            def provider_call():
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RetryableError("temporary")
                return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3))

            with patch("ragstudy.adapters.time.sleep"):
                adapter._provider_call(
                    provider_call, operation="query", model="generator", model_kind="generator",
                    request_value="prompt", output_value=lambda _: "answer",
                )
            events = read_jsonl(adapter.workspace / "usage_events.jsonl")

        self.assertEqual([event["attempt"] for event in events], [1, 2])
        self.assertEqual([event["status"] for event in events], ["failed", "ok"])
        self.assertEqual(events[0]["usage_source"], "partial_estimate")
        self.assertEqual(events[1]["input_tokens"], 8)
        self.assertEqual(events[1]["output_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
