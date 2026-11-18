from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from ragstudy.config import Config
from ragstudy.io import append_jsonl, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from ragstudy.pilot_collection import REQUIRED_SYSTEMS, collect_pilot, parse_run_selections


class PilotCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.dataset = "medical"
        self.definition = {"family": "benchmark", "domain": "medical", "sample_size": 3, "pilot_size": 3, "source_url": "source", "licence": "test"}
        self.config = Config(
            path=self.root / "config" / "experiment.toml",
            root=self.root,
            data={
                "study": {"output_root": "results", "data_root": "data", "seed": 7, "protocol_version": "1.0"},
                "common": {"chunk_tokens": 600, "chunk_overlap_tokens": 100, "concurrent_workers": 1},
                "provider": {
                    "generator_model": "generator", "embedding_model": "embedding",
                    "requests_per_minute": 30, "generator_input_per_million": 1,
                    "generator_output_per_million": 2, "embedding_input_per_million": 0.1,
                },
                "datasets": {self.dataset: self.definition},
                "systems": {system: {} for system in REQUIRED_SYSTEMS},
            },
        )
        prepared = self.config.data_root / "prepared" / self.dataset
        prepared.mkdir(parents=True)
        write_jsonl(prepared / "documents.jsonl", [{"id": "d1", "text": "Alpha evidence"}])
        self.questions = [
            {"id": "q1", "question": "Alpha?", "reference_answer": "Alpha", "evidence": ["Alpha evidence"]},
            {"id": "q2", "question": "Beta?", "reference_answer": "Beta", "evidence": ["Beta evidence"]},
        ]
        write_jsonl(prepared / "pilot_set.jsonl", self.questions)
        self.selections = {system: f"run-{system}" for system in REQUIRED_SYSTEMS}
        for system in REQUIRED_SYSTEMS:
            self._write_run(system)
        # This row must never leak into an explicitly selected collection.
        append_jsonl(self.config.output_root / "query_traces.jsonl", {
            "run_id": "stale-run", "phase": "pilot", "dataset": self.dataset, "system": "naive",
            "question_id": "q1", "question": "stale", "answer": "stale", "contexts": [],
            "elapsed_seconds": 999, "status": "ok", "repetition": 1, "usage": {}, "metadata": {},
        })

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def _spec(self, system: str) -> dict:
        prepared = self.config.data_root / "prepared" / self.dataset
        return {
            "phase": "pilot", "systems": [system], "repetition": 1, "protocol_version": "1.0",
            "inputs": {self.dataset: {"documents": sha256_file(prepared / "documents.jsonl"), "questions": sha256_file(prepared / "pilot_set.jsonl")}},
            "config": {
                "common": self.config.data["common"], "provider": self.config.data["provider"],
                "datasets": {self.dataset: self.definition}, "systems": {system: {}},
            },
        }

    def _write_run(self, system: str) -> None:
        run_id = self.selections[system]
        run_root = self.config.output_root / "runs" / run_id
        manifest = {
            "run_id": run_id, "phase": "pilot", "systems": [system], "repetition": 1,
            "status": "complete", "counts": {"indexed": 1, "indexes_incomplete": 0, "queries_ok": 2, "queries_failed": 0, "queries_skipped": 0},
            "spec": self._spec(system),
        }
        write_json(run_root / "run_manifest.json", manifest)
        write_json(run_root / "run_summary.json", {"run_id": run_id, "phase": "pilot", "systems": [system], "complete": True})
        workspace = run_root / self.dataset / system
        write_json(workspace / "index_run.json", {"run_id": run_id, "phase": "pilot", "dataset": self.dataset, "system": system, "status": "ok", "report": {"complete": True}})
        write_jsonl(workspace / "usage_events.jsonl", [
            {"event_id": f"{system}-index", "phase": "pilot", "system": system, "operation": "index", "model_kind": "embedding", "model": "embedding", "input_tokens": 100, "output_tokens": 0, "api_calls": 1, "usage_source": "provider", "status": "ok", "attempt": 1},
            {"event_id": f"{system}-query", "phase": "pilot", "system": system, "operation": "query", "model_kind": "generator", "model": "generator", "input_tokens": 20, "output_tokens": 5, "api_calls": 1, "usage_source": "estimated" if system == "lightrag" else "provider", "status": "ok", "attempt": 1},
        ])
        for index, question in enumerate(self.questions, 1):
            contexts = [{"id": "d1", "text": "Alpha evidence"}] if system == "naive" else []
            append_jsonl(self.config.output_root / "query_traces.jsonl", {
                "run_id": run_id, "phase": "pilot", "dataset": self.dataset, "system": system,
                "question_id": question["id"], "question": question["question"], "answer": question["reference_answer"],
                "contexts": contexts, "elapsed_seconds": float(index), "status": "ok", "repetition": 1,
                "usage": {"api_calls": 1}, "metadata": {"index_complete": True},
            })

    def test_requires_explicit_short_pilot_override_and_creates_audited_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prepared pilot has 2 questions"):
            collect_pilot(self.config, self.dataset, self.selections)

        append_jsonl(self.config.output_root / "query_traces.jsonl", {
            "run_id": self.selections["graphrag"], "phase": "pilot", "dataset": self.dataset,
            "system": "graphrag", "question_id": "q1", "question": "Alpha?", "answer": None,
            "contexts": [], "elapsed_seconds": 0.5, "status": "failed", "repetition": 1,
            "usage": {}, "metadata": {"index_complete": True}, "error": "earlier attempt",
        })
        result = collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)
        output = Path(result["path"])
        self.assertTrue(result["created"])
        self.assertEqual({path.name for path in output.iterdir()}, {
            "collection_manifest.json", "quality_metrics.jsonl", "report.md", "summary.csv",
            "summary.json", "traces.jsonl", "usage_events.jsonl",
        })
        manifest = read_json(output / "collection_manifest.json")
        summary = read_json(output / "summary.json")
        self.assertTrue(manifest["allow_short_pilot"])
        self.assertEqual(manifest["actual_pilot_size"], 2)
        self.assertEqual(len(read_jsonl(output / "traces.jsonl")), 7)
        self.assertEqual(len(read_jsonl(output / "usage_events.jsonl")), 6)
        self.assertEqual(summary["usage"]["api_calls"], 6)
        self.assertNotIn("stale-run", (output / "traces.jsonl").read_text(encoding="utf-8"))
        graph = next(row for row in summary["systems"] if row["system"] == "graphrag")
        self.assertEqual(graph["failed_trace_rows"], 1)
        self.assertEqual(graph["quality"]["evidence_recall_proxy"]["coverage"], 0.0)
        self.assertIn("contexts are unavailable", (output / "report.md").read_text(encoding="utf-8").lower())
        with (output / "summary.csv").open(encoding="utf-8") as source:
            csv_rows = list(csv.DictReader(source))
        self.assertEqual(len(csv_rows), 3)
        self.assertIn("index_input_tokens", csv_rows[0])
        self.assertIn("query_estimated_cost_usd", csv_rows[0])

        repeated = collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)
        self.assertFalse(repeated["created"])
        self.assertEqual(repeated["collection_id"], result["collection_id"])

    def test_blocks_incomplete_wrong_system_and_incompatible_runs(self) -> None:
        manifest_path = self.config.output_root / "runs" / self.selections["graphrag"] / "run_manifest.json"
        manifest = read_json(manifest_path)
        manifest["status"] = "incomplete"
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "must be a complete pilot"):
            collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)

        manifest["status"] = "complete"
        manifest["systems"] = ["lightrag"]
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "only declared system"):
            collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)

        manifest["systems"] = ["graphrag"]
        manifest["spec"]["config"]["common"]["chunk_tokens"] = 999
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)

    def test_blocks_missing_or_duplicate_successful_questions(self) -> None:
        trace_path = self.config.output_root / "query_traces.jsonl"
        rows = read_jsonl(trace_path)
        rows = [row for row in rows if not (row.get("run_id") == self.selections["lightrag"] and row.get("question_id") == "q2")]
        write_jsonl(trace_path, rows)
        with self.assertRaisesRegex(ValueError, "missing successful pilot questions: q2"):
            collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)

        duplicate = next(row for row in rows if row.get("run_id") == self.selections["lightrag"] and row.get("question_id") == "q1")
        append_jsonl(trace_path, duplicate)
        with self.assertRaisesRegex(ValueError, "multiple successful rows"):
            collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)

    def test_blocks_missing_usage_operation(self) -> None:
        events_path = self.config.output_root / "runs" / self.selections["naive"] / self.dataset / "naive" / "usage_events.jsonl"
        write_jsonl(events_path, [row for row in read_jsonl(events_path) if row["operation"] == "query"])
        with self.assertRaisesRegex(ValueError, "must include index and query"):
            collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)

    def test_blocks_wrong_phase_and_missing_dataset(self) -> None:
        manifest_path = self.config.output_root / "runs" / self.selections["naive"] / "run_manifest.json"
        manifest = read_json(manifest_path)
        manifest["phase"] = "full"
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "must be a complete pilot"):
            collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)

        manifest["phase"] = "pilot"
        del manifest["spec"]["inputs"][self.dataset]
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "does not include dataset"):
            collect_pilot(self.config, self.dataset, self.selections, allow_short_pilot=True)

    def test_parse_run_selections_requires_exact_distinct_systems(self) -> None:
        self.assertEqual(parse_run_selections([f"{system}={self.selections[system]}" for system in REQUIRED_SYSTEMS]), self.selections)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_run_selections(["naive=r1", "naive=r2", "graphrag=r3", "lightrag=r4"])
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            parse_run_selections(["naive=r1", "graphrag=r2"])
        with self.assertRaisesRegex(ValueError, "different run ID"):
            parse_run_selections(["naive=r1", "graphrag=r1", "lightrag=r3"])


if __name__ == "__main__":
    unittest.main()
