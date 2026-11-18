from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragstudy.config import Config
from ragstudy.io import read_jsonl, write_json, write_jsonl
from ragstudy.models import Usage
from ragstudy.runner import run_update


class FakeAdapter:
    def __init__(self, system: str, workspace: Path, calls: list[dict[str, object]]):
        self.system = system
        self.workspace = workspace
        self.calls = calls

    def index(self, documents, incremental=False, resume=False, retries=2):
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.calls.append({
            "system": self.system,
            "workspace": str(self.workspace),
            "document_ids": [document.id for document in documents],
            "incremental": incremental,
        })
        return {"complete": True, "document_count": len(documents), "incremental": incremental}

    def query(self, question, method):
        return f"answer for {question.id}", [], Usage(input_tokens=10, output_tokens=2, api_calls=1), {"method": method}


class UpdateExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "data" / "prepared" / "graphrag_bench_medical"
        initial = [{"id": f"d{i}", "text": f"initial {i}"} for i in range(5)]
        added = [{"id": "d5", "text": "added evidence"}]
        questions = [
            {
                "id": f"q{i}", "question": f"question {i}", "reference_answer": f"answer {i}",
                "evidence": ["added evidence"], "evidence_document_ids": ["d5"],
                "strata": {"experiment": "update_sensitive", "question_type": "Fact Retrieval"},
            }
            for i in range(3)
        ]
        write_jsonl(self.data_dir / "documents.jsonl", initial + added)
        write_jsonl(self.data_dir / "update_initial_documents.jsonl", initial)
        write_jsonl(self.data_dir / "update_added_documents.jsonl", added)
        write_jsonl(self.data_dir / "update_set.jsonl", questions)
        write_json(self.data_dir / "update_split.json", {"dataset": "graphrag_bench_medical", "initial_documents": 5, "added_documents": 1, "seed": 1})
        self.config = Config(
            path=self.root / "config" / "experiment.toml",
            root=self.root,
            data={
                "study": {"data_root": "data", "output_root": "results", "protocol_version": "test"},
                "common": {"concurrent_workers": 16},
                "provider": {},
                "datasets": {"graphrag_bench_medical": {"pilot_size": 3}},
                "systems": {"graphrag": {"kind": "command"}, "lightrag": {"kind": "command"}},
            },
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_medical_update_uses_clean_graph_workspace_and_incremental_light_workspace(self):
        calls: list[dict[str, object]] = []

        def factory(config, system, workspace):
            return FakeAdapter(system, workspace, calls)

        with patch("ragstudy.runner.build_adapter", side_effect=factory):
            summary = run_update(
                self.config, ["graphrag", "lightrag"],
                datasets=["graphrag_bench_medical"], index_retries=0,
            )

        self.assertTrue(summary["complete"])
        self.assertEqual(summary["reports"], 2)
        self.assertEqual(summary["query_traces"], 12)
        self.assertEqual(summary["successful_query_traces"], 12)
        self.assertEqual(summary["expected_query_traces"], 12)

        graph_calls = [row for row in calls if row["system"] == "graphrag"]
        light_calls = [row for row in calls if row["system"] == "lightrag"]
        self.assertEqual(len(graph_calls), 2)
        self.assertNotEqual(graph_calls[0]["workspace"], graph_calls[1]["workspace"])
        self.assertEqual(len(graph_calls[0]["document_ids"]), 5)
        self.assertEqual(len(graph_calls[1]["document_ids"]), 6)
        self.assertFalse(graph_calls[1]["incremental"])
        self.assertEqual(len(light_calls), 2)
        self.assertEqual(light_calls[0]["workspace"], light_calls[1]["workspace"])
        self.assertEqual(len(light_calls[0]["document_ids"]), 5)
        self.assertEqual(light_calls[1]["document_ids"], ["d5"])
        self.assertTrue(light_calls[1]["incremental"])

        traces = read_jsonl(self.config.output_root / "update_query_traces.jsonl")
        identities = {(row["system"], row["question_id"], row["metadata"]["stage"]) for row in traces}
        self.assertEqual(len(identities), 12)
        reports = read_jsonl(self.config.output_root / "update_runs.jsonl")
        self.assertEqual({row["update_mode"] for row in reports}, {"full_reindex", "incremental_insert"})
        self.assertTrue(all("initial_index_seconds" in row and "update_seconds" in row for row in reports))

    def test_update_questions_must_point_to_added_partition(self):
        rows = read_jsonl(self.data_dir / "update_set.jsonl")
        rows[0]["evidence_document_ids"] = ["d0"]
        write_jsonl(self.data_dir / "update_set.jsonl", rows)
        with patch("ragstudy.runner.build_adapter"):
            with self.assertRaisesRegex(RuntimeError, "evidence must belong only to the added partition"):
                run_update(self.config, ["graphrag"], datasets=["graphrag_bench_medical"])

    def test_resume_exact_update_run_reuses_completed_initial_index_and_pre_traces(self):
        run_id = "update-existing"
        run_root = self.config.output_root / "runs" / run_id
        write_json(run_root / "run_manifest.json", {
            "run_id": run_id,
            "phase": "update",
            "status": "running",
            "systems": ["graphrag"],
            "datasets": ["graphrag_bench_medical"],
        })
        initial_workspace = run_root / "graphrag_bench_medical" / "graphrag" / "initial"
        write_json(initial_workspace / "index_command_report.json", {
            "return_code": 0,
            "detail": {"documents": 5},
            "document_count": 5,
            "incremental": False,
        })
        questions = read_jsonl(self.data_dir / "update_set.jsonl")
        write_jsonl(self.config.output_root / "update_query_traces.jsonl", [
            {
                "run_id": run_id,
                "dataset": "graphrag_bench_medical",
                "system": "graphrag",
                "question_id": row["id"],
                "status": "ok",
                "metadata": {"stage": "pre_update"},
            }
            for row in questions
        ])
        calls: list[dict[str, object]] = []

        def factory(config, system, workspace):
            return FakeAdapter(system, workspace, calls)

        with patch("ragstudy.runner.build_adapter", side_effect=factory):
            summary = run_update(
                self.config,
                ["graphrag"],
                datasets=["graphrag_bench_medical"],
                index_retries=0,
                resume_run_id=run_id,
            )

        self.assertTrue(summary["complete"])
        self.assertEqual(summary["query_traces"], 6)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["document_ids"]), 6)
        traces = [
            row for row in read_jsonl(self.config.output_root / "update_query_traces.jsonl")
            if row.get("run_id") == run_id
        ]
        identities = {(row["question_id"], row["metadata"]["stage"]) for row in traces}
        self.assertEqual(len(identities), 6)


if __name__ == "__main__":
    unittest.main()
