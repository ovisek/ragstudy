from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragstudy.adapters import CommandAdapter
from ragstudy.config import Config
from ragstudy.io import read_json, read_jsonl, write_jsonl
from ragstudy.models import Document, Usage
from ragstudy.runner import run


class CommandAdapterCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.workspace = self.root / "workspace"
        self.bridge = self.root / "fake_bridge.py"
        self.bridge.write_text(
            """
import argparse, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
row = json.loads(Path(args.input).read_text().splitlines()[0])
print(f\"processing {row['id']}\", flush=True)
print(os.environ['TEST_RAGSTUDY_KEY'], file=sys.stderr, flush=True)
marker = Path(args.workspace) / f\"{row['id']}.failed-once\"
if row['id'] == 'd2' and not marker.exists():
    marker.write_text('failed')
    raise SystemExit(7)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'document': row['id']}))
""".strip()
            + "\n",
            encoding="utf-8",
        )
        self.config = Config(
            path=self.root / "config.toml",
            root=self.root,
            data={
                "provider": {"api_key_env": "TEST_RAGSTUDY_KEY"},
                "systems": {
                    "lightrag": {
                        "kind": "command",
                        "index_checkpoint": "document",
                        "index_command": f"{sys.executable} {self.bridge} --workspace {{workspace}} --input {{input}} --output {{output}}",
                        "query_command": "unused",
                    }
                },
            },
        )
        self.previous_key = os.environ.get("TEST_RAGSTUDY_KEY")
        os.environ["TEST_RAGSTUDY_KEY"] = "secret-value"

    def tearDown(self) -> None:
        if self.previous_key is None:
            os.environ.pop("TEST_RAGSTUDY_KEY", None)
        else:
            os.environ["TEST_RAGSTUDY_KEY"] = self.previous_key
        shutil.rmtree(self.root)

    def test_streams_redacted_output_retries_and_resumes_by_document(self) -> None:
        documents = [Document("d1", "one"), Document("d2", "two")]
        stdout, stderr = io.StringIO(), io.StringIO()
        adapter = CommandAdapter(self.config, "lightrag", self.workspace)

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), patch("ragstudy.adapters.time.sleep"):
            report = adapter.index(documents, retries=1)

        self.assertTrue(report["complete"])
        self.assertIn("processing d1", stdout.getvalue())
        self.assertIn("processing d2", stdout.getvalue())
        self.assertNotIn("secret-value", stderr.getvalue())
        self.assertIn("[REDACTED]", stderr.getvalue())
        terminal = [row for row in read_jsonl(self.workspace / "index_progress.jsonl") if row["status"] != "running"]
        self.assertEqual([row["status"] for row in terminal], ["ok", "failed", "ok"])
        for log in (self.workspace / "logs" / "index").glob("**/*.log"):
            self.assertNotIn("secret-value", log.read_text(encoding="utf-8"))

        before = len(read_jsonl(self.workspace / "index_progress.jsonl"))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            resumed = adapter.index(documents, resume=True, retries=0)
        self.assertTrue(resumed["complete"])
        self.assertEqual(len(read_jsonl(self.workspace / "index_progress.jsonl")), before)

    def test_final_document_failure_does_not_prevent_later_documents(self) -> None:
        documents = [Document("d1", "one"), Document("d2", "two"), Document("d3", "three")]
        adapter = CommandAdapter(self.config, "lightrag", self.workspace)
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            report = adapter.index(documents, retries=0)

        self.assertFalse(report["complete"])
        self.assertEqual(report["failed_units"], ["d2"])
        self.assertIn("processing d3", stdout.getvalue())

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            resumed = adapter.index(documents, resume=True, retries=0)
        self.assertTrue(resumed["complete"])


class FakeAdapter:
    complete = False
    query_calls = 0

    def index(self, documents, incremental=False, resume=False, retries=2):
        return {"complete": self.complete, "failed_units": [] if self.complete else [documents[0].id]}

    def query(self, question, method):
        type(self).query_calls += 1
        return "answer", [], Usage(), {"fake": True}


class RunResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        prepared = self.root / "data" / "prepared" / "dataset"
        prepared.mkdir(parents=True)
        write_jsonl(prepared / "documents.jsonl", [{"id": "d1", "text": "one"}])
        write_jsonl(prepared / "pilot_set.jsonl", [{"id": "q1", "question": "one?"}])
        self.config = Config(
            path=self.root / "config" / "experiment.toml",
            root=self.root,
            data={
                "study": {"output_root": "results", "data_root": "data"},
                "common": {},
                "provider": {},
                "datasets": {"dataset": {}},
                "systems": {"lightrag": {"kind": "command", "index_command": "unused", "query_command": "unused"}},
            },
        )
        FakeAdapter.complete = False
        FakeAdapter.query_calls = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_resume_reuses_latest_run_and_retries_queries(self) -> None:
        with patch("ragstudy.runner.build_adapter", return_value=FakeAdapter()):
            first = run(self.config, "pilot", ["lightrag"], index_retries=0)
        self.assertFalse(first["complete"])
        self.assertEqual(FakeAdapter.query_calls, 0)

        FakeAdapter.complete = True
        with patch("ragstudy.runner.build_adapter", return_value=FakeAdapter()):
            resumed = run(self.config, "pilot", ["lightrag"], resume=True, index_retries=0)
        self.assertEqual(resumed["run_id"], first["run_id"])
        self.assertTrue(resumed["complete"])
        self.assertEqual(FakeAdapter.query_calls, 1)
        manifest = read_json(self.config.output_root / "runs" / first["run_id"] / "run_manifest.json")
        self.assertEqual(manifest["status"], "complete")

        with patch("ragstudy.runner.build_adapter", return_value=FakeAdapter()):
            fresh = run(self.config, "pilot", ["lightrag"], index_retries=0)
        self.assertNotEqual(fresh["run_id"], first["run_id"])
        self.assertEqual(FakeAdapter.query_calls, 2)

    def test_resume_refuses_changed_inputs(self) -> None:
        with patch("ragstudy.runner.build_adapter", return_value=FakeAdapter()):
            run(self.config, "pilot", ["lightrag"], index_retries=0)
        write_jsonl(self.config.data_root / "prepared" / "dataset" / "documents.jsonl", [{"id": "d1", "text": "changed"}])

        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            run(self.config, "pilot", ["lightrag"], resume=True, index_retries=0)

    def test_partial_index_requires_override_and_marks_trace(self) -> None:
        with patch("ragstudy.runner.build_adapter", return_value=FakeAdapter()):
            result = run(self.config, "pilot", ["lightrag"], allow_partial_index=True, index_retries=0)
        self.assertFalse(result["complete"])
        self.assertEqual(FakeAdapter.query_calls, 1)
        traces = read_jsonl(self.config.output_root / "query_traces.jsonl")
        self.assertFalse(traces[-1]["metadata"]["index_complete"])
        self.assertEqual(traces[-1]["metadata"]["failed_index_units"], ["d1"])


if __name__ == "__main__":
    unittest.main()
