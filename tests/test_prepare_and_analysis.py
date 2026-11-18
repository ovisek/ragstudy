from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ragstudy.analysis import analyse
from ragstudy.config import load_config
from ragstudy.evaluation import export_evaluation_inputs, import_metrics, score_retrieval_contract
from ragstudy.io import append_jsonl, write_jsonl
from ragstudy.prepare import make_update_split, prepare_dataset


class ExperimentHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        (self.root / "config").mkdir()
        source = Path(__file__).parents[1] / "config" / "experiment.example.toml"
        content = source.read_text(encoding="utf-8").replace('output_root = "results"', f'output_root = "{self.root / "results"}"').replace('data_root = "data"', f'data_root = "{self.root / "data"}"')
        (self.root / "config" / "experiment.toml").write_text(content, encoding="utf-8")
        self.config = load_config(self.root / "config" / "experiment.toml")

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_prepare_deduplicates_and_is_deterministic(self) -> None:
        docs = self.root / "docs.jsonl"
        questions = self.root / "questions.jsonl"
        write_jsonl(docs, [
            {"id": "d1", "text": "Alpha beta gamma", "title": "A"},
            {"id": "d2", "text": "Alpha beta gamma", "title": "Duplicate"},
            {"id": "d3", "text": "Delta epsilon", "title": "D"},
        ])
        write_jsonl(questions, [
            {"id": "q1", "question": "Alpha?", "reference_answer": "Alpha", "evidence": ["Alpha beta"], "strata": {"type": "a"}},
            {"id": "q2", "question": "Delta?", "reference_answer": "Delta", "evidence": ["Delta"], "strata": {"type": "b"}},
        ])
        report = prepare_dataset(self.config, "ultradomain_agriculture", docs, questions)
        split = make_update_split(self.config, "ultradomain_agriculture")
        self.assertEqual(report["document_count_deduplicated"], 2)
        self.assertEqual(report["evaluation_count"], 2)
        self.assertEqual(split["initial_documents"] + split["added_documents"], 2)

    def test_score_and_analyse(self) -> None:
        dataset_dir = self.config.data_root / "prepared" / "ultradomain_agriculture"
        dataset_dir.mkdir(parents=True)
        write_jsonl(dataset_dir / "evaluation_set.jsonl", [{"id": "q1", "question": "Alpha?", "reference_answer": "Alpha", "evidence": ["Alpha beta"], "strata": {}}])
        trace = {"run_id": "r1", "phase": "full", "dataset": "ultradomain_agriculture", "system": "naive", "question_id": "q1", "question": "Alpha?", "answer": "Alpha", "contexts": [{"text": "Alpha beta gamma"}], "elapsed_seconds": 1.0, "usage": {"api_calls": 2}, "status": "ok", "repetition": 1, "metadata": {"estimated_generator_cost_usd": 0.01}}
        append_jsonl(self.config.output_root / "query_traces.jsonl", trace)
        counts = export_evaluation_inputs(self.config, self.config.output_root / "query_traces.jsonl")
        score_retrieval_contract(self.config, self.config.output_root / "query_traces.jsonl")
        result = analyse(self.config)
        self.assertEqual(counts["ragas_rows"], 1)
        self.assertGreater(result["summary_rows"], 0)
        self.assertTrue((self.config.output_root / "analysis" / "summary.csv").exists())

    def test_metric_import_rejects_duplicate_identity(self) -> None:
        source = self.root / "metrics.jsonl"
        row = {
            "run_id": "r1", "dataset": "ultradomain_agriculture", "system": "naive",
            "question_id": "q1", "repetition": 1, "metric": "answer_relevancy",
            "value": 0.5, "backend": "ragas", "coverage": "available",
            "evaluator_model": "judge",
        }
        write_jsonl(source, [row])
        self.assertEqual(import_metrics(self.config, source, "ragas"), 1)
        with self.assertRaisesRegex(ValueError, "Duplicate metric identity"):
            import_metrics(self.config, source, "ragas")

    def test_metric_import_accepts_explicit_unavailable_value(self) -> None:
        source = self.root / "unavailable.jsonl"
        write_jsonl(source, [{
            "run_id": "r1", "dataset": "ultradomain_agriculture", "system": "graphrag",
            "question_id": "q1", "metric": "faithfulness", "value": None,
            "backend": "ragas", "coverage": "unavailable", "reason": "contexts_unavailable",
            "evaluator_model": "judge",
        }])
        self.assertEqual(import_metrics(self.config, source, "ragas"), 1)
