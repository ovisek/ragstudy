from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import Config
from .io import append_jsonl, read_jsonl, write_jsonl


def _normalise(value: str) -> str:
    return re.sub(r"\W+", " ", value.lower()).strip()


def _questions(config: Config) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in config.data["datasets"]:
        path = config.data_root / "prepared" / dataset / "evaluation_set.jsonl"
        if path.exists():
            output.update({(dataset, row["id"]): row for row in read_jsonl(path)})
    return output


def export_evaluation_inputs(config: Config, traces_path: Path) -> dict[str, int]:
    """Export stable RAGAS/RAGChecker input without allowing evaluators to mutate traces."""
    questions = _questions(config)
    ragas, ragchecker = [], []
    for trace in read_jsonl(traces_path):
        if trace.get("status") != "ok":
            continue
        source = questions.get((trace["dataset"], trace["question_id"]), {})
        contexts = [item.get("text", "") for item in trace.get("contexts", [])]
        common = {
            "trace_key": f"{trace['dataset']}::{trace['system']}::{trace['question_id']}::{trace.get('repetition', 1)}",
            "run_id": trace.get("run_id"),
            "dataset": trace["dataset"], "system": trace["system"], "question_id": trace["question_id"],
            "question": trace["question"], "answer": trace.get("answer", ""), "contexts": contexts,
            "reference_answer": source.get("reference_answer"), "evidence": source.get("evidence", []),
        }
        ragas.append(common)
        if common["evidence"]:
            ragchecker.append(common)
    write_jsonl(config.output_root / "evaluation" / "ragas_input.jsonl", ragas)
    write_jsonl(config.output_root / "evaluation" / "ragchecker_input.jsonl", ragchecker)
    return {"ragas_rows": len(ragas), "ragchecker_rows": len(ragchecker)}


def score_retrieval_contract(config: Config, traces_path: Path) -> dict[str, int]:
    """Compute transparent non-LLM retrieval checks; LLM judge metrics are imported separately."""
    questions = _questions(config)
    output = config.output_root / "metric_results.jsonl"
    count = 0
    for trace in read_jsonl(traces_path):
        if trace.get("status") != "ok":
            continue
        source = questions.get((trace["dataset"], trace["question_id"]), {})
        context = " ".join(item.get("text", "") for item in trace.get("contexts", []))
        evidence = [item for item in source.get("evidence", []) if isinstance(item, str) and item.strip()]
        if evidence:
            found = sum(_normalise(item) in _normalise(context) for item in evidence)
            append_jsonl(output, {"dataset": trace["dataset"], "system": trace["system"], "question_id": trace["question_id"], "repetition": trace.get("repetition", 1), "metric": "evidence_recall_proxy", "value": found / len(evidence), "backend": "deterministic", "coverage": "available"})
            count += 1
        reference = source.get("reference_answer")
        if isinstance(reference, str) and reference.strip() and trace.get("answer"):
            append_jsonl(output, {"dataset": trace["dataset"], "system": trace["system"], "question_id": trace["question_id"], "repetition": trace.get("repetition", 1), "metric": "answer_exact_match_proxy", "value": float(_normalise(reference) == _normalise(trace["answer"])), "backend": "deterministic", "coverage": "available"})
            count += 1
    return {"metric_rows": count}


def import_metrics(config: Config, path: Path, backend: str) -> int:
    """Import external evaluator rows without replacing the raw results.

    Required fields: dataset, system, question_id, metric, value. Optional:
    repetition, coverage, error, evaluator_model, evaluator_cost_usd.
    """
    rows = read_jsonl(path)
    required = {"dataset", "system", "question_id", "metric", "value"}
    output = config.output_root / "metric_results.jsonl"
    existing = read_jsonl(output) if output.exists() else []

    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("run_id"), row.get("dataset"), row.get("system"),
            row.get("question_id"), int(row.get("repetition", 1)),
            row.get("metric"), row.get("backend", backend),
            row.get("evaluator_model"),
        )

    seen = {identity(row) for row in existing}
    prepared: list[dict[str, Any]] = []
    for row in rows:
        absent = required - set(row)
        if absent:
            raise ValueError(f"Metric row misses {sorted(absent)}")
        row.setdefault("backend", backend)
        row.setdefault("coverage", "available")
        row.setdefault("repetition", 1)
        key = identity(row)
        if key in seen:
            raise ValueError(f"Duplicate metric identity: {key}")
        seen.add(key)
        prepared.append(row)
    for row in prepared:
        append_jsonl(output, row)
    return len(prepared)
