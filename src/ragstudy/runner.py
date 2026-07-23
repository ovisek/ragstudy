from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Iterable

from .adapters import build_adapter
from .config import Config
from .io import append_jsonl, read_jsonl, write_json
from .models import Document, Question, Trace, Usage


def _documents(path: Path) -> list[Document]:
    return [Document(id=row["id"], text=row["text"], title=row.get("title"), metadata=row.get("metadata", {})) for row in read_jsonl(path)]


def _questions(path: Path) -> list[Question]:
    return [Question(id=row["id"], question=row["question"], reference_answer=row.get("reference_answer"), evidence=row.get("evidence", []), strata=row.get("strata", {})) for row in read_jsonl(path)]


def method_for(question: Question) -> str:
    kind = " ".join(question.strata.values()).lower()
    if any(term in kind for term in ("global", "thematic", "summary", "sensemaking")):
        return "global"
    if any(term in kind for term in ("multi", "hop", "reasoning")):
        return "hybrid"
    return "local"


def _already_done(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    return {(row["dataset"], row["system"], row["question_id"], int(row.get("repetition", 1))) for row in read_jsonl(path) if row.get("status") == "ok"}


def _cost(config: Config, usage: Usage, kind: str) -> float:
    prices = config.data["provider"]
    prefix = "embedding" if kind == "embedding" else ("judge" if kind == "judge" else "generator")
    return ((usage.input_tokens * float(prices.get(f"{prefix}_input_per_million", 0))) + (usage.output_tokens * float(prices.get(f"{prefix}_output_per_million", 0)))) / 1_000_000


def run(config: Config, phase: str, systems: list[str], repetition: int = 1) -> dict[str, int]:
    if phase not in {"pilot", "full"}:
        raise ValueError("phase must be pilot or full")
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    trace_path = config.output_root / "query_traces.jsonl"
    completed = _already_done(trace_path)
    count = {"indexed": 0, "queries_ok": 0, "queries_failed": 0, "queries_skipped": 0}
    for dataset in config.data["datasets"]:
        input_dir = config.data_root / "prepared" / dataset
        documents = _documents(input_dir / "documents.jsonl")
        questions = _questions(input_dir / ("pilot_set.jsonl" if phase == "pilot" else "evaluation_set.jsonl"))
        for system in systems:
            workspace = config.output_root / "runs" / run_id / dataset / system
            adapter = build_adapter(config, system, workspace)
            started = monotonic()
            index_status = "ok"
            index_error = None
            index_report = {}
            try:
                index_report = adapter.index(documents)
                count["indexed"] += 1
            except Exception as exc:  # retained as study evidence
                index_status, index_error = "failed", str(exc)
            write_json(workspace / "index_run.json", {"run_id": run_id, "phase": phase, "dataset": dataset, "system": system, "status": index_status, "error": index_error, "elapsed_seconds": monotonic() - started, "report": index_report})
            if index_status != "ok":
                continue
            for question in questions:
                key = (dataset, system, question.id, repetition)
                if key in completed:
                    count["queries_skipped"] += 1
                    continue
                started_at = datetime.now(timezone.utc).isoformat()
                started = monotonic()
                try:
                    answer, contexts, usage, metadata = adapter.query(question, method_for(question))
                    trace = Trace(run_id, phase, dataset, system, question.id, question.question, answer, contexts, started_at, monotonic() - started, usage, "ok", repetition=repetition, metadata={**metadata, "estimated_generator_cost_usd": _cost(config, usage, "generator")})
                    count["queries_ok"] += 1
                except Exception as exc:
                    trace = Trace(run_id, phase, dataset, system, question.id, question.question, None, [], started_at, monotonic() - started, Usage(), "failed", error=str(exc), repetition=repetition)
                    count["queries_failed"] += 1
                append_jsonl(trace_path, trace.record())
    write_json(config.output_root / "run_summary.json", {"run_id": run_id, "phase": phase, "systems": systems, **count})
    return count


def run_update(config: Config, systems: list[str]) -> list[dict[str, object]]:
    """Record equivalent 90/10 update workflows; GraphRAG is intentionally re-indexed."""
    reports: list[dict[str, object]] = []
    run_id = f"update-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    for dataset in config.data["datasets"]:
        data_dir = config.data_root / "prepared" / dataset
        initial = _documents(data_dir / "update_initial_documents.jsonl")
        added = _documents(data_dir / "update_added_documents.jsonl")
        all_docs = initial + added
        for system in systems:
            workspace = config.output_root / "runs" / run_id / dataset / system
            adapter = build_adapter(config, system, workspace)
            started = monotonic()
            try:
                adapter.index(initial)
                if system == "graphrag":
                    update_report = adapter.index(all_docs, incremental=False)
                    update_mode = "full_reindex"
                else:
                    update_report = adapter.index(added, incremental=True)
                    update_mode = "incremental_insert"
                status, error = "ok", None
            except Exception as exc:
                update_report, update_mode, status, error = {}, "failed", "failed", str(exc)
            report = {"run_id": run_id, "dataset": dataset, "system": system, "status": status, "error": error, "update_mode": update_mode, "initial_documents": len(initial), "added_documents": len(added), "elapsed_seconds": monotonic() - started, "report": update_report}
            reports.append(report)
            append_jsonl(config.output_root / "update_runs.jsonl", report)
    return reports

