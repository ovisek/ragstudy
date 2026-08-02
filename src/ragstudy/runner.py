from __future__ import annotations

import json
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from .adapters import build_adapter
from .config import Config
from .io import append_jsonl, read_json, read_jsonl, sha256_file, sha256_text, write_json
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


def _already_done(path: Path, run_id: str) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    return {
        (row["dataset"], row["system"], row["question_id"], int(row.get("repetition", 1)))
        for row in read_jsonl(path)
        if row.get("run_id") == run_id and row.get("status") == "ok" and row.get("metadata", {}).get("index_complete", True)
    }


def _run_spec(config: Config, phase: str, systems: list[str], repetition: int) -> dict[str, object]:
    inputs: dict[str, dict[str, str]] = {}
    question_file = "pilot_set.jsonl" if phase == "pilot" else "evaluation_set.jsonl"
    for dataset in config.data["datasets"]:
        prepared = config.data_root / "prepared" / dataset
        inputs[dataset] = {
            "documents": sha256_file(prepared / "documents.jsonl"),
            "questions": sha256_file(prepared / question_file),
        }
    relevant_config = {
        "common": config.data["common"],
        "provider": config.data["provider"],
        "datasets": {name: config.data["datasets"][name] for name in config.data["datasets"]},
        "systems": {name: config.data["systems"][name] for name in systems},
    }
    return {"phase": phase, "systems": systems, "repetition": repetition, "inputs": inputs, "config": relevant_config}


def _spec_fingerprint(spec: dict[str, object]) -> str:
    compatible = deepcopy(spec)
    config = compatible.get("config", {})
    if isinstance(config, dict):
        common = config.get("common", {})
        if isinstance(common, dict):
            common.pop("concurrent_workers", None)
        provider = config.get("provider", {})
        if isinstance(provider, dict):
            provider.pop("requests_per_minute", None)
            selected_systems = compatible.get("systems", [])
            if isinstance(selected_systems, list) and "lightrag" not in selected_systems:
                provider.pop("embedding_dimension", None)
        configured_systems = config.get("systems", {})
        if isinstance(configured_systems, dict):
            # Ignore legacy per-system operational concurrency fields too.
            lightrag = configured_systems.get("lightrag", {})
            if isinstance(lightrag, dict):
                lightrag.pop("chunk_llm_concurrency", None)
            graphrag = configured_systems.get("graphrag", {})
            if isinstance(graphrag, dict):
                graphrag.pop("concurrent_requests", None)
    return sha256_text(json.dumps(compatible, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _resume_run(config: Config, spec: dict[str, object]) -> tuple[str, Path, dict[str, object]]:
    runs_root = config.output_root / "runs"
    candidates: list[tuple[str, Path, dict[str, object]]] = []
    if runs_root.exists():
        for manifest_path in runs_root.glob("*/run_manifest.json"):
            manifest = read_json(manifest_path)
            if manifest.get("status") == "complete":
                continue
            if manifest.get("phase") != spec["phase"] or manifest.get("systems") != spec["systems"] or int(manifest.get("repetition", 1)) != spec["repetition"]:
                continue
            candidates.append((str(manifest.get("started_at", "")), manifest_path, manifest))
    if not candidates:
        raise RuntimeError("No incomplete run matches the requested phase, systems, and repetition")
    _, manifest_path, manifest = max(candidates, key=lambda item: item[0])
    expected = _spec_fingerprint(spec)
    previous_spec = manifest.get("spec")
    previous_matches = isinstance(previous_spec, dict) and _spec_fingerprint(previous_spec) == expected
    if not previous_matches and manifest.get("compatibility_fingerprint") != expected:
        raise RuntimeError(f"Latest incomplete run {manifest.get('run_id')} is incompatible because its configuration or prepared inputs changed; start a fresh run")
    return str(manifest["run_id"]), manifest_path.parent, manifest


def _cost(config: Config, usage: Usage, kind: str) -> float:
    prices = config.data["provider"]
    prefix = "embedding" if kind == "embedding" else ("judge" if kind == "judge" else "generator")
    return ((usage.input_tokens * float(prices.get(f"{prefix}_input_per_million", 0))) + (usage.output_tokens * float(prices.get(f"{prefix}_output_per_million", 0)))) / 1_000_000


def run(config: Config, phase: str, systems: list[str], repetition: int = 1, *, resume: bool = False, allow_partial_index: bool = False, index_retries: int = 2) -> dict[str, object]:
    if phase not in {"pilot", "full"}:
        raise ValueError("phase must be pilot or full")
    if index_retries < 0:
        raise ValueError("index_retries must be zero or greater")
    spec = _run_spec(config, phase, systems, repetition)
    now = datetime.now(timezone.utc).isoformat()
    if resume:
        run_id, run_root, manifest = _resume_run(config, spec)
        manifest.update({"status": "running", "resumed_at": now, "compatibility_fingerprint": _spec_fingerprint(spec), "index_retries": index_retries, "allow_partial_index": allow_partial_index})
        manifest.setdefault("resume_specs", []).append({"resumed_at": now, "spec": spec})
        print(f"Resuming run {run_id}", flush=True)
    else:
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        run_root = config.output_root / "runs" / run_id
        manifest = {
            "run_id": run_id, "started_at": now, "phase": phase, "systems": systems, "repetition": repetition,
            "status": "running", "compatibility_fingerprint": _spec_fingerprint(spec), "spec": spec,
            "index_retries": index_retries, "allow_partial_index": allow_partial_index,
        }
        print(f"Starting run {run_id}", flush=True)
    manifest_path = run_root / "run_manifest.json"
    write_json(manifest_path, manifest)
    trace_path = config.output_root / "query_traces.jsonl"
    completed = _already_done(trace_path, run_id)
    count = {"indexed": 0, "indexes_incomplete": 0, "queries_ok": 0, "queries_failed": 0, "queries_skipped": 0}
    try:
        for dataset in config.data["datasets"]:
            input_dir = config.data_root / "prepared" / dataset
            documents = _documents(input_dir / "documents.jsonl")
            questions = _questions(input_dir / ("pilot_set.jsonl" if phase == "pilot" else "evaluation_set.jsonl"))
            for system in systems:
                workspace = run_root / dataset / system
                print(f"Indexing {dataset} with {system}", flush=True)
                started = monotonic()
                adapter = None
                index_status = "ok"
                index_error = None
                index_report: dict[str, object] = {}
                try:
                    adapter = build_adapter(config, system, workspace)
                    index_report = adapter.index(documents, resume=resume, retries=index_retries)
                    index_complete = bool(index_report.get("complete", True))
                    if index_complete:
                        count["indexed"] += 1
                    else:
                        index_status = "incomplete"
                        count["indexes_incomplete"] += 1
                except Exception as exc:
                    index_complete = False
                    index_status, index_error = "failed", str(exc)
                    index_report = {"complete": False, "failed_units": ["adapter_error"]}
                    count["indexes_incomplete"] += 1
                write_json(workspace / "index_run.json", {"run_id": run_id, "phase": phase, "dataset": dataset, "system": system, "status": index_status, "error": index_error, "elapsed_seconds": monotonic() - started, "report": index_report})
                if not index_complete and (not allow_partial_index or adapter is None):
                    print(f"Skipping {dataset}/{system} queries because its index is incomplete", file=sys.stderr, flush=True)
                    continue
                failed_units = list(index_report.get("failed_units", []))
                for position, question in enumerate(questions, 1):
                    key = (dataset, system, question.id, repetition)
                    if key in completed:
                        count["queries_skipped"] += 1
                        continue
                    print(f"[{position}/{len(questions)}] {system}: querying {question.id}", flush=True)
                    started_at = datetime.now(timezone.utc).isoformat()
                    started = monotonic()
                    try:
                        answer, contexts, usage, metadata = adapter.query(question, method_for(question))
                        trace = Trace(run_id, phase, dataset, system, question.id, question.question, answer, contexts, started_at, monotonic() - started, usage, "ok", repetition=repetition, metadata={**metadata, "index_complete": index_complete, "failed_index_units": failed_units, "estimated_generator_cost_usd": _cost(config, usage, "generator")})
                        count["queries_ok"] += 1
                    except KeyboardInterrupt:
                        trace = Trace(run_id, phase, dataset, system, question.id, question.question, None, [], started_at, monotonic() - started, Usage(), "interrupted", error="Interrupted by user", repetition=repetition, metadata={"index_complete": index_complete, "failed_index_units": failed_units})
                        append_jsonl(trace_path, trace.record())
                        raise
                    except Exception as exc:
                        trace = Trace(run_id, phase, dataset, system, question.id, question.question, None, [], started_at, monotonic() - started, Usage(), "failed", error=str(exc), repetition=repetition, metadata={"index_complete": index_complete, "failed_index_units": failed_units})
                        count["queries_failed"] += 1
                    append_jsonl(trace_path, trace.record())
    except KeyboardInterrupt:
        manifest.update({"status": "interrupted", "interrupted_at": datetime.now(timezone.utc).isoformat(), "counts": count})
        write_json(manifest_path, manifest)
        raise
    complete = count["indexes_incomplete"] == 0 and count["queries_failed"] == 0
    summary = {"run_id": run_id, "phase": phase, "systems": systems, **count, "complete": complete}
    manifest.update({"status": "complete" if complete else "incomplete", "finished_at": datetime.now(timezone.utc).isoformat(), "counts": count})
    write_json(manifest_path, manifest)
    write_json(run_root / "run_summary.json", summary)
    write_json(config.output_root / "run_summary.json", summary)
    return summary


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
                initial_report = adapter.index(initial)
                if not initial_report.get("complete", True):
                    raise RuntimeError(f"{system} initial index incomplete: {initial_report.get('failed_units', [])}")
                if system == "graphrag":
                    update_report = adapter.index(all_docs, incremental=False)
                    update_mode = "full_reindex"
                else:
                    update_report = adapter.index(added, incremental=True)
                    update_mode = "incremental_insert"
                if not update_report.get("complete", True):
                    raise RuntimeError(f"{system} update index incomplete: {update_report.get('failed_units', [])}")
                status, error = "ok", None
            except Exception as exc:
                update_report, update_mode, status, error = {}, "failed", "failed", str(exc)
            report = {"run_id": run_id, "dataset": dataset, "system": system, "status": status, "error": error, "update_mode": update_mode, "initial_documents": len(initial), "added_documents": len(added), "elapsed_seconds": monotonic() - started, "report": update_report}
            reports.append(report)
            append_jsonl(config.output_root / "update_runs.jsonl", report)
    return reports
