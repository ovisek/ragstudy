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
from .usage_tracking import summarize_usage, usage_cost_usd


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
    return {
        "phase": phase,
        "systems": systems,
        "repetition": repetition,
        "protocol_version": config.data["study"].get("protocol_version"),
        "inputs": inputs,
        "config": relevant_config,
    }


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


def _run_usage(run_root: Path, config: Config) -> dict[str, object]:
    events: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in run_root.rglob("usage_events.jsonl"):
        for event in read_jsonl(path):
            event_id = str(event.get("event_id", ""))
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            events.append(event)
    summary = summarize_usage(events)
    summary["estimated_cost_usd"] = usage_cost_usd(summary, config.data["provider"])
    summary["event_count"] = len(events)
    return summary


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
                    setattr(adapter, "phase", phase)
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
                        query_cost = float(metadata.get("estimated_cost_usd", _cost(config, usage, "generator")))
                        trace = Trace(run_id, phase, dataset, system, question.id, question.question, answer, contexts, started_at, monotonic() - started, usage, "ok", repetition=repetition, metadata={**metadata, "index_complete": index_complete, "failed_index_units": failed_units, "estimated_query_cost_usd": query_cost})
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
        manifest.update({"status": "interrupted", "interrupted_at": datetime.now(timezone.utc).isoformat(), "counts": count, "usage": _run_usage(run_root, config)})
        write_json(manifest_path, manifest)
        raise
    complete = count["indexes_incomplete"] == 0 and count["queries_failed"] == 0
    usage = _run_usage(run_root, config)
    summary = {"run_id": run_id, "phase": phase, "systems": systems, **count, "complete": complete, "usage": usage}
    manifest.update({"status": "complete" if complete else "incomplete", "finished_at": datetime.now(timezone.utc).isoformat(), "counts": count, "usage": usage})
    write_json(manifest_path, manifest)
    write_json(run_root / "run_summary.json", summary)
    write_json(config.output_root / "run_summary.json", summary)
    return summary


def _workspace_usage_events(workspace: Path) -> list[dict[str, object]]:
    path = workspace / "usage_events.jsonl"
    return read_jsonl(path) if path.exists() else []


def _usage_since(workspace: Path, start: int, config: Config) -> dict[str, object]:
    summary = summarize_usage(_workspace_usage_events(workspace)[start:])
    summary["notional_cost_usd"] = usage_cost_usd(summary, config.data["provider"])
    summary["actual_api_expenditure_usd"] = 0.0
    return summary


def _completed_index_report(workspace: Path) -> dict[str, object] | None:
    path = workspace / "index_command_report.json"
    if not path.exists():
        return None
    report = read_json(path)
    if int(report.get("return_code", 1)) != 0:
        return None
    detail = report.get("detail", {})
    if not isinstance(detail, dict):
        return None
    return report


def _cumulative_index_seconds(workspace: Path) -> float:
    path = workspace / "index_command_reports.jsonl"
    if not path.exists():
        return 0.0
    return sum(float(row.get("elapsed_seconds", 0.0)) for row in read_jsonl(path))


def _update_trace(
    *, config: Config, run_id: str, dataset: str, system: str, stage: str,
    update_mode: str, question: Question, question_metadata: dict[str, object], adapter,
) -> dict[str, object]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = monotonic()
    try:
        answer, contexts, usage, metadata = adapter.query(question, method_for(question))
        trace = Trace(
            run_id, f"update_{stage}", dataset, system, question.id, question.question,
            answer, contexts, started_at, monotonic() - started, usage, "ok",
            metadata={
                **metadata,
                "stage": stage,
                "update_mode": update_mode,
                "reference_answer": question.reference_answer,
                "evidence": question.evidence,
                "evidence_document_ids": question_metadata.get("evidence_document_ids", []),
                "actual_api_expenditure_usd": 0.0,
                "notional_query_cost_usd": float(metadata.get("estimated_cost_usd", _cost(config, usage, "generator"))),
            },
        )
    except Exception as exc:
        trace = Trace(
            run_id, f"update_{stage}", dataset, system, question.id, question.question,
            None, [], started_at, monotonic() - started, Usage(), "failed", error=str(exc),
            metadata={
                "stage": stage,
                "update_mode": update_mode,
                "reference_answer": question.reference_answer,
                "evidence": question.evidence,
                "evidence_document_ids": question_metadata.get("evidence_document_ids", []),
                "actual_api_expenditure_usd": 0.0,
            },
        )
    record = trace.record()
    append_jsonl(config.output_root / "update_query_traces.jsonl", record)
    return record


def run_update(
    config: Config,
    systems: list[str],
    datasets: list[str] | None = None,
    *,
    index_retries: int = 2,
    resume_run_id: str | None = None,
) -> dict[str, object]:
    """Run an auditable initial-index/pre-query/update/post-query experiment."""
    selected = datasets or list(config.data["datasets"])
    unsupported = [system for system in systems if system not in {"graphrag", "lightrag"}]
    if unsupported:
        raise ValueError(f"RQ3 supports graphrag and lightrag only, not: {', '.join(unsupported)}")
    if index_retries < 0:
        raise ValueError("index_retries must be zero or greater")

    now = datetime.now(timezone.utc).isoformat()
    if resume_run_id:
        run_id = resume_run_id
        run_root = config.output_root / "runs" / run_id
        manifest_path = run_root / "run_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"RQ3 resume manifest not found: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("phase") != "update":
            raise RuntimeError(f"Run {run_id} is not an update experiment")
        previous_systems = list(manifest.get("systems", []))
        previous_datasets = list(manifest.get("datasets", []))
        if previous_systems != systems or previous_datasets != selected:
            raise RuntimeError(
                f"Run {run_id} was configured for systems={previous_systems}, datasets={previous_datasets}; "
                f"resume requested systems={systems}, datasets={selected}"
            )
        manifest.setdefault("resume_history", []).append({
            "resumed_at": now,
            "index_retries": index_retries,
            "controls": deepcopy(config.data["common"]),
            "provider": {
                key: value for key, value in config.data["provider"].items()
                if key != "api_key"
            },
        })
        manifest["status"] = "running"
        print(f"Resuming RQ3 run {run_id}", flush=True)
    else:
        run_id = f"update-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        run_root = config.output_root / "runs" / run_id
        manifest_path = run_root / "run_manifest.json"
        manifest = {
            "run_id": run_id,
            "phase": "update",
            "started_at": now,
            "status": "running",
            "systems": systems,
            "datasets": selected,
            "index_retries": index_retries,
            "controls": config.data["common"],
            "provider": {
                key: value for key, value in config.data["provider"].items()
                if key != "api_key"
            },
            "actual_api_expenditure_usd": 0.0,
        }
    write_json(manifest_path, manifest)
    reports: list[dict[str, object]] = []
    trace_path = config.output_root / "update_query_traces.jsonl"
    query_records = [
        row for row in read_jsonl(trace_path)
        if row.get("run_id") == run_id
    ] if trace_path.exists() else []
    completed_traces = {
        (str(row.get("dataset")), str(row.get("system")), str(row.get("question_id")), str(row.get("metadata", {}).get("stage")))
        for row in query_records
        if row.get("status") == "ok"
    }
    expected_query_traces = 0

    try:
        for dataset in selected:
            if dataset not in config.data["datasets"]:
                raise ValueError(f"Unknown configured dataset: {dataset}")
            data_dir = config.data_root / "prepared" / dataset
            required = [
                data_dir / "update_initial_documents.jsonl",
                data_dir / "update_added_documents.jsonl",
                data_dir / "update_split.json",
                data_dir / "update_set.jsonl",
                data_dir / "documents.jsonl",
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise RuntimeError("RQ3 input files missing: " + ", ".join(missing))
            initial = _documents(required[0])
            added = _documents(required[1])
            all_docs = initial + added
            raw_questions = read_jsonl(required[3])
            questions = _questions(required[3])
            question_metadata = {str(row["id"]): row for row in raw_questions}
            if not questions:
                raise RuntimeError(f"{dataset}: update_set.jsonl has no questions")
            if len(question_metadata) != len(questions):
                raise RuntimeError(f"{dataset}: update question IDs must be unique")
            configured_question_count = int(config.dataset(dataset).get("pilot_size", len(questions)))
            if len(questions) != configured_question_count:
                raise RuntimeError(f"{dataset}: expected {configured_question_count} update questions, found {len(questions)}")
            initial_ids = {document.id for document in initial}
            added_ids = {document.id for document in added}
            prepared_ids = {document.id for document in _documents(required[4])}
            if initial_ids & added_ids:
                raise RuntimeError(f"{dataset}: initial and added partitions overlap")
            if initial_ids | added_ids != prepared_ids:
                raise RuntimeError(f"{dataset}: update partitions do not reconstruct the prepared corpus")
            split = read_json(required[2])
            if int(split.get("initial_documents", -1)) != len(initial) or int(split.get("added_documents", -1)) != len(added):
                raise RuntimeError(f"{dataset}: update_split.json counts do not match the partition files")
            for row in raw_questions:
                evidence_ids = {str(value) for value in row.get("evidence_document_ids", [])}
                if not evidence_ids:
                    raise RuntimeError(f"{dataset}/{row.get('id')}: evidence_document_ids is required")
                if not evidence_ids <= added_ids:
                    raise RuntimeError(f"{dataset}/{row.get('id')}: evidence must belong only to the added partition")
            expected_query_traces += len(systems) * len(questions) * 2

            dataset_inputs = {
                "initial_documents_sha256": sha256_file(required[0]),
                "added_documents_sha256": sha256_file(required[1]),
                "update_split_sha256": sha256_file(required[2]),
                "update_questions_sha256": sha256_file(required[3]),
                "initial_documents": len(initial),
                "added_documents": len(added),
                "total_documents": len(all_docs),
                "realized_update_fraction": len(added) / len(all_docs),
                "questions": len(questions),
            }
            manifest.setdefault("inputs", {})[dataset] = dataset_inputs  # type: ignore[index]
            write_json(manifest_path, manifest)

            for system in systems:
                update_mode = "full_reindex" if system == "graphrag" else "incremental_insert"
                system_root = run_root / dataset / system
                initial_workspace = system_root / "initial"
                update_workspace = system_root / "full_reindex" if system == "graphrag" else initial_workspace
                report: dict[str, object] = {
                    "run_id": run_id,
                    "dataset": dataset,
                    "system": system,
                    "status": "running",
                    "error": None,
                    "update_mode": update_mode,
                    **dataset_inputs,
                    "initial_workspace": str(initial_workspace),
                    "update_workspace": str(update_workspace),
                    "actual_api_expenditure_usd": 0.0,
                }
                print(f"RQ3 {dataset}/{system}: indexing initial {len(initial)} document(s)", flush=True)
                try:
                    initial_adapter = build_adapter(config, system, initial_workspace)
                    setattr(initial_adapter, "phase", "update_initial")
                    completed_initial = _completed_index_report(initial_workspace) if resume_run_id else None
                    if completed_initial is not None:
                        initial_report = {
                            "complete": True,
                            "resumed_from_completed_index": True,
                            "source_report": str(initial_workspace / "index_command_report.json"),
                        }
                    else:
                        initial_report = initial_adapter.index(initial, incremental=False, retries=index_retries)
                    initial_seconds = _cumulative_index_seconds(initial_workspace)
                    initial_usage = _usage_since(initial_workspace, 0, config)
                    report.update({
                        "initial_index_seconds": initial_seconds,
                        "initial_index_usage": initial_usage,
                        "initial_report": initial_report,
                    })
                    if not initial_report.get("complete", True):
                        raise RuntimeError(f"{system} initial index incomplete: {initial_report.get('failed_units', [])}")

                    for position, question in enumerate(questions, 1):
                        trace_key = (dataset, system, question.id, "pre_update")
                        if trace_key in completed_traces:
                            print(f"RQ3 {dataset}/{system}: pre-update query already complete {question.id}", flush=True)
                            continue
                        print(f"RQ3 {dataset}/{system}: pre-update query {position}/{len(questions)} {question.id}", flush=True)
                        record = _update_trace(
                            config=config, run_id=run_id, dataset=dataset, system=system,
                            stage="pre_update", update_mode=update_mode, question=question,
                            question_metadata=question_metadata[question.id], adapter=initial_adapter,
                        )
                        query_records.append(record)
                        if record.get("status") == "ok":
                            completed_traces.add(trace_key)

                    if system == "graphrag":
                        if update_workspace.exists() and not resume_run_id:
                            raise RuntimeError(f"GraphRAG full-reindex workspace is not clean: {update_workspace}")
                        update_adapter = build_adapter(config, system, update_workspace)
                        update_documents, incremental = all_docs, False
                    else:
                        update_adapter = initial_adapter
                        update_documents, incremental = added, True
                    setattr(update_adapter, "phase", "update_apply")
                    print(f"RQ3 {dataset}/{system}: applying {update_mode} to {len(update_documents)} document(s)", flush=True)
                    completed_update = _completed_index_report(update_workspace) if resume_run_id else None
                    update_already_complete = (
                        completed_update is not None
                        and bool(completed_update.get("incremental", False)) == incremental
                        and int(completed_update.get("document_count", -1)) == len(update_documents)
                    )
                    if update_already_complete:
                        update_report = {
                            "complete": True,
                            "resumed_from_completed_index": True,
                            "source_report": str(update_workspace / "index_command_report.json"),
                        }
                    else:
                        update_report = update_adapter.index(update_documents, incremental=incremental, retries=index_retries)
                    update_seconds = _cumulative_index_seconds(update_workspace)
                    update_usage = _usage_since(update_workspace, 0, config)
                    report.update({
                        "update_seconds": update_seconds,
                        "update_usage": update_usage,
                        "update_report": update_report,
                    })
                    if not update_report.get("complete", True):
                        raise RuntimeError(f"{system} update index incomplete: {update_report.get('failed_units', [])}")

                    for position, question in enumerate(questions, 1):
                        trace_key = (dataset, system, question.id, "post_update")
                        if trace_key in completed_traces:
                            print(f"RQ3 {dataset}/{system}: post-update query already complete {question.id}", flush=True)
                            continue
                        print(f"RQ3 {dataset}/{system}: post-update query {position}/{len(questions)} {question.id}", flush=True)
                        record = _update_trace(
                            config=config, run_id=run_id, dataset=dataset, system=system,
                            stage="post_update", update_mode=update_mode, question=question,
                            question_metadata=question_metadata[question.id], adapter=update_adapter,
                        )
                        query_records.append(record)
                        if record.get("status") == "ok":
                            completed_traces.add(trace_key)
                    system_queries = [row for row in query_records if row["dataset"] == dataset and row["system"] == system]
                    observed_query_keys = {
                        (str(row.get("question_id")), str(row.get("metadata", {}).get("stage")))
                        for row in system_queries
                    }
                    successful_query_keys = {
                        (str(row.get("question_id")), str(row.get("metadata", {}).get("stage")))
                        for row in system_queries
                        if row.get("status") == "ok"
                    }
                    expected_query_keys = {
                        (question.id, stage)
                        for question in questions
                        for stage in ("pre_update", "post_update")
                    }
                    failed_queries = len(expected_query_keys - successful_query_keys)
                    report["query_coverage"] = {
                        "expected": len(questions) * 2,
                        "observed": len(observed_query_keys),
                        "successful": len(successful_query_keys),
                        "failed": failed_queries,
                    }
                    report["status"] = "ok" if failed_queries == 0 else "incomplete"
                except KeyboardInterrupt:
                    report.update({"status": "interrupted", "error": "Interrupted by user"})
                    reports.append(report)
                    append_jsonl(config.output_root / "update_runs.jsonl", report)
                    raise
                except Exception as exc:
                    report.update({"status": "failed", "error": str(exc)})
                reports.append(report)
                append_jsonl(config.output_root / "update_runs.jsonl", report)
    except KeyboardInterrupt:
        manifest.update({"status": "interrupted", "interrupted_at": datetime.now(timezone.utc).isoformat(), "reports": len(reports), "query_traces": len(query_records)})
        write_json(manifest_path, manifest)
        raise

    expected_reports = len(selected) * len(systems)
    complete = len(reports) == expected_reports and all(report.get("status") == "ok" for report in reports)
    summary = {
        "run_id": run_id,
        "phase": "update",
        "datasets": selected,
        "systems": systems,
        "reports": len(reports),
        "successful_reports": sum(report.get("status") == "ok" for report in reports),
        "query_traces": len(query_records),
        "successful_query_traces": sum(row.get("status") == "ok" for row in query_records),
        "expected_query_traces": expected_query_traces,
        "complete": complete,
        "actual_api_expenditure_usd": 0.0,
    }
    manifest.update({"status": "complete" if complete else "incomplete", "finished_at": datetime.now(timezone.utc).isoformat(), "summary": summary})
    write_json(manifest_path, manifest)
    write_json(run_root / "run_summary.json", summary)
    write_json(config.output_root / "update_run_summary.json", summary)
    return summary
