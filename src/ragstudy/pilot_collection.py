from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .config import Config
from .io import read_json, read_jsonl, sha256_file, sha256_text, write_json, write_jsonl
from .usage_tracking import summarize_usage, usage_cost_usd


REQUIRED_SYSTEMS = ("naive", "graphrag", "lightrag")


def parse_run_selections(values: list[str]) -> dict[str, str]:
    selections: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --run {value!r}; expected SYSTEM=RUN_ID")
        system, run_id = (part.strip() for part in value.split("=", 1))
        if not system or not run_id:
            raise ValueError(f"Invalid --run {value!r}; expected SYSTEM=RUN_ID")
        if system in selections:
            raise ValueError(f"Duplicate --run selection for {system}")
        selections[system] = run_id
    required, selected = set(REQUIRED_SYSTEMS), set(selections)
    if selected != required:
        missing = sorted(required - selected)
        extra = sorted(selected - required)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ValueError("Pilot collection requires exactly naive, graphrag, and lightrag (" + "; ".join(details) + ")")
    if len(set(selections.values())) != len(selections):
        raise ValueError("Each system must select a different run ID")
    return selections


def _canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _comparison_basis_from_manifest(manifest: dict[str, Any], dataset: str) -> dict[str, Any]:
    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        raise ValueError(f"Run {manifest.get('run_id')} has no auditable run spec")
    config = spec.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Run {manifest.get('run_id')} has no configuration snapshot")
    common = dict(config.get("common", {}))
    provider = dict(config.get("provider", {}))
    common.pop("concurrent_workers", None)
    provider.pop("requests_per_minute", None)
    datasets = config.get("datasets", {})
    inputs = spec.get("inputs", {})
    if dataset not in datasets or dataset not in inputs:
        raise ValueError(f"Run {manifest.get('run_id')} does not include dataset {dataset}")
    return {
        "phase": spec.get("phase"),
        "repetition": int(spec.get("repetition", manifest.get("repetition", 1))),
        "protocol_version": spec.get("protocol_version"),
        "common": common,
        "provider": provider,
        "dataset": datasets[dataset],
        "inputs": inputs[dataset],
    }


def _comparison_basis_from_current(config: Config, dataset: str, repetition: int) -> dict[str, Any]:
    prepared = config.data_root / "prepared" / dataset
    common = dict(config.data["common"])
    provider = dict(config.data["provider"])
    common.pop("concurrent_workers", None)
    provider.pop("requests_per_minute", None)
    return {
        "phase": "pilot",
        "repetition": repetition,
        "protocol_version": config.data["study"].get("protocol_version"),
        "common": common,
        "provider": provider,
        "dataset": config.dataset(dataset),
        "inputs": {
            "documents": sha256_file(prepared / "documents.jsonl"),
            "questions": sha256_file(prepared / "pilot_set.jsonl"),
        },
    }


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _normalise(value: str) -> str:
    return re.sub(r"\W+", " ", value.lower()).strip()


def _quality_rows(dataset: str, questions: dict[str, dict[str, Any]], successes: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system in REQUIRED_SYSTEMS:
        for question_id, question in questions.items():
            trace = successes[system][question_id]
            base = {
                "run_id": trace["run_id"], "phase": "pilot", "dataset": dataset,
                "system": system, "question_id": question_id,
                "repetition": int(trace.get("repetition", 1)), "backend": "deterministic",
            }
            reference = question.get("reference_answer")
            answer = trace.get("answer")
            if isinstance(reference, str) and reference.strip() and isinstance(answer, str):
                rows.append({**base, "metric": "answer_exact_match_proxy", "value": float(_normalise(reference) == _normalise(answer)), "coverage": "available"})
            else:
                rows.append({**base, "metric": "answer_exact_match_proxy", "value": None, "coverage": "unavailable", "reason": "reference_answer_unavailable"})

            evidence = [item for item in question.get("evidence", []) if isinstance(item, str) and item.strip()]
            contexts = [item.get("text", "") for item in trace.get("contexts", []) if isinstance(item, dict) and item.get("text")]
            if not evidence:
                rows.append({**base, "metric": "evidence_recall_proxy", "value": None, "coverage": "unavailable", "reason": "evidence_unavailable"})
            elif not contexts:
                rows.append({**base, "metric": "evidence_recall_proxy", "value": None, "coverage": "unavailable", "reason": "contexts_unavailable"})
            else:
                context = _normalise(" ".join(contexts))
                found = sum(_normalise(item) in context for item in evidence)
                rows.append({**base, "metric": "evidence_recall_proxy", "value": found / len(evidence), "coverage": "available"})
    return rows


def _usage_subset(events: list[dict[str, Any]], **filters: str) -> dict[str, Any]:
    selected = [event for event in events if all(event.get(key) == value for key, value in filters.items())]
    return summarize_usage(selected)


def _system_summary(
    system: str,
    run_id: str,
    traces: list[dict[str, Any]],
    successes: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    provider: dict[str, Any],
    expected_questions: int,
) -> dict[str, Any]:
    latency = [float(row["elapsed_seconds"]) for row in successes.values()]
    usage = summarize_usage(events)
    usage["estimated_cost_usd"] = usage_cost_usd(usage, provider)
    dimensions: dict[str, dict[str, Any]] = {}
    for label, filters in (
        ("index", {"operation": "index"}), ("query", {"operation": "query"}),
        ("generator", {"model_kind": "generator"}), ("embedding", {"model_kind": "embedding"}),
    ):
        item = _usage_subset(events, **filters)
        item["estimated_cost_usd"] = usage_cost_usd(item, provider)
        dimensions[label] = item

    quality_summary: dict[str, dict[str, Any]] = {}
    for metric in ("answer_exact_match_proxy", "evidence_recall_proxy"):
        metric_rows = [row for row in quality if row["system"] == system and row["metric"] == metric]
        available = [float(row["value"]) for row in metric_rows if row["coverage"] == "available"]
        quality_summary[metric] = {
            "mean": mean(available) if available else None,
            "available": len(available),
            "total": len(metric_rows),
            "coverage": len(available) / len(metric_rows) if metric_rows else 0.0,
        }
    return {
        "system": system,
        "run_id": run_id,
        "expected_questions": expected_questions,
        "successful_questions": len(successes),
        "failed_trace_rows": sum(row.get("status") != "ok" for row in traces),
        "latency_seconds": {
            "mean": mean(latency) if latency else None,
            "median": median(latency) if latency else None,
            "p95": _quantile(latency, 0.95),
        },
        "usage": usage,
        "usage_dimensions": dimensions,
        "quality": quality_summary,
    }


def _flat_summary(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "system": row["system"], "run_id": row["run_id"],
        "expected_questions": row["expected_questions"], "successful_questions": row["successful_questions"],
        "failed_trace_rows": row["failed_trace_rows"],
        "latency_mean_seconds": row["latency_seconds"]["mean"],
        "latency_median_seconds": row["latency_seconds"]["median"],
        "latency_p95_seconds": row["latency_seconds"]["p95"],
        "answer_exact_match_mean": row["quality"]["answer_exact_match_proxy"]["mean"],
        "answer_exact_match_coverage": row["quality"]["answer_exact_match_proxy"]["coverage"],
        "evidence_recall_mean": row["quality"]["evidence_recall_proxy"]["mean"],
        "evidence_recall_coverage": row["quality"]["evidence_recall_proxy"]["coverage"],
    }
    for label, usage in (("total", row["usage"]), *row["usage_dimensions"].items()):
        for field in ("input_tokens", "output_tokens", "api_calls", "retries", "failed_calls", "unknown_output_calls"):
            flat[f"{label}_{field}"] = usage[field]
        flat[f"{label}_estimated_cost_usd"] = usage["estimated_cost_usd"]
        for source in ("provider", "estimated", "partial_estimate"):
            flat[f"{label}_{source}_calls"] = usage.get("usage_sources", {}).get(source, 0)
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: Any, digits: int = 4) -> str:
    return "unavailable" if value is None else f"{float(value):.{digits}f}"


def _report(dataset: str, rows: list[dict[str, Any]], actual: int, configured: int) -> str:
    lines = [
        "# Pilot Collection Report", "", f"Dataset: `{dataset}`", "", "Structural status: **VALID FOR PILOT REVIEW**", "",
        "## Selected runs", "", "| System | Run ID | Successful questions | Failed trace rows |", "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['system']} | `{row['run_id']}` | {row['successful_questions']}/{row['expected_questions']} | {row['failed_trace_rows']} |")
    lines += ["", "## Operational summary", "", "| System | Mean latency (s) | P95 latency (s) | API calls | Input tokens | Output tokens | Estimated cost (USD) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        usage = row["usage"]
        lines.append(f"| {row['system']} | {_format_number(row['latency_seconds']['mean'])} | {_format_number(row['latency_seconds']['p95'])} | {usage['api_calls']} | {usage['input_tokens']} | {usage['output_tokens']} | {_format_number(usage['estimated_cost_usd'], 6)} |")
    lines += ["", "## Deterministic quality summary", "", "| System | Exact-match mean | Exact-match coverage | Evidence-recall mean | Evidence-recall coverage |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        answer = row["quality"]["answer_exact_match_proxy"]
        evidence = row["quality"]["evidence_recall_proxy"]
        lines.append(f"| {row['system']} | {_format_number(answer['mean'])} | {_format_number(answer['coverage'])} | {_format_number(evidence['mean'])} | {_format_number(evidence['coverage'])} |")
    size_limitation = (
        f"- The configured pilot size is {configured}, but the frozen prepared pilot contains {actual} questions. This collection was created with the explicit short-pilot override."
        if actual < configured else f"- The frozen prepared pilot contains the configured {actual} questions."
    )
    lines += [
        "", "## Limitations", "", size_limitation,
        "- Where retrieved contexts are unavailable, context-dependent metrics are marked unavailable rather than treated as zero.",
        "- Token counts identify provider-reported, estimated, and partial-estimate calls in the structured artifacts.",
        "- This pilot report is descriptive. It does not perform confidence intervals, significance tests, or claim final comparative findings.",
        "- Paid RAGAS/RAGChecker evaluation was not run by collection.", "",
        "## Source of truth", "",
        "Use `collection_manifest.json`, `summary.json`, `traces.jsonl`, and `usage_events.jsonl` for audited values. Do not use the mutable top-level `results/run_summary.json` for this comparison.", "",
    ]
    return "\n".join(lines)


def collect_pilot(config: Config, dataset: str, selections: dict[str, str], *, allow_short_pilot: bool = False) -> dict[str, Any]:
    if dataset not in config.data["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset}")
    if set(selections) != set(REQUIRED_SYSTEMS) or len(set(selections.values())) != len(REQUIRED_SYSTEMS):
        raise ValueError("Pilot collection requires one distinct run ID for each of naive, graphrag, and lightrag")

    prepared = config.data_root / "prepared" / dataset
    pilot_path = prepared / "pilot_set.jsonl"
    if not pilot_path.exists():
        raise ValueError(f"Prepared pilot is missing: {pilot_path}")
    pilot_rows = read_jsonl(pilot_path)
    question_ids = [str(row.get("id", "")) for row in pilot_rows]
    if not question_ids or any(not item for item in question_ids) or len(set(question_ids)) != len(question_ids):
        raise ValueError("Prepared pilot questions require unique non-empty IDs")
    questions = {row["id"]: row for row in pilot_rows}
    configured_size = int(config.dataset(dataset)["pilot_size"])
    if len(pilot_rows) < configured_size and not allow_short_pilot:
        raise ValueError(f"Prepared pilot has {len(pilot_rows)} questions but config requires {configured_size}; pass --allow-short-pilot to record this exception")

    manifests: dict[str, dict[str, Any]] = {}
    manifest_hashes: dict[str, str] = {}
    bases: dict[str, dict[str, Any]] = {}
    repetitions: set[int] = set()
    for system in REQUIRED_SYSTEMS:
        run_id = selections[system]
        run_root = config.output_root / "runs" / run_id
        manifest_path = run_root / "run_manifest.json"
        summary_path = run_root / "run_summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            raise ValueError(f"Run {run_id} is missing its manifest or summary")
        manifest = read_json(manifest_path)
        run_summary = read_json(summary_path)
        if manifest.get("run_id") != run_id or run_summary.get("run_id") != run_id:
            raise ValueError(f"Run directory {run_id} contains mismatched identifiers")
        if manifest.get("phase") != "pilot" or manifest.get("status") != "complete" or not run_summary.get("complete"):
            raise ValueError(f"Run {run_id} must be a complete pilot")
        if manifest.get("systems") != [system] or run_summary.get("systems") != [system]:
            raise ValueError(f"Run {run_id} must contain only declared system {system}")
        counts = manifest.get("counts", {})
        if int(counts.get("indexes_incomplete", 0)) != 0:
            raise ValueError(f"Run {run_id} has an incomplete index")
        index_path = run_root / dataset / system / "index_run.json"
        if not index_path.exists() or read_json(index_path).get("status") != "ok":
            raise ValueError(f"Run {run_id} has no complete index record for {dataset}/{system}")
        basis = _comparison_basis_from_manifest(manifest, dataset)
        repetitions.add(int(basis["repetition"]))
        manifests[system], bases[system] = manifest, basis
        manifest_hashes[system] = sha256_file(manifest_path)
    if len(repetitions) != 1:
        raise ValueError("Selected runs use different repetition numbers")
    repetition = next(iter(repetitions))
    current_basis = _comparison_basis_from_current(config, dataset, repetition)
    expected_basis_hash = _canonical_hash(current_basis)
    for system, basis in bases.items():
        if _canonical_hash(basis) != expected_basis_hash:
            raise ValueError(f"Run {selections[system]} is incompatible with the other runs or current prepared inputs")

    trace_path = config.output_root / "query_traces.jsonl"
    if not trace_path.exists():
        raise ValueError(f"Global trace journal is missing: {trace_path}")
    all_traces = read_jsonl(trace_path)
    selected_traces: list[dict[str, Any]] = []
    successes: dict[str, dict[str, dict[str, Any]]] = {system: {} for system in REQUIRED_SYSTEMS}
    traces_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_set = set(question_ids)
    for trace in all_traces:
        system = str(trace.get("system", ""))
        if system not in selections or trace.get("run_id") != selections[system] or trace.get("dataset") != dataset:
            continue
        if trace.get("phase") != "pilot" or int(trace.get("repetition", 1)) != repetition:
            raise ValueError(f"Trace in run {trace.get('run_id')} has incompatible phase or repetition")
        if trace.get("question_id") not in expected_set:
            raise ValueError(f"Run {trace.get('run_id')} contains unexpected pilot question {trace.get('question_id')}")
        selected_traces.append(trace)
        traces_by_system[system].append(trace)
        if trace.get("status") == "ok" and trace.get("metadata", {}).get("index_complete", True):
            question_id = str(trace["question_id"])
            if question_id in successes[system]:
                raise ValueError(f"Run {trace['run_id']} has multiple successful rows for {question_id}")
            successes[system][question_id] = trace
    for system in REQUIRED_SYSTEMS:
        missing = sorted(expected_set - set(successes[system]))
        if missing:
            raise ValueError(f"Run {selections[system]} is missing successful pilot questions: {', '.join(missing)}")

    selected_events: list[dict[str, Any]] = []
    events_by_system: dict[str, list[dict[str, Any]]] = {}
    seen_events: set[str] = set()
    for system in REQUIRED_SYSTEMS:
        events_path = config.output_root / "runs" / selections[system] / dataset / system / "usage_events.jsonl"
        if not events_path.exists():
            raise ValueError(f"Run {selections[system]} has no usage journal; start a fresh instrumented pilot run")
        events = read_jsonl(events_path)
        if not events:
            raise ValueError(f"Run {selections[system]} has an empty usage journal")
        operations = {event.get("operation") for event in events}
        if not {"index", "query"}.issubset(operations):
            raise ValueError(f"Run {selections[system]} usage journal must include index and query calls")
        normalized: list[dict[str, Any]] = []
        for event in events:
            if event.get("system") != system or event.get("phase") != "pilot":
                raise ValueError(f"Run {selections[system]} contains incompatible usage events")
            event_id = str(event.get("event_id", ""))
            if not event_id or event_id in seen_events:
                raise ValueError(f"Run {selections[system]} contains missing or duplicate usage event IDs")
            seen_events.add(event_id)
            enriched = {**event, "run_id": selections[system], "dataset": dataset}
            normalized.append(enriched)
            selected_events.append(enriched)
        events_by_system[system] = normalized

    quality = _quality_rows(dataset, questions, successes)
    system_rows = [
        _system_summary(system, selections[system], traces_by_system[system], successes[system], events_by_system[system], quality, config.data["provider"], len(pilot_rows))
        for system in REQUIRED_SYSTEMS
    ]
    overall_usage = summarize_usage(selected_events)
    overall_usage["estimated_cost_usd"] = usage_cost_usd(overall_usage, config.data["provider"])
    summary = {
        "dataset": dataset, "phase": "pilot", "structurally_valid": True,
        "configured_pilot_size": configured_size, "actual_pilot_size": len(pilot_rows),
        "short_pilot_override": len(pilot_rows) < configured_size and allow_short_pilot,
        "systems": system_rows, "usage": overall_usage,
    }

    selection_basis = {
        "dataset": dataset, "phase": "pilot", "runs": selections,
        "protocol_version": config.data["study"].get("protocol_version"),
        "manifest_sha256": manifest_hashes, "comparison_fingerprint": expected_basis_hash,
        "pilot_questions_sha256": sha256_file(pilot_path), "allow_short_pilot": allow_short_pilot,
    }
    selection_hash = _canonical_hash(selection_basis)
    collection_id = f"pilot-{selection_hash[:16]}"
    collection_root = config.output_root / "collections" / dataset / collection_id
    if collection_root.exists():
        manifest_path = collection_root / "collection_manifest.json"
        if not manifest_path.exists() or read_json(manifest_path).get("selection_hash") != selection_hash:
            raise RuntimeError(f"Collection destination already exists but is not the expected immutable collection: {collection_root}")
        return {"collection_id": collection_id, "path": str(collection_root), "created": False, "structurally_valid": True}

    collection_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{collection_id}-", dir=collection_root.parent))
    try:
        write_jsonl(temporary / "traces.jsonl", selected_traces)
        write_jsonl(temporary / "usage_events.jsonl", selected_events)
        write_jsonl(temporary / "quality_metrics.jsonl", quality)
        write_json(temporary / "summary.json", {"collection_id": collection_id, **summary})
        _write_csv(temporary / "summary.csv", [_flat_summary(row) for row in system_rows])
        (temporary / "report.md").write_text(_report(dataset, system_rows, len(pilot_rows), configured_size), encoding="utf-8")
        artifacts = {path.name: sha256_file(path) for path in temporary.iterdir() if path.is_file()}
        collection_manifest = {
            "collection_id": collection_id, "selection_hash": selection_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset, "phase": "pilot", "runs": selections,
            "protocol_version": config.data["study"].get("protocol_version"),
            "manifest_sha256": manifest_hashes, "comparison_fingerprint": expected_basis_hash,
            "pilot_questions_sha256": sha256_file(pilot_path),
            "configured_pilot_size": configured_size, "actual_pilot_size": len(pilot_rows),
            "allow_short_pilot": allow_short_pilot, "structurally_valid": True,
            "validation": [
                "complete_single_system_runs", "compatible_configuration_and_inputs",
                "identical_successful_question_set", "index_and_query_usage_present",
            ],
            "artifacts_sha256": artifacts,
        }
        write_json(temporary / "collection_manifest.json", collection_manifest)
        os.rename(temporary, collection_root)
    except FileExistsError:
        shutil.rmtree(temporary)
        manifest_path = collection_root / "collection_manifest.json"
        if not manifest_path.exists() or read_json(manifest_path).get("selection_hash") != selection_hash:
            raise RuntimeError(f"Concurrent collection creation produced an incompatible destination: {collection_root}")
        return {"collection_id": collection_id, "path": str(collection_root), "created": False, "structurally_valid": True}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"collection_id": collection_id, "path": str(collection_root), "created": True, "structurally_valid": True}
