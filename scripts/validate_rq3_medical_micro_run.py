#!/usr/bin/env python3
"""Validate the completed RQ3 Medical micro-run and register safe evidence use."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = "graphrag_bench_medical_rq3_micro"
OLD_RUN_ID = "update-20260824T045852Z-a2544873"
PREFLIGHT_ID = "rq3-medical-micro-20260830T032019Z"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def usage_fields(value: dict) -> dict:
    return {
        "input_tokens": int(value.get("input_tokens", 0)),
        "output_tokens": int(value.get("output_tokens", 0)),
        "api_calls": int(value.get("api_calls", 0)),
        "failed_calls": int(value.get("failed_calls", 0)),
        "retries": int(value.get("retries", 0)),
        "actual_api_expenditure_usd": float(value.get("actual_api_expenditure_usd", 0.0)),
        "notional_cost_usd": float(value.get("notional_cost_usd", value.get("notional_estimated_cost_usd", 0.0))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_id = args.run_id
    run_root = ROOT / "results" / "runs" / run_id
    manifest = read_json(run_root / "run_manifest.json")
    reports = [row for row in read_jsonl(ROOT / "results" / "update_runs.jsonl") if row.get("run_id") == run_id]
    traces = [row for row in read_jsonl(ROOT / "results" / "update_query_traces.jsonl") if row.get("run_id") == run_id]
    correction_root = run_root / DATASET / "lightrag" / "post_update_cache_bypass"
    correction_summary = read_json(correction_root / "correction_summary.json")
    corrected = read_jsonl(correction_root / "corrected_query_traces.jsonl")
    original_lightrag_post = {
        row["question_id"]: row for row in traces
        if row.get("system") == "lightrag" and row.get("metadata", {}).get("stage") == "post_update"
    }

    expected_trace_keys = {
        (system, stage, question)
        for system in ("graphrag", "lightrag")
        for stage in ("pre_update", "post_update")
        for question in ("Medical-422500d5", "Medical-73586ddc", "Medical-a8bad1cf")
    }
    observed_trace_keys = {
        (row.get("system"), row.get("metadata", {}).get("stage"), row.get("question_id"))
        for row in traces
    }
    corrected_keys = {(row.get("system"), row.get("stage"), row.get("question_id")) for row in corrected}

    data = ROOT / "data" / "prepared" / DATASET
    expected_hashes = manifest["inputs"][DATASET]
    hash_checks = {
        "initial_documents": sha256(data / "update_initial_documents.jsonl") == expected_hashes["initial_documents_sha256"],
        "added_documents": sha256(data / "update_added_documents.jsonl") == expected_hashes["added_documents_sha256"],
        "questions": sha256(data / "update_set.jsonl") == expected_hashes["update_questions_sha256"],
        "split": sha256(data / "update_split.json") == expected_hashes["update_split_sha256"],
    }

    graph_initial = read_json(run_root / DATASET / "graphrag" / "initial" / "index_summary.json")
    graph_update = read_json(run_root / DATASET / "graphrag" / "full_reindex" / "index_summary.json")
    light_index = read_json(run_root / DATASET / "lightrag" / "initial" / "index_summary.json")
    graph_report = next(row for row in reports if row.get("system") == "graphrag")
    light_report = next(row for row in reports if row.get("system") == "lightrag")
    light_commands = read_jsonl(run_root / DATASET / "lightrag" / "initial" / "index_command_reports.jsonl")
    light_incremental = [
        row for row in light_commands
        if row.get("incremental") is True and row.get("return_code") == 0
        and int(row.get("document_count", -1)) == 1
    ]
    if len(light_incremental) != 1:
        raise RuntimeError(f"Expected one successful LightRAG incremental command, found {len(light_incremental)}")
    light_incremental_command = light_incremental[0]
    light_incremental_usage = usage_fields(light_incremental_command.get("detail", {}).get("usage", {}))

    correction_hashes_match = all(
        row.get("metadata", {}).get("supersedes_original_trace_sha256")
        == canonical_hash(original_lightrag_post[row["question_id"]])
        for row in corrected
    )
    corrected_cache_bypassed = all(row.get("metadata", {}).get("query_cache_bypassed") is True for row in corrected)
    corrected_unique = len(corrected_keys) == len(corrected) == 3
    contexts_unavailable = all(not row.get("contexts") for row in traces + corrected)

    checks = {
        "manifest_complete": manifest.get("status") == "complete" and manifest.get("summary", {}).get("complete") is True,
        "two_successful_system_reports": len(reports) == 2 and all(row.get("status") == "ok" for row in reports),
        "twelve_unique_successful_original_traces": len(traces) == 12 and observed_trace_keys == expected_trace_keys and all(row.get("status") == "ok" for row in traces),
        "input_hashes_match": all(hash_checks.values()),
        "exact_nine_plus_one_partition": expected_hashes.get("initial_documents") == 9 and expected_hashes.get("added_documents") == 1 and expected_hashes.get("realized_update_fraction") == 0.1,
        "all_indexes_complete": all(item.get("complete") is True for item in (graph_initial, graph_update, light_index)),
        "three_corrected_lightrag_post_traces": correction_summary.get("status") == "ok" and corrected_unique and all(row.get("status") == "ok" for row in corrected),
        "corrected_traces_supersede_exact_originals": correction_hashes_match,
        "corrected_queries_bypassed_stale_cache": corrected_cache_bypassed,
        "actual_api_expenditure_zero": manifest.get("actual_api_expenditure_usd") == 0.0 and all(float(row.get("actual_api_expenditure_usd", 0.0)) == 0.0 for row in reports) and correction_summary.get("actual_api_expenditure_usd") == 0.0,
        "context_metrics_blocked_not_zero_filled": contexts_unavailable,
    }

    update_comparison = [
        {
            "system": "GraphRAG",
            "update_method": "full re-index of 10 documents",
            "duration_seconds": float(graph_report["update_seconds"]),
            "duration_minutes": float(graph_report["update_seconds"]) / 60,
            **usage_fields(graph_report["update_usage"]),
            "duration_source": "separate full_reindex workspace cumulative duration",
        },
        {
            "system": "LightRAG",
            "update_method": "incremental insertion of 1 document",
            "duration_seconds": float(light_incremental_command["elapsed_seconds"]),
            "duration_minutes": float(light_incremental_command["elapsed_seconds"]) / 60,
            **light_incremental_usage,
            "duration_source": "successful incremental index command only",
        },
    ]
    update_comparison[1]["duration_reduction_vs_graphrag"] = 1 - update_comparison[1]["duration_seconds"] / update_comparison[0]["duration_seconds"]

    question_labels = {
        "Medical-422500d5": "Anatomical locations",
        "Medical-73586ddc": "Most common type",
        "Medical-a8bad1cf": "Cell type of origin",
    }
    provisional_recovery = [
        {"system": "GraphRAG", "question_id": q, "question_label": label, "post_update_recovered": False, "coding": "provisional manual reference consistency"}
        for q, label in question_labels.items()
    ] + [
        {"system": "LightRAG", "question_id": q, "question_label": label, "post_update_recovered": q in {"Medical-422500d5", "Medical-a8bad1cf"}, "coding": "provisional manual reference consistency; corrected cache-bypassed answer"}
        for q, label in question_labels.items()
    ]

    preflight = read_json(ROOT / "results" / "preflight" / PREFLIGHT_ID / "preflight_reassessment.json")
    old_hash_manifest = ROOT / "results" / f"{OLD_RUN_ID}_preservation_hashes.txt"
    evidence_register = [
        {
            "evidence_id": "E-OLD-FULL",
            "artifact": OLD_RUN_ID,
            "classification": "diagnostic",
            "safe_use": "Operational reliability evidence only",
            "excluded_use": "Do not compare its timings with the micro-experiment",
            "status": "preserved" if old_hash_manifest.exists() else "missing",
        },
        {
            "evidence_id": "E-PREFLIGHT",
            "artifact": PREFLIGHT_ID,
            "classification": "diagnostic",
            "safe_use": "Provider and adapter readiness evidence",
            "excluded_use": "Do not include in measured RQ3 performance",
            "status": preflight.get("status"),
        },
        {
            "evidence_id": "E-MICRO-MAIN",
            "artifact": run_id,
            "classification": "observed feasibility experiment",
            "safe_use": "GraphRAG traces, both indexing phases, LightRAG initial/pre-update traces, and index measurements subject to normalized duration fields",
            "excluded_use": "Original LightRAG post-update traces are stale-cache artifacts",
            "status": manifest.get("status"),
        },
        {
            "evidence_id": "E-LIGHT-CORRECTED",
            "artifact": correction_summary.get("correction_id"),
            "classification": "observed corrective measurement",
            "safe_use": "Authoritative LightRAG post-update answers and query timings",
            "excluded_use": "Does not change original indexing duration or usage",
            "status": correction_summary.get("status"),
        },
    ]

    findings = [
        {
            "severity": "high",
            "finding": "Original LightRAG post-update answers were served from pre-update query cache entries.",
            "impact": "Those three original traces are invalid for retrieval-effectiveness analysis.",
            "remediation": "Use only the three cache-bypassed corrected traces; preserve originals for audit.",
        },
        {
            "severity": "high",
            "finding": "LightRAG report.update_seconds is cumulative because initial and incremental work share one workspace.",
            "impact": "Using 14,041.7 seconds as incremental duration would overstate update burden by including initial indexing.",
            "remediation": f"Use the successful incremental command duration of {light_incremental_command['elapsed_seconds']:.3f} seconds.",
        },
        {
            "severity": "medium",
            "finding": "Provider instability affected LightRAG more heavily than GraphRAG during this single run.",
            "impact": "Observed elapsed time combines system behavior with endpoint reliability and retry delays.",
            "remediation": "Report failures/retries beside timings and avoid general performance claims.",
        },
        {
            "severity": "medium",
            "finding": "Neither graph adapter exposes retrieved contexts in the canonical traces.",
            "impact": "Context precision, context recall and faithfulness cannot be computed for RQ3.",
            "remediation": "Mark context metrics unavailable; never encode them as zero.",
        },
        {
            "severity": "medium",
            "finding": "The experiment has three questions and no repetitions.",
            "impact": "Results are descriptive feasibility evidence, not inferential or generalizable evidence.",
            "remediation": "Recommend repeated, larger-corpus validation as future work.",
        },
    ]

    report = {
        "validation_id": f"{run_id}:rq3-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset": DATASET,
        "grain": "one system-stage-question trace; three questions per system; one 9+1 corpus update",
        "step_7_complete": all(checks.values()),
        "safe_for_thesis_use_with_restrictions": all(checks.values()),
        "checks": checks,
        "hash_checks": hash_checks,
        "counts": {"reports": len(reports), "original_traces": len(traces), "corrected_traces": len(corrected)},
        "update_comparison": update_comparison,
        "provisional_reference_recovery": provisional_recovery,
        "findings": findings,
        "evidence_register": evidence_register,
        "interpretation_limits": [
            "Use descriptive language only; do not claim statistical significance or system superiority.",
            "Do not mix the old full-corpus diagnostic timings with the micro-experiment.",
            "Use corrected LightRAG post-update traces in place of the stale originals.",
            "Use update-only normalized duration and usage fields, not cumulative LightRAG report fields.",
            "Actual API expenditure is USD 0; price-derived amounts are notional estimates.",
        ],
    }
    validation_path = run_root / "rq3_validation_report.json"
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    recovery_rows = []
    for row in provisional_recovery:
        recovery_rows.append(
            f"| {row['system']} | {row['question_label']} | {'Recovered' if row['post_update_recovered'] else 'Not recovered'} | {row['coding']} |"
        )
    evidence_rows = []
    for row in evidence_register:
        evidence_rows.append(
            f"| {row['evidence_id']} | {row['classification']} | {row['status']} | {row['safe_use']} | {row['excluded_use']} |"
        )
    graph, light = update_comparison
    markdown = f"""# RQ3 Medical Micro-Experiment Validation and Evidence Register

## Technical summary

Step 7 is **complete with a mandatory trace correction**. The measured run contains two successful system reports and all 12 expected original traces. The three original LightRAG post-update traces are retained for audit but excluded because they reused pre-update query-cache entries. Three cache-bypassed corrected traces are the authoritative LightRAG post-update source.

The experiment is safe to use as bounded, descriptive feasibility evidence. It does not support statistical significance, cross-domain generalization or a claim of overall system superiority.

## Controlled update measurements

| System | Update operation | Duration (s) | Duration (min) | Input tokens | Output tokens | API calls | Failed calls |
|---|---|---:|---:|---:|---:|---:|---:|
| GraphRAG | {graph['update_method']} | {graph['duration_seconds']:.3f} | {graph['duration_minutes']:.2f} | {graph['input_tokens']:,} | {graph['output_tokens']:,} | {graph['api_calls']} | {graph['failed_calls']} |
| LightRAG | {light['update_method']} | {light['duration_seconds']:.3f} | {light['duration_minutes']:.2f} | {light['input_tokens']:,} | {light['output_tokens']:,} | {light['api_calls']} | {light['failed_calls']} |

LightRAG's update-only duration is taken from the successful incremental command. The cumulative `update_seconds` field in the system report is excluded because that workspace also contains initial-index activity.

## Provisional post-update answer recovery

| System | Question | Outcome | Basis |
|---|---|---|---|
{chr(10).join(recovery_rows)}

This coding is a transparent diagnostic, not a formal evaluator score. GraphRAG did not provide a context-grounded reference answer for any of the three questions. Corrected LightRAG answers recovered two of three reference facts.

## Evidence separation rules

| Evidence ID | Classification | Status | Permitted use | Excluded use |
|---|---|---|---|---|
{chr(10).join(evidence_rows)}

## Data-quality findings

- **High — stale LightRAG post-update cache:** exclude the three original post-update LightRAG traces and use the correction ledger.
- **High — cumulative LightRAG update field:** use {light['duration_seconds']:.3f} seconds, not {float(light_report['update_seconds']):.3f} seconds, for incremental insertion.
- **Medium — provider instability:** elapsed time includes retries and endpoint failures; show failure and retry counts beside timings.
- **Medium — unavailable context traces:** context-dependent evaluation is blocked and must not be represented as zero.
- **Medium — small unrepeated design:** three questions and one run permit descriptive feasibility interpretation only.

## Required thesis-use rules

1. Keep the old full-corpus run as operational diagnostic evidence only.
2. Use the new micro-run for the controlled 9+1 RQ3 comparison.
3. Substitute corrected LightRAG post-update traces for the stale originals.
4. Use normalized update-only duration and usage fields.
5. Report actual API expenditure as USD 0 and price-derived values as notional estimates.
6. Do not claim significance, general superiority or full-corpus scalability.
"""
    register_path = run_root / "rq3_evidence_register.md"
    register_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "step_7_complete": report["step_7_complete"],
        "safe_for_thesis_use_with_restrictions": report["safe_for_thesis_use_with_restrictions"],
        "validation_report": str(validation_path),
        "evidence_register": str(register_path),
        "checks": checks,
    }, indent=2, sort_keys=True))
    return 0 if report["step_7_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
