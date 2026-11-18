#!/usr/bin/env python3
"""Reissue stale LightRAG post-update queries with answer caching bypassed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ragstudy.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/rq3_medical_micro.toml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="graphrag_bench_medical_rq3_micro")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    run_root = config.output_root / "runs" / args.run_id
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Correction requires a structurally complete source run")
    if args.dataset not in manifest.get("datasets", []):
        raise RuntimeError(f"Dataset {args.dataset} is not part of run {args.run_id}")

    workspace = run_root / args.dataset / "lightrag" / "initial"
    questions = read_jsonl(config.data_root / "prepared" / args.dataset / "update_set.jsonl")
    all_traces = read_jsonl(config.output_root / "update_query_traces.jsonl")
    original = {
        row["question_id"]: row for row in all_traces
        if row.get("run_id") == args.run_id and row.get("dataset") == args.dataset
        and row.get("system") == "lightrag" and row.get("metadata", {}).get("stage") == "post_update"
    }
    if set(original) != {row["id"] for row in questions}:
        raise RuntimeError("Expected exactly one original LightRAG post-update trace per question")

    correction_root = run_root / args.dataset / "lightrag" / "post_update_cache_bypass"
    if correction_root.exists():
        raise RuntimeError(f"Correction directory already exists: {correction_root}")
    correction_root.mkdir(parents=True)
    environment = dict(os.environ)
    environment["RAGSTUDY_BYPASS_QUERY_CACHE"] = "1"
    environment["RAGSTUDY_PHASE"] = "update_post_cache_correction"
    rows: list[dict] = []
    status = "ok"
    error = None

    try:
        for position, question in enumerate(questions, 1):
            output = correction_root / f"{question['id']}.json"
            command = [
                str(ROOT / ".venv" / "bin" / "python"), "scripts/lightrag_bridge.py", "query",
                "--config", str(config.path), "--workspace", str(workspace),
                "--question", question["question"], "--method", "local", "--output", str(output),
            ]
            print(f"[{position}/{len(questions)}] cache-bypassed post-update query {question['id']}", flush=True)
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, env=environment, text=True)
            elapsed = time.monotonic() - started
            if result.returncode != 0 or not output.exists():
                raise RuntimeError(f"Corrective query failed for {question['id']} with exit code {result.returncode}")
            response = json.loads(output.read_text(encoding="utf-8"))
            source = original[question["id"]]
            rows.append({
                "correction_id": f"{args.run_id}:lightrag:post_update:{question['id']}:cache_bypass",
                "run_id": args.run_id,
                "dataset": args.dataset,
                "system": "lightrag",
                "stage": "post_update",
                "question_id": question["id"],
                "question": question["question"],
                "reference_answer": question.get("reference_answer"),
                "status": "ok",
                "started_at": started_at,
                "elapsed_seconds": elapsed,
                "answer": response.get("answer"),
                "contexts": response.get("contexts", []),
                "usage": response.get("usage", {}),
                "metadata": {
                    **response.get("metadata", {}),
                    "correction_reason": "Original post-update answer was served from a pre-update LightRAG query cache entry.",
                    "supersedes_original_trace_sha256": canonical_hash(source),
                    "original_elapsed_seconds": source.get("elapsed_seconds"),
                    "original_answer_sha256": hashlib.sha256(str(source.get("answer", "")).encode("utf-8")).hexdigest(),
                    "answer_changed": response.get("answer") != source.get("answer"),
                    "index_measurements_unchanged": True,
                },
            })
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"

    trace_path = correction_root / "corrected_query_traces.jsonl"
    write_jsonl(trace_path, rows)
    summary = {
        "correction_id": f"{args.run_id}:lightrag:post_update:cache_bypass",
        "source_run_id": args.run_id,
        "dataset": args.dataset,
        "system": "lightrag",
        "stage": "post_update",
        "status": status,
        "error": error,
        "expected_queries": len(questions),
        "successful_queries": len(rows),
        "all_answers_changed": len(rows) == len(questions) and all(row["metadata"]["answer_changed"] for row in rows),
        "actual_api_expenditure_usd": 0.0,
        "notional_estimated_cost_usd": sum(float(row["metadata"].get("notional_estimated_cost_usd", 0.0)) for row in rows),
        "corrected_query_traces": str(trace_path),
        "original_traces_preserved": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (correction_root / "correction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if status == "ok" and len(rows) == len(questions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
