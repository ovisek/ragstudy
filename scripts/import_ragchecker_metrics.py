#!/usr/bin/env python3
"""Convert RAGChecker per-result metrics to the BAS metric JSONL contract."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluator-model", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    for result in payload.get("results", []):
        parts = str(result.get("query_id", "")).split("::")
        if len(parts) != 4:
            raise ValueError(f"Invalid RAGChecker query_id: {result.get('query_id')!r}")
        dataset, system, question_id, repetition = parts
        metrics = result.get("metrics", {})
        for metric in ("precision", "recall", "f1"):
            value = metrics.get(metric)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Missing per-query {metric} for {result.get('query_id')}")
            rows.append({
                "dataset": dataset,
                "system": system,
                "question_id": question_id,
                "repetition": int(repetition),
                "metric": metric,
                "value": float(value),
                "backend": "ragchecker",
                "coverage": "available",
                "evaluator_model": args.evaluator_model,
                "evaluator_version": importlib.metadata.version("ragchecker"),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} BAS metric rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
