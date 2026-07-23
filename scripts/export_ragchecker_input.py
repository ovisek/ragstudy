#!/usr/bin/env python3
"""Convert harness trace exports to the documented RAGChecker CLI input shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for row in rows:
        answer = row.get("reference_answer")
        if not answer or not row.get("contexts"):
            continue
        results.append({
            "query_id": row["trace_key"],
            "query": row["question"],
            "gt_answer": answer if isinstance(answer, str) else " ".join(answer),
            "response": row["answer"],
            "retrieved_context": [{"doc_id": context.get("id", "unknown"), "text": context.get("text", "")} for context in row["contexts"]],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} RAGChecker rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

