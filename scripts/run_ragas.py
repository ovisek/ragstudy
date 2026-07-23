#!/usr/bin/env python3
"""Run the pinned RAGAS default four-metric suite and emit importable JSONL.

This deliberately lives outside the main runner because RAGAS API changes are
frequent. Archive this script with the exact RAGAS version in the experiment
manifest, and run it only after the evaluator model/pricing fields are frozen.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    provider = config["provider"]
    key = os.environ.get(provider["api_key_env"])
    if not key:
        raise SystemExit(f"Missing {provider['api_key_env']}")
    try:
        from datasets import Dataset
        from openai import AsyncOpenAI
        from ragas import evaluate
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise SystemExit("Install evaluator dependencies: pip install -e '.[evaluation,data,providers]'") from exc
    source_rows = [row for row in read_jsonl(args.input) if row.get("reference_answer") and row.get("contexts")]
    # The default RAGAS suite provides answer relevancy, context precision,
    # faithfulness, and context recall. Rows without gold references are not
    # coerced into context-recall scores; they remain explicitly unavailable.
    dataset_rows = [{
        "trace_key": row["trace_key"],
        "user_input": row["question"],
        "response": row["answer"],
        "retrieved_contexts": row["contexts"],
        "reference": row["reference_answer"] if isinstance(row["reference_answer"], str) else " ".join(row["reference_answer"]),
    } for row in source_rows]
    if not dataset_rows:
        print("No rows with reference answers and retrieved contexts to score.", file=sys.stderr)
        return 2
    client = AsyncOpenAI(api_key=key, base_url=provider["base_url"])
    llm = llm_factory(provider["judge_model"], client=client)
    embeddings = embedding_factory("openai", model=provider["embedding_model"], client=client)
    result = evaluate(Dataset.from_list(dataset_rows), llm=llm, embeddings=embeddings, experiment_name=config["study"]["name"], allow_nest_asyncio=False)
    table = result.to_pandas()
    keys = {row["trace_key"]: row for row in source_rows}
    metric_names = ("answer_relevancy", "context_precision", "faithfulness", "context_recall")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for _, result_row in table.iterrows():
            source = keys[result_row["trace_key"]]
            for metric in metric_names:
                if metric in result_row and result_row[metric] is not None:
                    target.write(json.dumps({"dataset": source["dataset"], "system": source["system"], "question_id": source["question_id"], "metric": metric, "value": float(result_row[metric]), "backend": "ragas", "coverage": "available", "evaluator_model": provider["judge_model"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

