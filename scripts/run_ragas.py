#!/usr/bin/env python3
"""Run the pinned RAGAS suite with explicit partial-context coverage."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from dotenv import load_dotenv

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

ANSWER_METRICS = ("answer_relevancy",)
CONTEXT_METRICS = ("context_precision", "faithfulness", "context_recall")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def source_identity(source: dict, metric: str, value, coverage: str, model: str, version: str, **extra) -> dict:
    return {
        "run_id": source.get("run_id"),
        "dataset": source["dataset"],
        "system": source["system"],
        "question_id": source["question_id"],
        "repetition": int(str(source["trace_key"]).rsplit("::", 1)[-1]),
        "metric": metric,
        "value": value,
        "backend": "ragas",
        "coverage": coverage,
        "evaluator_model": model,
        "evaluator_version": version,
        **extra,
    }


def result_rows(table, sources: list[dict], metric_names: tuple[str, ...], model: str, version: str) -> list[dict]:
    if len(table) != len(sources):
        raise RuntimeError(f"RAGAS returned {len(table)} results for {len(sources)} input rows")
    rows: list[dict] = []
    for source, (_, result_row) in zip(sources, table.iterrows()):
        for metric in metric_names:
            value = result_row.get(metric)
            if value is not None and math.isfinite(float(value)):
                rows.append(source_identity(source, metric, float(value), "available", model, version))
            else:
                raise RuntimeError(f"RAGAS returned no finite {metric} value for {source['trace_key']}")
    return rows


def metric_identity(row: dict) -> tuple:
    return (
        row.get("run_id"), row.get("dataset"), row.get("system"),
        row.get("question_id"), int(row.get("repetition", 1)), row.get("metric"),
    )


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            target.flush()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def with_heartbeat(label: str, function):
    stop = Event()
    started = monotonic()

    def report() -> None:
        while not stop.wait(30):
            print(f"[{timestamp()}] ACTIVE {label}; elapsed={monotonic() - started:.1f}s", flush=True)

    print(f"[{timestamp()}] START {label}", flush=True)
    thread = Thread(target=report, daemon=True)
    thread.start()
    try:
        return function()
    finally:
        stop.set()
        thread.join(timeout=1)
        print(f"[{timestamp()}] END {label}; elapsed={monotonic() - started:.1f}s", flush=True)


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
        from openai import AsyncOpenAI, OpenAI
        from ragas import evaluate
        from ragas.embeddings.base import BaseRagasEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise SystemExit("Install evaluator dependencies: uv sync --extra evaluation --extra data --extra providers") from exc

    all_sources = [row for row in read_jsonl(args.input) if row.get("answer")]
    context_sources = [row for row in all_sources if row.get("reference_answer") and row.get("contexts")]
    if not all_sources:
        print("No successful answer rows to score.", file=sys.stderr)
        return 2

    client = AsyncOpenAI(api_key=key, base_url=provider["base_url"], timeout=900.0, max_retries=2)
    sync_client = OpenAI(api_key=key, base_url=provider["base_url"], timeout=900.0, max_retries=2)
    llm = llm_factory(provider["judge_model"], client=client, temperature=0.0, max_tokens=512, max_retries=2)

    class CompatibleEmbeddings(BaseRagasEmbeddings):
        """Expose RAGAS 0.4's legacy methods over the working OpenAI client."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            response = sync_client.embeddings.create(model=provider["embedding_model"], input=texts)
            return [item.embedding for item in response.data]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            response = await client.embeddings.create(model=provider["embedding_model"], input=texts)
            return [item.embedding for item in response.data]

        async def aembed_query(self, text: str) -> list[float]:
            return (await self.aembed_documents([text]))[0]

    embeddings = CompatibleEmbeddings()
    ragas_version = importlib.metadata.version("ragas")
    run_config = RunConfig(timeout=900, max_retries=2, max_wait=30, max_workers=1, seed=int(config["study"]["seed"]))
    started = monotonic()
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    trusted: dict[tuple, dict] = {}
    for row in existing_rows:
        if row.get("coverage") == "available" or row.get("reason") == "contexts_unavailable":
            trusted[metric_identity(row)] = row
    trusted_rows = list(trusted.values())
    if trusted_rows != existing_rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as target:
            for row in trusted_rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"[{timestamp()}] SANITIZED metrics: retained={len(trusted_rows)} removed={len(existing_rows) - len(trusted_rows)}", flush=True)
    completed = set(trusted)

    for index, source in enumerate(all_sources, 1):
        expected = metric_identity(source_identity(source, "answer_relevancy", None, "available", provider["judge_model"], ragas_version))
        if expected in completed:
            print(f"Skipping completed answer_relevancy {index}/{len(all_sources)}: {source['trace_key']}", flush=True)
            continue
        dataset = Dataset.from_list([{"user_input": source["question"], "response": source["answer"]}])
        label = f"answer_relevancy {index}/{len(all_sources)} {source['trace_key']}"
        result = with_heartbeat(
            label,
            lambda: evaluate(
                dataset,
                metrics=[AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)],
                llm=llm,
                embeddings=embeddings,
                experiment_name=f"{config['study']['name']}-answer-{index}",
                run_config=run_config,
                allow_nest_asyncio=False,
            ),
        )
        rows = result_rows(result.to_pandas(), [source], ANSWER_METRICS, provider["judge_model"], ragas_version)
        append_rows(args.output, rows)
        completed.update(metric_identity(row) for row in rows)
        print(f"Checkpointed answer_relevancy {index}/{len(all_sources)}: {source['trace_key']}", flush=True)

    context_keys = {row["trace_key"] for row in context_sources}
    for source in all_sources:
        if source["trace_key"] in context_keys:
            continue
        rows = []
        for metric in CONTEXT_METRICS:
            row = source_identity(source, metric, None, "unavailable", provider["judge_model"], ragas_version, reason="contexts_unavailable")
            if metric_identity(row) not in completed:
                rows.append(row)
        append_rows(args.output, rows)
        completed.update(metric_identity(row) for row in rows)

    context_classes = {
        "context_precision": ContextPrecision,
        "faithfulness": Faithfulness,
        "context_recall": ContextRecall,
    }
    for source_index, source in enumerate(context_sources, 1):
        context_record = {
            "user_input": source["question"],
            "response": source["answer"],
            "retrieved_contexts": source["contexts"],
            "reference": source["reference_answer"] if isinstance(source["reference_answer"], str) else " ".join(source["reference_answer"]),
        }
        for metric in CONTEXT_METRICS:
            expected = metric_identity(source_identity(source, metric, None, "available", provider["judge_model"], ragas_version))
            if expected in completed:
                continue
            label = f"{metric} {source_index}/{len(context_sources)} {source['trace_key']}"
            result = with_heartbeat(
                label,
                lambda metric=metric, source_index=source_index: evaluate(
                    Dataset.from_list([context_record]),
                    metrics=[context_classes[metric](llm=llm)],
                    llm=llm,
                    embeddings=embeddings,
                    experiment_name=f"{config['study']['name']}-{metric}-{source_index}",
                    run_config=run_config,
                    allow_nest_asyncio=False,
                ),
            )
            rows = result_rows(result.to_pandas(), [source], (metric,), provider["judge_model"], ragas_version)
            append_rows(args.output, rows)
            completed.update(metric_identity(row) for row in rows)
            print(f"Checkpointed {metric} {source_index}/{len(context_sources)}: {source['trace_key']}", flush=True)

    output_rows = read_jsonl(args.output)

    manifest = {
        "backend": "ragas",
        "evaluator_model": provider["judge_model"],
        "embedding_model": provider["embedding_model"],
        "evaluator_version": ragas_version,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output),
        "answer_rows": len(all_sources),
        "context_rows": len(context_sources),
        "metric_rows": len(output_rows),
        "answer_relevancy_strictness": 1,
        "judge_max_tokens": 512,
        "run_config": {"timeout_seconds": 900, "max_retries": 2, "max_wait_seconds": 30, "max_workers": 1},
        "elapsed_seconds_this_invocation": monotonic() - started,
        "judge_token_usage": None,
        "judge_token_usage_status": "unavailable_from_ragas_wrapper",
        "actual_expenditure_usd": 0.0,
        "notional_cost_usd": None,
    }
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metric_rows": len(output_rows), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
