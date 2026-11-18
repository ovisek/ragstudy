#!/usr/bin/env python3
"""Bridge official LightRAG Core to ragstudy's JSON contract.

Pinned against lightrag-hku==1.4.14. It uses explicit OpenAI-compatible model
functions and local storage, and performs true incremental insertion when
called again with the same workspace.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ragstudy.usage_tracking import ModelUsageTracker, UsageRecorder, estimate_tokens, summarize_usage, usage_cost_usd

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def config(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def usage_dict(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {"input_tokens": int(value.get("input_tokens", value.get("prompt_tokens", 0)) or 0), "output_tokens": int(value.get("output_tokens", value.get("completion_tokens", 0)) or 0), "api_calls": int(value.get("api_calls", 0) or 0), "retries": int(value.get("retries", 0) or 0)}


def status_record(value: object) -> dict:
    if value is None:
        return {"status": "missing"}
    if isinstance(value, dict):
        record = dict(value)
    elif hasattr(value, "__dict__"):
        record = dict(vars(value))
    else:
        record = {"status": str(value)}
    status = record.get("status", "unknown")
    record["status"] = str(getattr(status, "value", status)).lower()
    return json.loads(json.dumps(record, ensure_ascii=False, default=str))


def concurrent_workers(study: dict) -> int:
    configured = study["common"].get("concurrent_workers", 1)
    raw = os.getenv("RAGSTUDY_CONCURRENT_WORKERS")
    if raw is None:
        if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
            raise ValueError("common.concurrent_workers must be a positive integer")
        return configured
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("common.concurrent_workers must be a positive integer") from exc
    if parsed < 1 or raw.strip() != str(parsed):
        raise ValueError("common.concurrent_workers must be a positive integer")
    return parsed


def requests_per_minute(study: dict) -> int:
    configured = study["provider"].get("requests_per_minute", 30)
    raw = os.getenv("RAGSTUDY_REQUESTS_PER_MINUTE")
    if raw is None:
        if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
            raise ValueError("provider.requests_per_minute must be a positive integer")
        return configured
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("provider.requests_per_minute must be a positive integer") from exc
    if parsed < 1 or raw.strip() != str(parsed):
        raise ValueError("provider.requests_per_minute must be a positive integer")
    return parsed


def embedding_dimension(study: dict) -> int:
    value = study["provider"].get("embedding_dimension")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("provider.embedding_dimension must be a positive integer")
    return value


def embedding_timeout_seconds(study: dict) -> int:
    value = study["provider"].get("embedding_timeout_seconds", 600)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("provider.embedding_timeout_seconds must be a positive integer")
    return value


def generator_request_timeout_seconds(study: dict) -> int:
    value = study["provider"].get("generator_request_timeout_seconds", 300)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "provider.generator_request_timeout_seconds must be a positive integer"
        )
    return value


def generator_worker_timeout_seconds(study: dict) -> int:
    value = study["provider"].get("generator_worker_timeout_seconds", 600)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "provider.generator_worker_timeout_seconds must be a positive integer"
        )
    return value


def request_max_attempts(study: dict) -> int:
    value = study["provider"].get("request_max_attempts", 3)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("provider.request_max_attempts must be a positive integer")
    return value


def retry_delays(study: dict, retries: int) -> list[float]:
    provider = study["provider"]
    base = float(provider.get("retry_backoff_seconds", 4.0))
    maximum = float(provider.get("retry_max_delay_seconds", 60.0))
    if base <= 0 or maximum <= 0:
        raise ValueError("provider retry delays must be positive")
    return [min(maximum, base * (2 ** attempt) + random.uniform(0.0, min(1.0, base))) for attempt in range(retries)]


def is_retryable_provider_error(exc: Exception, retryable: tuple[type[BaseException], ...]) -> bool:
    if not isinstance(exc, retryable):
        return False
    status_code = getattr(exc, "status_code", None)
    return status_code is None or status_code in {408, 409, 429} or int(status_code) >= 500


def environment_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def lightrag_cache_settings(study: dict, operation: str | None = None) -> dict[str, bool]:
    configured = study.get("systems", {}).get("lightrag", {})
    settings = {
        "enable_llm_cache": bool(configured.get("enable_llm_cache", False)),
        "enable_llm_cache_for_entity_extract": bool(configured.get("enable_llm_cache_for_entity_extract", False)),
        "enable_embedding_cache": bool(configured.get("enable_embedding_cache", False)),
    }
    if operation == "query" and environment_flag("RAGSTUDY_BYPASS_QUERY_CACHE"):
        settings["enable_llm_cache"] = False
    return settings


class AsyncRequestPacer:
    """Space API request starts so a fast endpoint cannot exceed its RPM cap."""

    def __init__(self, requests_per_minute: int) -> None:
        if isinstance(requests_per_minute, bool) or not isinstance(requests_per_minute, int) or requests_per_minute < 1:
            raise ValueError("provider.requests_per_minute must be a positive integer")
        self.minimum_interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_request = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._next_request - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_request = time.monotonic() + self.minimum_interval


async def build_rag(study: dict, workspace: Path, operation: str):
    try:
        from lightrag import LightRAG
        from lightrag.llm.openai import InvalidResponseError, openai_complete_if_cache, openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
    except ImportError as exc:
        raise RuntimeError("Install the pinned systems extra before running LightRAG") from exc
    provider, common = study["provider"], study["common"]
    concurrency = concurrent_workers(study)
    dimension = embedding_dimension(study)
    embedding_timeout = embedding_timeout_seconds(study)
    generator_request_timeout = generator_request_timeout_seconds(study)
    generator_worker_timeout = generator_worker_timeout_seconds(study)
    max_attempts = request_max_attempts(study)
    delays = retry_delays(study, max_attempts - 1)
    cache_settings = lightrag_cache_settings(study, operation)
    pacer = AsyncRequestPacer(requests_per_minute(study))
    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing {provider['api_key_env']}")

    recorder = UsageRecorder(workspace / "usage_events.jsonl", system="lightrag", operation=operation)
    retryable_completion = (RateLimitError, APIConnectionError, APIStatusError, APITimeoutError, InvalidResponseError)
    retryable_embedding = (RateLimitError, APIConnectionError, APIStatusError, APITimeoutError)

    async def tracked_call(func, *, tracker, request_value, retryable, output_kind, **kwargs):
        # Bypass LightRAG's opaque tenacity wrapper and the OpenAI SDK's own
        # retries so every actual provider attempt becomes one auditable event.
        raw_func = getattr(func, "__wrapped__", func)
        for attempt in range(max_attempts):
            tracker.attempt = attempt + 1
            await pacer.wait()
            before = tracker.event_count
            try:
                result = await raw_func(token_tracker=tracker, **kwargs)
                if tracker.event_count == before:
                    recorder.record(
                        model=tracker.model,
                        model_kind=tracker.model_kind,
                        input_tokens=estimate_tokens(request_value, tracker.model),
                        output_tokens=0 if output_kind == "embedding" else estimate_tokens(result, tracker.model),
                        usage_source="estimated",
                        status="ok",
                        attempt=attempt + 1,
                    )
                return result
            except Exception as exc:
                recorder.record(
                    model=tracker.model,
                    model_kind=tracker.model_kind,
                    input_tokens=estimate_tokens(request_value, tracker.model),
                    output_tokens=None,
                    usage_source="partial_estimate",
                    status="failed",
                    error=str(exc),
                    attempt=attempt + 1,
                )
                if not is_retryable_provider_error(exc, retryable) or attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(delays[attempt])

    async def completion(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
        request_value = {"system_prompt": system_prompt, "history_messages": history_messages, "prompt": prompt}
        client_configs = dict(kwargs.pop("openai_client_configs", {}) or {})
        client_configs["max_retries"] = 0
        client_configs["timeout"] = generator_request_timeout
        tracker = ModelUsageTracker(recorder, model=provider["generator_model"], model_kind="generator")
        return await tracked_call(
            openai_complete_if_cache,
            tracker=tracker,
            request_value=request_value,
            retryable=retryable_completion,
            output_kind="completion",
            model=provider["generator_model"], prompt=prompt, system_prompt=system_prompt,
            history_messages=history_messages, api_key=api_key, base_url=provider["base_url"],
            keyword_extraction=keyword_extraction, openai_client_configs=client_configs, **kwargs,
        )

    @wrap_embedding_func_with_attrs(embedding_dim=dimension, max_token_size=8192, model_name=provider["embedding_model"])
    async def embeddings(texts):
        tracker = ModelUsageTracker(recorder, model=provider["embedding_model"], model_kind="embedding")
        return await tracked_call(
            openai_embed.func,
            tracker=tracker,
            request_value=texts,
            retryable=retryable_embedding,
            output_kind="embedding",
            texts=texts, model=provider["embedding_model"], api_key=api_key,
            base_url=provider["base_url"], embedding_dim=dimension, max_token_size=8192,
            client_configs={"max_retries": 0},
        )

    rag = LightRAG(working_dir=str(workspace), llm_model_func=completion, llm_model_name=provider["generator_model"], llm_model_max_async=concurrency, default_llm_timeout=generator_worker_timeout, embedding_func=embeddings, default_embedding_timeout=embedding_timeout, chunk_token_size=int(common["chunk_tokens"]), chunk_overlap_token_size=int(common["chunk_overlap_tokens"]), enable_llm_cache=cache_settings["enable_llm_cache"], enable_llm_cache_for_entity_extract=cache_settings["enable_llm_cache_for_entity_extract"], embedding_cache_config={"enabled": cache_settings["enable_embedding_cache"]})
    await rag.initialize_storages()
    return rag, recorder


async def finalize_rag(rag) -> None:
    """Flush storages and stop LightRAG's background priority workers."""
    await rag.finalize_storages()
    worker_functions = (
        getattr(rag, "llm_model_func", None),
        getattr(getattr(rag, "embedding_func", None), "func", None),
    )
    for worker_function in worker_functions:
        shutdown = getattr(worker_function, "shutdown", None)
        if shutdown is not None:
            await shutdown()


async def run_index(args: argparse.Namespace, study: dict) -> None:
    workspace, output = Path(args.workspace), Path(args.output)
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    concurrency = concurrent_workers(study)
    print(f"LightRAG: initializing workspace {workspace} (concurrent workers: {concurrency})", flush=True)
    rag, recorder = await build_rag(study, workspace, "index")
    usage_snapshot = recorder.snapshot()
    try:
        print(f"LightRAG: inserting {len(rows)} document(s)", flush=True)
        await rag.ainsert([row["text"] for row in rows], ids=[row["id"] for row in rows])
        statuses = {row["id"]: status_record(await rag.doc_status.get_by_id(row["id"])) for row in rows}
        unsuccessful = {doc_id: status for doc_id, status in statuses.items() if status["status"] not in {"processed", "completed"}}
        if unsuccessful:
            raise RuntimeError(f"LightRAG did not complete document(s): {json.dumps(unsuccessful, ensure_ascii=False, default=str)}")
        usage = summarize_usage(recorder.events_since(usage_snapshot))
        notional_cost = usage_cost_usd(usage, study["provider"])
        result = {"implementation": "HKUDS/LightRAG==1.4.14", "documents_inserted": len(rows), "document_statuses": statuses, "concurrent_workers": concurrency, "incremental": args.incremental, "cache_settings": lightrag_cache_settings(study, "index"), "request_max_attempts": request_max_attempts(study), "usage": usage, "actual_api_expenditure_usd": 0.0, "notional_estimated_cost_usd": notional_cost, "estimated_cost_usd": notional_cost}
    finally:
        await finalize_rag(rag)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"LightRAG: completed {len(rows)} document(s); wrote {output}", flush=True)


async def run_query(args: argparse.Namespace, study: dict) -> None:
    from lightrag import QueryParam
    workspace, output = Path(args.workspace), Path(args.output)
    rag, recorder = await build_rag(study, workspace, "query")
    usage_snapshot = recorder.snapshot()
    try:
        mode = {"local": "local", "global": "global", "hybrid": "hybrid"}.get(args.method, "hybrid")
        param = QueryParam(mode=mode, top_k=int(study["common"]["top_k"]), user_prompt=study["common"]["answer_prompt"])
        answer = await rag.aquery(args.question, param=param)
        usage = summarize_usage(recorder.events_since(usage_snapshot))
        notional_cost = usage_cost_usd(usage, study["provider"])
        result = {"answer": answer, "contexts": [], "usage": usage, "metadata": {"implementation": "HKUDS/LightRAG==1.4.14", "requested_method": args.method, "executed_method": mode, "context_trace": "unavailable_from_core_API", "cache_settings": lightrag_cache_settings(study, "query"), "query_cache_bypassed": environment_flag("RAGSTUDY_BYPASS_QUERY_CACHE"), "request_max_attempts": request_max_attempts(study), "usage_breakdown": usage["by_operation_model"], "usage_unknown_output_calls": usage["unknown_output_calls"], "actual_api_expenditure_usd": 0.0, "notional_estimated_cost_usd": notional_cost, "estimated_cost_usd": notional_cost}}
    finally:
        await finalize_rag(rag)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("index", "query"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--question")
    parser.add_argument("--method", default="hybrid")
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()
    study = config(Path(args.config))
    if args.action == "index":
        if not args.input:
            parser.error("index requires --input")
        asyncio.run(run_index(args, study))
    else:
        if not args.question:
            parser.error("query requires --question")
        asyncio.run(run_query(args, study))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
