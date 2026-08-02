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
import sys
from pathlib import Path

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


def chunk_llm_concurrency(study: dict) -> int:
    value = study["systems"]["lightrag"].get("chunk_llm_concurrency", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("systems.lightrag.chunk_llm_concurrency must be a positive integer")
    return value


async def build_rag(study: dict, workspace: Path):
    try:
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
    except ImportError as exc:
        raise RuntimeError("Install the pinned systems extra before running LightRAG") from exc
    provider, common = study["provider"], study["common"]
    concurrency = chunk_llm_concurrency(study)
    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing {provider['api_key_env']}")

    async def completion(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
        return await openai_complete_if_cache(provider["generator_model"], prompt, system_prompt=system_prompt, history_messages=history_messages, api_key=api_key, base_url=provider["base_url"], keyword_extraction=keyword_extraction, **kwargs)

    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192, model_name=provider["embedding_model"])
    async def embeddings(texts):
        return await openai_embed.func(texts, model=provider["embedding_model"], api_key=api_key, base_url=provider["base_url"])

    rag = LightRAG(working_dir=str(workspace), llm_model_func=completion, llm_model_name=provider["generator_model"], llm_model_max_async=concurrency, embedding_func=embeddings, chunk_token_size=int(common["chunk_tokens"]), chunk_overlap_token_size=int(common["chunk_overlap_tokens"]), enable_llm_cache=False, enable_llm_cache_for_entity_extract=False, embedding_cache_config={"enabled": False})
    await rag.initialize_storages()
    return rag


async def run_index(args: argparse.Namespace, study: dict) -> None:
    workspace, output = Path(args.workspace), Path(args.output)
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    concurrency = chunk_llm_concurrency(study)
    print(f"LightRAG: initializing workspace {workspace} (chunk LLM concurrency: {concurrency})", flush=True)
    rag = await build_rag(study, workspace)
    try:
        print(f"LightRAG: inserting {len(rows)} document(s)", flush=True)
        await rag.ainsert([row["text"] for row in rows], ids=[row["id"] for row in rows])
        statuses = {row["id"]: status_record(await rag.doc_status.get_by_id(row["id"])) for row in rows}
        unsuccessful = {doc_id: status for doc_id, status in statuses.items() if status["status"] not in {"processed", "completed"}}
        if unsuccessful:
            raise RuntimeError(f"LightRAG did not complete document(s): {json.dumps(unsuccessful, ensure_ascii=False, default=str)}")
        result = {"implementation": "HKUDS/LightRAG==1.4.14", "documents_inserted": len(rows), "document_statuses": statuses, "chunk_llm_concurrency": concurrency, "incremental": args.incremental}
    finally:
        await rag.finalize_storages()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"LightRAG: completed {len(rows)} document(s); wrote {output}", flush=True)


async def run_query(args: argparse.Namespace, study: dict) -> None:
    from lightrag import QueryParam
    workspace, output = Path(args.workspace), Path(args.output)
    rag = await build_rag(study, workspace)
    try:
        mode = {"local": "local", "global": "global", "hybrid": "hybrid"}.get(args.method, "hybrid")
        param = QueryParam(mode=mode, top_k=int(study["common"]["top_k"]), user_prompt=study["common"]["answer_prompt"])
        answer = await rag.aquery(args.question, param=param)
        result = {"answer": answer, "contexts": [], "usage": {}, "metadata": {"implementation": "HKUDS/LightRAG==1.4.14", "requested_method": args.method, "executed_method": mode, "context_trace": "unavailable_from_core_API"}}
    finally:
        await rag.finalize_storages()
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
