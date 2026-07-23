from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Config
from .io import read_jsonl, write_json, write_jsonl
from .models import Document, Question, Usage


def chunk_text(document: Document, size: int, overlap: int) -> list[dict[str, Any]]:
    """Stable whitespace-token chunking; its exact implementation is frozen in the manifest."""
    tokens = document.text.split()
    if not tokens:
        return []
    if overlap >= size:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_tokens")
    chunks = []
    step = size - overlap
    for offset in range(0, len(tokens), step):
        slice_ = tokens[offset : offset + size]
        if not slice_:
            break
        chunks.append({
            "id": f"{document.id}::chunk::{len(chunks)}",
            "document_id": document.id,
            "title": document.title,
            "text": " ".join(slice_),
            "token_start": offset,
            "token_end": offset + len(slice_),
        })
        if offset + size >= len(tokens):
            break
    return chunks


def _usage_from(value: Any) -> Usage:
    if value is None:
        return Usage()
    if isinstance(value, dict):
        getter = value.get
    else:
        getter = lambda name, default=0: getattr(value, name, default)
    return Usage(
        input_tokens=int(getter("prompt_tokens", getter("input_tokens", 0)) or 0),
        output_tokens=int(getter("completion_tokens", getter("output_tokens", 0)) or 0),
        api_calls=1,
    )


class RagAdapter(ABC):
    def __init__(self, config: Config, system: str, workspace: Path):
        self.config = config
        self.system = system
        self.workspace = workspace

    @abstractmethod
    def index(self, documents: list[Document], incremental: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def query(self, question: Question, method: str) -> tuple[str, list[dict[str, Any]], Usage, dict[str, Any]]:
        raise NotImplementedError


class NaiveRagAdapter(RagAdapter):
    def __init__(self, config: Config, system: str, workspace: Path):
        super().__init__(config, system, workspace)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the providers extra: pip install -e '.[providers]'") from exc
        provider = config.data["provider"]
        api_key = os.environ.get(provider["api_key_env"])
        if not api_key:
            raise RuntimeError(f"Missing {provider['api_key_env']}")
        self.client = OpenAI(api_key=api_key, base_url=provider["base_url"])
        self.provider = provider
        self.common = config.data["common"]
        self.index_file = self.workspace / "naive_index.jsonl"

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]:
        response = self.client.embeddings.create(model=self.provider["embedding_model"], input=texts)
        vectors = [item.embedding for item in response.data]
        return vectors, _usage_from(response.usage)

    def index(self, documents: list[Document], incremental: bool = False) -> dict[str, Any]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        chunks = [chunk for document in documents for chunk in chunk_text(document, int(self.common["chunk_tokens"]), int(self.common["chunk_overlap_tokens"]))]
        if not chunks:
            raise ValueError("No non-empty chunks available for NaiveRAG")
        usage = Usage()
        records: list[dict[str, Any]] = []
        # A modest batch retains a clear mapping between provider usage and index input.
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            vectors, batch_usage = self._embed([chunk["text"] for chunk in batch])
            usage.add(batch_usage)
            records.extend({**chunk, "embedding": vector} for chunk, vector in zip(batch, vectors, strict=True))
        if incremental and self.index_file.exists():
            existing = read_jsonl(self.index_file)
            existing_ids = {item["id"] for item in existing}
            records = existing + [item for item in records if item["id"] not in existing_ids]
        write_jsonl(self.index_file, records)
        return {"chunks": len(records), "usage": asdict(usage), "index_path": str(self.index_file), "incremental": incremental}

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def query(self, question: Question, method: str) -> tuple[str, list[dict[str, Any]], Usage, dict[str, Any]]:
        if not self.index_file.exists():
            raise RuntimeError("NaiveRAG index is absent")
        vector, usage = self._embed([question.question])
        records = read_jsonl(self.index_file)
        ranked = sorted(records, key=lambda item: self._cosine(vector[0], item["embedding"]), reverse=True)
        contexts = [{key: item[key] for key in ("id", "document_id", "title", "text")} | {"rank": rank + 1} for rank, item in enumerate(ranked[: int(self.common["top_k"])])]
        context_text = "\n\n".join(f"[{item['id']}] {item['text']}" for item in contexts)
        response = self.client.chat.completions.create(
            model=self.provider["generator_model"],
            temperature=float(self.common["temperature"]),
            messages=[
                {"role": "system", "content": self.common["answer_prompt"]},
                {"role": "user", "content": f"Question: {question.question}\n\nContext:\n{context_text}"},
            ],
        )
        usage.add(_usage_from(response.usage))
        answer = response.choices[0].message.content or ""
        return answer, contexts, usage, {"method": "dense_top_k", "requested_method": method}


class CommandAdapter(RagAdapter):
    """Version-isolated official GraphRAG/LightRAG bridge using audited JSON output."""
    def _run(self, template: str, **variables: str) -> tuple[subprocess.CompletedProcess[str], str]:
        command = template.format(**variables)
        # Templates are research-controlled config, recorded in the manifest.
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
        return result, command

    def index(self, documents: list[Document], incremental: bool = False) -> dict[str, Any]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        input_path = self.workspace / ("incremental_input.jsonl" if incremental else "input.jsonl")
        write_jsonl(input_path, [{"id": doc.id, "text": doc.text, "title": doc.title, "metadata": doc.metadata} for doc in documents])
        command, _ = self.config.system(self.system)["index_command"], None
        started = time.monotonic()
        result, expanded = self._run(command, workspace=str(self.workspace), input=str(input_path), question="", method="", output=str(self.workspace / "index_output.json"))
        output = self.workspace / "index_output.json"
        detail = json.loads(output.read_text()) if output.exists() else {}
        report = {
            "return_code": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "command": expanded,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "detail": detail,
            "incremental": incremental,
            "document_count": len(documents),
        }
        write_json(self.workspace / "index_command_report.json", report)
        if result.returncode != 0:
            raise RuntimeError(f"{self.system} index command failed; see {self.workspace / 'index_command_report.json'}")
        return {**detail, "command_report": str(self.workspace / "index_command_report.json")}

    def query(self, question: Question, method: str) -> tuple[str, list[dict[str, Any]], Usage, dict[str, Any]]:
        output = self.workspace / "query_outputs" / f"{question.id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        result, expanded = self._run(self.config.system(self.system)["query_command"], workspace=str(self.workspace), input="", question=shlex.quote(question.question), method=method, output=str(output))
        if result.returncode != 0:
            raise RuntimeError(f"{self.system} query failed: {result.stderr[-1000:]}")
        if not output.exists():
            raise RuntimeError(f"{self.system} query command did not create {output}")
        response = json.loads(output.read_text(encoding="utf-8"))
        answer = response.get("answer")
        if not isinstance(answer, str):
            raise RuntimeError(f"{self.system} query output requires string answer")
        raw_usage = response.get("usage", {})
        usage = Usage(
            input_tokens=int(raw_usage.get("input_tokens", 0)),
            output_tokens=int(raw_usage.get("output_tokens", 0)),
            api_calls=int(raw_usage.get("api_calls", 0)),
            retries=int(raw_usage.get("retries", 0)),
        )
        return answer, response.get("contexts", []), usage, {"method": method, "command": expanded, **response.get("metadata", {})}


def build_adapter(config: Config, system: str, workspace: Path) -> RagAdapter:
    kind = config.system(system).get("kind")
    if kind == "native":
        return NaiveRagAdapter(config, system, workspace)
    if kind == "command":
        return CommandAdapter(config, system, workspace)
    raise ValueError(f"Unknown adapter kind {kind!r} for {system}")

