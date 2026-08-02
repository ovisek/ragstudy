from __future__ import annotations

import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .io import append_jsonl, read_jsonl, sha256_text, write_json, write_jsonl
from .models import Document, Question, Usage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _document_record(document: Document) -> dict[str, Any]:
    return {"id": document.id, "text": document.text, "title": document.title, "metadata": document.metadata}


def _records_hash(records: list[dict[str, Any]]) -> str:
    return sha256_text(json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _unit_token(unit_id: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in unit_id).strip("_")[:48]
    return f"{safe or 'unit'}-{sha256_text(unit_id)[:12]}"


def _journal_rows(workspace: Path) -> list[dict[str, Any]]:
    path = workspace / "index_progress.jsonl"
    return read_jsonl(path) if path.exists() else []


def _successful_units(workspace: Path) -> set[tuple[str, str]]:
    latest: dict[tuple[str, str], str] = {}
    for row in _journal_rows(workspace):
        if row.get("status") != "running":
            latest[(row["unit_id"], row["content_hash"])] = str(row.get("status"))
    return {key for key, status in latest.items() if status == "ok"}


def _next_attempt(workspace: Path, unit_id: str, content_hash: str) -> int:
    attempts = [int(row.get("attempt", 0)) for row in _journal_rows(workspace) if row.get("unit_id") == unit_id and row.get("content_hash") == content_hash]
    return 1 + max(attempts, default=0)


def _write_index_summary(workspace: Path, *, backend: str, total: int, unit_hashes: dict[str, str]) -> dict[str, Any]:
    rows = _journal_rows(workspace)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("unit_id")), str(row.get("content_hash")))
        if row.get("status") != "running":
            latest[key] = row
    successful = [unit_id for unit_id, content_hash in unit_hashes.items() if latest.get((unit_id, content_hash), {}).get("status") == "ok"]
    failed = [unit_id for unit_id, content_hash in unit_hashes.items() if latest.get((unit_id, content_hash), {}).get("status") in {"failed", "interrupted"}]
    pending = [unit_id for unit_id in unit_hashes if unit_id not in successful and unit_id not in failed]
    summary = {
        "backend": backend,
        "updated_at": _now(),
        "total_units": total,
        "successful_units": len(successful),
        "failed_units": failed,
        "pending_units": pending,
        "complete": len(successful) == total,
        "attempts": sum(1 for row in rows if row.get("status") != "running"),
    }
    write_json(workspace / "index_summary.json", summary)
    return summary


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
    def index(self, documents: list[Document], incremental: bool = False, resume: bool = False, retries: int = 2) -> dict[str, Any]:
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

    def index(self, documents: list[Document], incremental: bool = False, resume: bool = False, retries: int = 2) -> dict[str, Any]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        document_dir = self.workspace / "naive_documents"
        document_dir.mkdir(parents=True, exist_ok=True)
        unit_hashes = {document.id: _records_hash([_document_record(document)]) for document in documents}
        successful = _successful_units(self.workspace) if resume else set()
        usage = Usage()

        for position, document in enumerate(documents, 1):
            content_hash = unit_hashes[document.id]
            artifact = document_dir / f"{_unit_token(document.id)}.jsonl"
            if (document.id, content_hash) in successful and artifact.exists():
                print(f"[{position}/{len(documents)}] naive: skipping indexed document {document.id}", flush=True)
                continue
            attempt = _next_attempt(self.workspace, document.id, content_hash)
            for retry_number in range(retries + 1):
                current_attempt = attempt + retry_number
                print(f"[{position}/{len(documents)}] naive: indexing {document.id} (attempt {current_attempt})", flush=True)
                started = time.monotonic()
                append_jsonl(self.workspace / "index_progress.jsonl", {"timestamp": _now(), "unit_id": document.id, "content_hash": content_hash, "attempt": current_attempt, "status": "running"})
                try:
                    chunks = chunk_text(document, int(self.common["chunk_tokens"]), int(self.common["chunk_overlap_tokens"]))
                    if not chunks:
                        raise ValueError("Document has no non-empty chunks")
                    document_records: list[dict[str, Any]] = []
                    attempt_usage = Usage()
                    for start in range(0, len(chunks), 64):
                        batch = chunks[start : start + 64]
                        vectors, batch_usage = self._embed([chunk["text"] for chunk in batch])
                        attempt_usage.add(batch_usage)
                        document_records.extend({**chunk, "embedding": vector} for chunk, vector in zip(batch, vectors, strict=True))
                    write_jsonl(artifact, document_records)
                    usage.add(attempt_usage)
                    append_jsonl(self.workspace / "index_progress.jsonl", {"timestamp": _now(), "unit_id": document.id, "content_hash": content_hash, "attempt": current_attempt, "status": "ok", "elapsed_seconds": time.monotonic() - started, "chunks": len(document_records), "usage": asdict(attempt_usage)})
                    break
                except KeyboardInterrupt:
                    append_jsonl(self.workspace / "index_progress.jsonl", {"timestamp": _now(), "unit_id": document.id, "content_hash": content_hash, "attempt": current_attempt, "status": "interrupted", "elapsed_seconds": time.monotonic() - started})
                    _write_index_summary(self.workspace, backend=self.system, total=len(documents), unit_hashes=unit_hashes)
                    raise
                except Exception as exc:
                    append_jsonl(self.workspace / "index_progress.jsonl", {"timestamp": _now(), "unit_id": document.id, "content_hash": content_hash, "attempt": current_attempt, "status": "failed", "elapsed_seconds": time.monotonic() - started, "error": str(exc)})
                    if retry_number < retries:
                        time.sleep(2 ** retry_number)

        successful = _successful_units(self.workspace)
        records = []
        for document in documents:
            if (document.id, unit_hashes[document.id]) in successful:
                artifact = document_dir / f"{_unit_token(document.id)}.jsonl"
                if artifact.exists():
                    records.extend(read_jsonl(artifact))
        if incremental and self.index_file.exists():
            existing = read_jsonl(self.index_file)
            existing_ids = {item["id"] for item in existing}
            records = existing + [item for item in records if item["id"] not in existing_ids]
        write_jsonl(self.index_file, records)
        summary = _write_index_summary(self.workspace, backend=self.system, total=len(documents), unit_hashes=unit_hashes)
        return {**summary, "chunks": len(records), "usage": asdict(usage), "index_path": str(self.index_file), "incremental": incremental}

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
    def _redact(self, value: str) -> str:
        key_name = str(self.config.data.get("provider", {}).get("api_key_env", ""))
        secret = os.environ.get(key_name) if key_name else None
        return value.replace(secret, "[REDACTED]") if secret else value

    def _run(self, template: str, log_path: Path, **variables: str) -> tuple[subprocess.CompletedProcess[str], str]:
        command = template.format(**variables)
        # Templates are research-controlled config, recorded in the manifest.
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=environment, start_new_session=True)
        stdout_lines: deque[str] = deque(maxlen=4000)
        stderr_lines: deque[str] = deque(maxlen=4000)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()

        def copy_stream(source, target, captured: deque[str], label: str) -> None:
            assert source is not None
            with log_path.open("a", encoding="utf-8") as log:
                for raw_line in iter(source.readline, ""):
                    line = self._redact(raw_line)
                    captured.append(line)
                    target.write(line)
                    target.flush()
                    with lock:
                        log.write(f"[{label}] {line}")
                        log.flush()
            source.close()

        threads = [
            threading.Thread(target=copy_stream, args=(process.stdout, sys.stdout, stdout_lines, "stdout"), daemon=True),
            threading.Thread(target=copy_stream, args=(process.stderr, sys.stderr, stderr_lines, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        finally:
            for thread in threads:
                thread.join()
        result = subprocess.CompletedProcess(command, return_code, "".join(stdout_lines), "".join(stderr_lines))
        return result, self._redact(command)

    def index(self, documents: list[Document], incremental: bool = False, resume: bool = False, retries: int = 2) -> dict[str, Any]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        records = [_document_record(document) for document in documents]
        default_granularity = "document" if self.system == "lightrag" else "corpus"
        granularity = str(self.config.system(self.system).get("index_checkpoint", default_granularity))
        if granularity not in {"document", "corpus"}:
            raise ValueError(f"Unsupported index checkpoint granularity {granularity!r} for {self.system}")
        units = [(record["id"], [record]) for record in records] if granularity == "document" else [("corpus", records)]
        unit_hashes = {unit_id: _records_hash(unit_records) for unit_id, unit_records in units}
        successful = _successful_units(self.workspace) if resume else set()
        details: dict[str, Any] = {}

        for position, (unit_id, unit_records) in enumerate(units, 1):
            content_hash = unit_hashes[unit_id]
            if (unit_id, content_hash) in successful:
                print(f"[{position}/{len(units)}] {self.system}: skipping indexed {granularity} {unit_id}", flush=True)
                continue
            token = _unit_token(unit_id)
            input_path = self.workspace / "index_inputs" / f"{token}.jsonl"
            output = self.workspace / "index_outputs" / f"{token}.json"
            write_jsonl(input_path, unit_records)
            attempt = _next_attempt(self.workspace, unit_id, content_hash)
            for retry_number in range(retries + 1):
                current_attempt = attempt + retry_number
                log_path = self.workspace / "logs" / "index" / token / f"attempt-{current_attempt:03d}.log"
                output.unlink(missing_ok=True)
                print(f"[{position}/{len(units)}] {self.system}: indexing {granularity} {unit_id} (attempt {current_attempt})", flush=True)
                started = time.monotonic()
                append_jsonl(self.workspace / "index_progress.jsonl", {"timestamp": _now(), "unit_id": unit_id, "content_hash": content_hash, "attempt": current_attempt, "status": "running", "granularity": granularity, "log_path": str(log_path)})
                execution_error = None
                try:
                    result, expanded = self._run(self.config.system(self.system)["index_command"], log_path=log_path, workspace=str(self.workspace), input=str(input_path), question="", method="", output=str(output), incremental_flag="--incremental" if incremental else "")
                except KeyboardInterrupt:
                    append_jsonl(self.workspace / "index_progress.jsonl", {"timestamp": _now(), "unit_id": unit_id, "content_hash": content_hash, "attempt": current_attempt, "status": "interrupted", "granularity": granularity, "elapsed_seconds": time.monotonic() - started, "log_path": str(log_path)})
                    _write_index_summary(self.workspace, backend=self.system, total=len(units), unit_hashes=unit_hashes)
                    raise
                except Exception as exc:
                    execution_error = self._redact(str(exc))
                    expanded = self._redact(self.config.system(self.system)["index_command"].format(workspace=str(self.workspace), input=str(input_path), question="", method="", output=str(output), incremental_flag="--incremental" if incremental else ""))
                    result = subprocess.CompletedProcess(expanded, 1, "", execution_error)
                try:
                    detail = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
                except (OSError, json.JSONDecodeError) as exc:
                    detail = {}
                    execution_error = f"Invalid index output {output}: {exc}"
                report = {
                    "timestamp": _now(), "unit_id": unit_id, "content_hash": content_hash, "attempt": current_attempt,
                    "return_code": result.returncode, "elapsed_seconds": time.monotonic() - started,
                    "command": expanded, "stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:],
                    "detail": detail, "incremental": incremental, "document_count": len(unit_records), "log_path": str(log_path),
                }
                append_jsonl(self.workspace / "index_command_reports.jsonl", report)
                write_json(self.workspace / "index_command_report.json", report)
                if result.returncode == 0 and output.exists() and execution_error is None:
                    details[unit_id] = detail
                    append_jsonl(self.workspace / "index_progress.jsonl", {"timestamp": _now(), "unit_id": unit_id, "content_hash": content_hash, "attempt": current_attempt, "status": "ok", "granularity": granularity, "elapsed_seconds": report["elapsed_seconds"], "log_path": str(log_path), "detail": detail})
                    break
                error = execution_error or result.stderr[-1000:] or f"command exited {result.returncode} without creating {output}"
                append_jsonl(self.workspace / "index_progress.jsonl", {"timestamp": _now(), "unit_id": unit_id, "content_hash": content_hash, "attempt": current_attempt, "status": "failed", "granularity": granularity, "elapsed_seconds": report["elapsed_seconds"], "log_path": str(log_path), "error": error})
                if retry_number < retries:
                    print(f"{self.system}: retrying {unit_id} after {2 ** retry_number}s", file=sys.stderr, flush=True)
                    time.sleep(2 ** retry_number)

        summary = _write_index_summary(self.workspace, backend=self.system, total=len(units), unit_hashes=unit_hashes)
        return {**summary, "granularity": granularity, "incremental": incremental, "document_count": len(documents), "details": details, "command_report": str(self.workspace / "index_command_report.json")}

    def query(self, question: Question, method: str) -> tuple[str, list[dict[str, Any]], Usage, dict[str, Any]]:
        token = _unit_token(question.id)
        attempt = 1 + len(list((self.workspace / "logs" / "query" / token).glob("attempt-*.log")))
        output = self.workspace / "query_outputs" / token / f"attempt-{attempt:03d}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.workspace / "logs" / "query" / token / f"attempt-{attempt:03d}.log"
        result, expanded = self._run(self.config.system(self.system)["query_command"], log_path=log_path, workspace=str(self.workspace), input="", question=shlex.quote(question.question), method=method, output=str(output))
        if result.returncode != 0:
            raise RuntimeError(f"{self.system} query failed; see {log_path}: {result.stderr[-1000:]}")
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
        return answer, response.get("contexts", []), usage, {"method": method, "command": expanded, "log_path": str(log_path), "attempt": attempt, **response.get("metadata", {})}


def build_adapter(config: Config, system: str, workspace: Path) -> RagAdapter:
    kind = config.system(system).get("kind")
    if kind == "native":
        return NaiveRagAdapter(config, system, workspace)
    if kind == "command":
        return CommandAdapter(config, system, workspace)
    raise ValueError(f"Unknown adapter kind {kind!r} for {system}")
