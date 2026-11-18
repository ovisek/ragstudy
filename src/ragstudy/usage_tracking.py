from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(value: Any, model: str) -> int:
    """Estimate tokens for a request/response value with an explicit fallback."""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model.removeprefix("openai/"))
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Deliberately conservative, and always labeled as an estimate by callers.
        return max(1, (len(text.encode("utf-8")) + 3) // 4) if text else 0


class UsageRecorder:
    """Append-only, process-safe-enough journal for individual provider attempts."""

    def __init__(self, path: Path, *, system: str, operation: str, phase: str | None = None):
        self.path = path
        self.system = system
        self.operation = operation
        self.phase = phase or os.getenv("RAGSTUDY_PHASE", "unknown")
        self._lock = threading.Lock()

    def snapshot(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def record(
        self,
        *,
        model: str,
        model_kind: str,
        input_tokens: int,
        output_tokens: int | None,
        usage_source: str,
        status: str,
        error: str | None = None,
        request_id: str | None = None,
        attempt: int | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": uuid.uuid4().hex,
            "timestamp": _now(),
            "system": self.system,
            "operation": self.operation,
            "phase": self.phase,
            "model": model,
            "model_kind": model_kind,
            "input_tokens": max(0, int(input_tokens)),
            "output_tokens": None if output_tokens is None else max(0, int(output_tokens)),
            "api_calls": 1,
            "usage_source": usage_source,
            "status": status,
            "attempt": attempt,
        }
        if request_id:
            event["request_id"] = request_id
        if error:
            event["error"] = error[:1000]
        payload = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A single O_APPEND write prevents concurrent GraphRAG workers from
        # overwriting one another's journal entries.
        with self._lock:
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)
        return event

    def events_since(self, snapshot: int) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return rows[snapshot:]


class ModelUsageTracker:
    """Adapter for LightRAG's TokenTracker-compatible callback contract."""

    def __init__(self, recorder: UsageRecorder, *, model: str, model_kind: str):
        self.recorder = recorder
        self.model = model
        self.model_kind = model_kind
        self.event_count = 0
        self.attempt: int | None = None

    def add_usage(self, token_counts: dict[str, Any]) -> None:
        self.recorder.record(
            model=self.model,
            model_kind=self.model_kind,
            input_tokens=int(token_counts.get("prompt_tokens", token_counts.get("input_tokens", 0)) or 0),
            output_tokens=int(token_counts.get("completion_tokens", token_counts.get("output_tokens", 0)) or 0),
            usage_source="provider",
            status="ok",
            attempt=self.attempt,
        )
        self.event_count += 1


def summarize_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for event in events:
        key = (
            str(event.get("system", "unknown")), str(event.get("phase", "unknown")),
            str(event.get("operation", "unknown")), str(event.get("model_kind", "unknown")),
            str(event.get("model", "unknown")),
        )
        group = groups.setdefault(key, {
            "system": key[0], "phase": key[1], "operation": key[2],
            "model_kind": key[3], "model": key[4],
            "input_tokens": 0, "output_tokens": 0, "api_calls": 0,
            "failed_calls": 0, "unknown_output_calls": 0, "usage_sources": {},
        })
        group["input_tokens"] += int(event.get("input_tokens", 0) or 0)
        output_tokens = event.get("output_tokens")
        if output_tokens is None:
            group["unknown_output_calls"] += 1
        else:
            group["output_tokens"] += int(output_tokens or 0)
        group["api_calls"] += int(event.get("api_calls", 1) or 0)
        source = str(event.get("usage_source", "unknown"))
        group["usage_sources"][source] = int(group["usage_sources"].get(source, 0)) + 1
        if event.get("status") != "ok":
            group["failed_calls"] += 1

    breakdown = sorted(groups.values(), key=lambda row: (row["system"], row["phase"], row["operation"], row["model_kind"], row["model"]))
    known_retries = sum(1 for event in events if isinstance(event.get("attempt"), int) and int(event["attempt"]) > 1)
    unknown_failed_calls = sum(1 for event in events if event.get("attempt") is None and event.get("status") != "ok")
    return {
        "input_tokens": sum(row["input_tokens"] for row in breakdown),
        "output_tokens": sum(row["output_tokens"] for row in breakdown),
        "api_calls": sum(row["api_calls"] for row in breakdown),
        "retries": known_retries + unknown_failed_calls,
        "retries_source": "estimated" if unknown_failed_calls else "provider_attempts",
        "failed_calls": sum(row["failed_calls"] for row in breakdown),
        "unknown_output_calls": sum(row["unknown_output_calls"] for row in breakdown),
        "usage_sources": {
            source: sum(int(row["usage_sources"].get(source, 0)) for row in breakdown)
            for source in sorted({source for row in breakdown for source in row["usage_sources"]})
        },
        "by_operation_model": breakdown,
    }


def usage_cost_usd(summary: dict[str, Any], provider: dict[str, Any]) -> float:
    total = 0.0
    for row in summary.get("by_operation_model", []):
        prefix = "embedding" if row.get("model_kind") == "embedding" else "generator"
        total += (
            int(row.get("input_tokens", 0)) * float(provider.get(f"{prefix}_input_per_million", 0))
            + int(row.get("output_tokens", 0)) * float(provider.get(f"{prefix}_output_per_million", 0))
        ) / 1_000_000
    return total
