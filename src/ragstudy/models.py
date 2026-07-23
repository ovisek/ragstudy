from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    reference_answer: str | list[str] | None = None
    evidence: list[str] = field(default_factory=list)
    strata: dict[str, str] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    retries: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.api_calls += other.api_calls
        self.retries += other.retries


@dataclass
class Trace:
    run_id: str
    phase: str
    dataset: str
    system: str
    question_id: str
    question: str
    answer: str | None
    contexts: list[dict[str, Any]]
    started_at: str
    elapsed_seconds: float
    usage: Usage
    status: str
    error: str | None = None
    repetition: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        item = asdict(self)
        item["usage"] = asdict(self.usage)
        return item

