#!/usr/bin/env python3
"""Build the deterministic GraphRAG-Bench Medical RQ3 micro-corpus.

The source prepared corpus is read-only input. This script writes a separate
9+1 logical-document partition with passage-level provenance.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "prepared" / "graphrag_bench_medical"
OUTPUT = ROOT / "data" / "prepared" / "graphrag_bench_medical_rq3_micro"
ENCODING = tiktoken.get_encoding("cl100k_base")
HELD_OUT_ID = "medical-rq3-micro::update::bcc"
DISTRACTOR_PLAN = (
    ("medical::source-chunk::01", 0.08),
    ("medical::source-chunk::01", 0.58),
    ("medical::source-chunk::02", 0.12),
    ("medical::source-chunk::02", 0.62),
    ("medical::source-chunk::03", 0.16),
    ("medical::source-chunk::03", 0.66),
    ("medical::source-chunk::04", 0.20),
    ("medical::source-chunk::04", 0.70),
    ("medical::source-chunk::05", 0.40),
)
FORBIDDEN_INITIAL_PHRASES = (
    "basal cell carcinoma",
    "most common type of skin cancer",
    "areas exposed to the sun",
    "lower part of the epidermis",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def passage(text: str, fraction: float, *, target_tokens: int = 850) -> tuple[str, int, int]:
    spans = sentence_spans(text)
    target_character = int(len(text) * fraction)
    start_index = min(range(len(spans)), key=lambda index: abs(spans[index][0] - target_character))
    start = spans[start_index][0]
    end = spans[start_index][1]
    for _, candidate_end in spans[start_index + 1:]:
        candidate = text[start:candidate_end].strip()
        if token_count(candidate) > 1100:
            break
        end = candidate_end
        if token_count(candidate) >= target_tokens:
            break
    selected = text[start:end].strip()
    count = token_count(selected)
    if not 600 <= count <= 1200:
        raise RuntimeError(f"Selected passage has {count} tokens; expected 600-1200")
    return selected, start, end


def document_record(*, new_id: str, source: dict, text: str, start: int, end: int, role: str) -> dict:
    return {
        "id": new_id,
        "title": f"GraphRAG-Bench Medical RQ3 micro-corpus: {role}",
        "text": text,
        "metadata": {
            "experiment": "rq3_micro",
            "partition": role,
            "source_dataset": "graphrag_bench_medical",
            "source_document_id": source["id"],
            "source_char_start": start,
            "source_char_end": end,
            "source_document_sha256": sha256_bytes(source["text"].encode("utf-8")),
            "passage_sha256": sha256_bytes(text.encode("utf-8")),
            "tokenizer": "cl100k_base",
            "token_count": token_count(text),
            "selection_method": "deterministic sentence-aligned passage",
        },
    }


def main() -> int:
    sources = {row["id"]: row for row in read_jsonl(SOURCE / "documents.jsonl")}
    questions = read_jsonl(SOURCE / "update_set.jsonl")
    initial: list[dict] = []
    for position, (source_id, fraction) in enumerate(DISTRACTOR_PLAN, 1):
        source = sources[source_id]
        text, start, end = passage(source["text"], fraction)
        normalized = " ".join(text.lower().split())
        present = [phrase for phrase in FORBIDDEN_INITIAL_PHRASES if phrase in normalized]
        if present:
            raise RuntimeError(f"Initial passage {position} contains held-out evidence phrase(s): {present}")
        initial.append(document_record(
            new_id=f"medical-rq3-micro::initial::{position:02d}", source=source,
            text=text, start=start, end=end, role="initial",
        ))

    held_source = sources["medical::source-chunk::00"]
    held_text, held_start, held_end = passage(held_source["text"], 0.0, target_tokens=900)
    added = [document_record(
        new_id=HELD_OUT_ID, source=held_source, text=held_text,
        start=held_start, end=held_end, role="held_out_update",
    )]
    for row in questions:
        row["evidence_document_ids"] = [HELD_OUT_ID]
        row.setdefault("strata", {})["experiment"] = "rq3_micro_update_sensitive"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_documents = initial + added
    for name, records in (
        ("documents.jsonl", all_documents),
        ("source_documents.jsonl", all_documents),
        ("update_initial_documents.jsonl", initial),
        ("update_added_documents.jsonl", added),
        ("update_set.jsonl", questions),
        ("pilot_set.jsonl", questions),
        ("evaluation_set.jsonl", questions),
        ("source_questions.jsonl", questions),
    ):
        write_jsonl(OUTPUT / name, records)

    split = {
        "dataset": "graphrag_bench_medical_rq3_micro",
        "initial_documents": len(initial),
        "added_documents": len(added),
        "total_documents": len(all_documents),
        "realized_update_fraction": len(added) / len(all_documents),
        "selection_seed": 20260722,
        "selection_plan": [{"source_document_id": source_id, "fraction": fraction} for source_id, fraction in DISTRACTOR_PLAN],
    }
    (OUTPUT / "update_split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hashed_names = (
        "documents.jsonl", "source_documents.jsonl", "update_initial_documents.jsonl",
        "update_added_documents.jsonl", "update_set.jsonl", "pilot_set.jsonl",
        "evaluation_set.jsonl", "source_questions.jsonl", "update_split.json",
    )
    provenance = {
        "dataset": "graphrag_bench_medical_rq3_micro",
        "purpose": "bounded RQ3 feasibility experiment",
        "source_dataset": "graphrag_bench_medical",
        "source_url": "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench",
        "source_documents_sha256": sha256_bytes((SOURCE / "documents.jsonl").read_bytes()),
        "source_update_questions_sha256": sha256_bytes((SOURCE / "update_set.jsonl").read_bytes()),
        "preparation_script": "scripts/prepare_rq3_medical_micro.py",
        "initial_documents": 9,
        "added_documents": 1,
        "questions": 3,
        "files": {name: sha256_bytes((OUTPUT / name).read_bytes()) for name in hashed_names},
        "limitations": [
            "This is a deliberately bounded micro-corpus and does not measure full-corpus scalability.",
            "The three questions share one held-out update document and are not independent updates.",
            "Correct pre-update answers may reflect parametric model knowledge rather than retrieval.",
        ],
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "initial": 9, "added": 1, "questions": 3}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
