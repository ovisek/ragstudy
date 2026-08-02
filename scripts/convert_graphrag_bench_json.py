#!/usr/bin/env python3
"""Create canonical JSONL inputs from the released GraphRAG-Bench corpus and questions.

The Medical release contains one large published corpus context.  To support a
small, inspectable sample, this script partitions that original context into a
requested number of contiguous, whitespace-aligned documents.  It never uses
answers or evidence to construct document text.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SOURCE_URL = "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench"


def _non_empty_string(row: Mapping[str, Any], field: str, description: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} requires non-empty string {field!r}")
    return value


def _read_array(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{description} at {path} must be a non-empty JSON array")
    if any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"{description} at {path} must contain only JSON objects")
    return payload


def split_source_text(text: str, count: int) -> list[tuple[int, int, str]]:
    """Partition text into contiguous, whitespace-aligned chunks without loss."""
    if count < 1:
        raise ValueError("document count must be at least 1")
    if count > len(text):
        raise ValueError("document count cannot exceed source text characters")
    boundaries = [0]
    for index in range(1, count):
        target = round(len(text) * index / count)
        boundary = target
        while boundary < len(text) and not text[boundary].isspace():
            boundary += 1
        boundaries.append(boundary)
    boundaries.append(len(text))
    return [(start, end, text[start:end]) for start, end in zip(boundaries, boundaries[1:])]


def convert(
    corpus_rows: list[dict[str, Any]], questions_rows: list[dict[str, Any]], document_count: int, question_count: int, source_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert the single released corpus context plus published questions."""
    if len(corpus_rows) != 1:
        raise ValueError(f"expected one released corpus record, received {len(corpus_rows)}")
    corpus = corpus_rows[0]
    corpus_name = _non_empty_string(corpus, "corpus_name", "Corpus record")
    text = _non_empty_string(corpus, "context", "Corpus record")
    chunks = split_source_text(text, document_count)
    documents = [
        {
            "id": f"{corpus_name.lower()}::source-chunk::{index:02d}",
            "text": chunk,
            "title": corpus_name,
            "metadata": {
                "source": source_url,
                "corpus_name": corpus_name,
                "source_record_index": 0,
                "character_start": start,
                "character_end": end,
            },
        }
        for index, (start, end, chunk) in enumerate(chunks)
    ]

    if question_count < 1:
        raise ValueError("question count must be at least 1")
    if len(questions_rows) < question_count:
        raise ValueError(f"only {len(questions_rows)} published questions available; cannot select {question_count}")
    questions: list[dict[str, Any]] = []
    question_ids: set[str] = set()
    for index, row in enumerate(questions_rows[:question_count], 1):
        question_id = _non_empty_string(row, "id", f"Question {index}")
        if question_id in question_ids:
            raise ValueError(f"Question {index} duplicates published ID {question_id!r}")
        question_ids.add(question_id)
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence or any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ValueError(f"Question {question_id!r} requires a non-empty string-list 'evidence'")
        converted = {
            "id": question_id,
            "question": _non_empty_string(row, "question", f"Question {question_id!r}"),
            "reference_answer": _non_empty_string(row, "answer", f"Question {question_id!r}"),
            "evidence": evidence,
            "strata": {"question_type": _non_empty_string(row, "question_type", f"Question {question_id!r}")},
        }
        if isinstance(row.get("evidence_relations"), str) and row["evidence_relations"].strip():
            converted["evidence_relations"] = row["evidence_relations"]
        questions.append(converted)
    return documents, questions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="Released GraphRAG-Bench corpus JSON")
    parser.add_argument("--questions", required=True, type=Path, help="Released GraphRAG-Bench questions JSON")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Output path prefix, for example data/raw/graphrag_bench_medical/medical_subset")
    parser.add_argument("--documents", type=int, default=6, help="Number of source-preserving corpus chunks (default: 6)")
    parser.add_argument("--question-count", type=int, default=3, help="Number of published questions to retain (default: 3)")
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    args = parser.parse_args()

    if args.documents < 1 or args.question_count < 1:
        parser.error("--documents and --question-count must be at least 1")
    if not args.corpus.is_file() or not args.questions.is_file():
        parser.error("--corpus and --questions must both be existing files")
    documents_path = args.output_prefix.parent / f"{args.output_prefix.name}_documents.jsonl"
    questions_path = args.output_prefix.parent / f"{args.output_prefix.name}_questions.jsonl"
    existing = [path for path in (documents_path, questions_path) if path.exists()]
    if existing and not args.force:
        parser.error("refusing to replace existing file(s): " + ", ".join(map(str, existing)) + "; use --force to replace them")

    try:
        documents, questions = convert(
            _read_array(args.corpus, "Corpus"), _read_array(args.questions, "Questions"), args.documents, args.question_count, args.source_url,
        )
        documents_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(documents_path, documents)
        _write_jsonl(questions_path, questions)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(documents)} documents to {documents_path}")
    print(f"Wrote {len(questions)} questions to {questions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
