#!/usr/bin/env python3
"""Create canonical document and question JSONL files from an UltraDomain split.

The source split has one row per question and repeats its source context.  This
script writes one document per ``context_id`` and one question per source row,
ready for ``ragstudy prepare``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _fail(line_number: int, message: str) -> ValueError:
    return ValueError(f"Source row {line_number}: {message}")


def _non_empty_string(row: Mapping[str, Any], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _fail(line_number, f"requires a non-empty string {field!r}")
    return value


def _read_source(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise _fail(line_number, "must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no JSONL rows")
    return rows


def convert(rows: list[dict[str, Any]], source_url: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return canonical documents, questions, and repeated-context count."""
    documents_by_id: dict[str, dict[str, Any]] = {}
    question_ids: set[str] = set()
    questions: list[dict[str, Any]] = []
    repeated_contexts = 0

    for line_number, row in enumerate(rows, 1):
        question_id = _non_empty_string(row, "_id", line_number)
        context_id = _non_empty_string(row, "context_id", line_number)
        question = _non_empty_string(row, "input", line_number)
        context = _non_empty_string(row, "context", line_number)
        label = _non_empty_string(row, "label", line_number)
        if question_id in question_ids:
            raise _fail(line_number, f"duplicates question ID {question_id!r}")
        question_ids.add(question_id)

        answers = row.get("answers")
        if not isinstance(answers, list) or not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise _fail(line_number, "requires a non-empty list of non-empty string 'answers'")

        meta = row.get("meta", {})
        if not isinstance(meta, Mapping):
            raise _fail(line_number, "field 'meta' must be an object when present")
        document = {
            "id": context_id,
            "text": context,
            "title": meta.get("title"),
            "metadata": {
                "source": source_url,
                "source_context_id": context_id,
                "authors": meta.get("authors"),
                "domain": label,
            },
        }
        existing = documents_by_id.get(context_id)
        if existing is None:
            documents_by_id[context_id] = document
        elif existing["text"] != context:
            raise _fail(line_number, f"context ID {context_id!r} is associated with different text")
        else:
            repeated_contexts += 1

        questions.append({
            "id": question_id,
            "question": question,
            "reference_answer": answers,
            "evidence": [context_id],
            "strata": {"domain": label},
        })

    return sorted(documents_by_id.values(), key=lambda item: item["id"]), sorted(questions, key=lambda item: item["id"]), repeated_contexts


def subset(
    documents: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    max_documents: int | None,
    max_questions: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a deterministic, internally consistent canonical subset.

    Documents are selected by stable ID. Questions are then selected by stable
    ID only when every evidence document is present in that document subset.
    """
    selected_documents = documents[:max_documents] if max_documents is not None else documents
    document_ids = {document["id"] for document in selected_documents}
    compatible_questions = [
        question for question in questions
        if set(question["evidence"]).issubset(document_ids)
    ]
    selected_questions = compatible_questions[:max_questions] if max_questions is not None else compatible_questions
    if max_questions is not None and len(selected_questions) < max_questions:
        raise ValueError(
            f"only {len(selected_questions)} questions have evidence in the selected documents; "
            f"cannot create requested {max_questions}-question subset"
        )
    return selected_documents, selected_questions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UltraDomain domain JSONL split")
    parser.add_argument("--domain", required=True, help="Output prefix, for example 'agriculture'")
    parser.add_argument("--output-dir", type=Path, help="Directory for the canonical files (defaults to the input directory)")
    parser.add_argument("--source-url", default="https://huggingface.co/datasets/TommyChien/UltraDomain")
    parser.add_argument("--max-documents", type=int, help="Keep at most this many documents, ordered by stable document ID")
    parser.add_argument("--max-questions", type=int, help="Keep at most this many compatible questions, ordered by stable question ID")
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    args = parser.parse_args()

    if not args.domain or not args.domain.strip() or "/" in args.domain or "\\" in args.domain:
        parser.error("--domain must be a non-empty filename prefix without path separators")
    if not args.input.is_file():
        parser.error(f"--input is not a file: {args.input}")
    for option, value in (("--max-documents", args.max_documents), ("--max-questions", args.max_questions)):
        if value is not None and value < 1:
            parser.error(f"{option} must be at least 1")

    output_dir = args.output_dir or args.input.parent
    documents_path = output_dir / f"{args.domain}_documents.jsonl"
    questions_path = output_dir / f"{args.domain}_questions.jsonl"
    existing = [path for path in (documents_path, questions_path) if path.exists()]
    if existing and not args.force:
        parser.error("refusing to replace existing file(s): " + ", ".join(str(path) for path in existing) + "; use --force to replace them")

    try:
        documents, questions, repeated_contexts = convert(_read_source(args.input), args.source_url)
        documents, questions = subset(documents, questions, args.max_documents, args.max_questions)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(documents_path, documents)
        _write_jsonl(questions_path, questions)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(documents)} documents to {documents_path}")
    print(f"Wrote {len(questions)} questions to {questions_path}")
    print(f"Merged {repeated_contexts} repeated source-context rows by context_id.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
