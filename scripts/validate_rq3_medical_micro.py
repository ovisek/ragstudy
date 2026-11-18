#!/usr/bin/env python3
"""Validate the frozen GraphRAG-Bench Medical RQ3 micro-experiment inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prepared" / "graphrag_bench_medical_rq3_micro"
SOURCE = ROOT / "data" / "prepared" / "graphrag_bench_medical" / "documents.jsonl"
OUTPUT = ROOT / "results" / "rq3_medical_micro_input_validation.json"
ENCODING = tiktoken.get_encoding("cl100k_base")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_contains(records: list[dict], phrase: str) -> bool:
    needle = " ".join(phrase.lower().split())
    return any(needle in " ".join(row["text"].lower().split()) for row in records)


def main() -> int:
    initial = rows(DATA / "update_initial_documents.jsonl")
    added = rows(DATA / "update_added_documents.jsonl")
    prepared = rows(DATA / "documents.jsonl")
    questions = rows(DATA / "update_set.jsonl")
    split = json.loads((DATA / "update_split.json").read_text(encoding="utf-8"))
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    source_by_id = {row["id"]: row for row in rows(SOURCE)}
    initial_ids = {row["id"] for row in initial}
    added_ids = {row["id"] for row in added}
    prepared_ids = {row["id"] for row in prepared}

    provenance_valid = True
    passage_checks: dict[str, dict] = {}
    for row in prepared:
        metadata = row.get("metadata", {})
        source = source_by_id.get(metadata.get("source_document_id"))
        start, end = metadata.get("source_char_start"), metadata.get("source_char_end")
        source_slice = source["text"][start:end].strip() if source and isinstance(start, int) and isinstance(end, int) else None
        count = len(ENCODING.encode(row["text"]))
        valid = bool(
            source
            and source_slice == row["text"]
            and metadata.get("source_document_sha256") == hashlib.sha256(source["text"].encode()).hexdigest()
            and metadata.get("passage_sha256") == hashlib.sha256(row["text"].encode()).hexdigest()
            and metadata.get("token_count") == count
            and 600 <= count <= 1200
        )
        provenance_valid &= valid
        passage_checks[row["id"]] = {"token_count": count, "source_document_id": metadata.get("source_document_id"), "valid": valid}

    evidence_phrases = (
        "basal cell carcinoma (bcc), is the most common type of skin cancer",
        "areas exposed to the sun, including the face, head, neck",
        "basal cells – found in the lower part of the epidermis",
    )
    checks = {
        "initial_count_is_9": len(initial) == 9,
        "added_count_is_1": len(added) == 1,
        "total_count_is_10": len(prepared) == 10,
        "question_count_is_3": len(questions) == 3,
        "partition_ids_are_disjoint": not bool(initial_ids & added_ids),
        "partitions_reconstruct_prepared_corpus": initial_ids | added_ids == prepared_ids,
        "split_counts_match": split.get("initial_documents") == 9 and split.get("added_documents") == 1,
        "update_fraction_is_exactly_10_percent": len(added) / len(prepared) == 0.1,
        "question_ids_are_unique": len({row["id"] for row in questions}) == 3,
        "all_evidence_ids_are_added_only": all(set(row.get("evidence_document_ids", [])) == added_ids for row in questions),
        "passage_provenance_and_token_bounds_valid": provenance_valid,
        "defining_evidence_present_only_after_update": all(
            normalized_contains(added, phrase) and not normalized_contains(initial, phrase)
            for phrase in evidence_phrases
        ),
        "prepared_file_hashes_match_provenance": all(
            digest(DATA / name) == expected for name, expected in provenance.get("files", {}).items()
        ),
    }
    report = {
        "dataset": "graphrag_bench_medical_rq3_micro",
        "grain": "one deterministic 9+1 micro-corpus split with three matched update-sensitive questions",
        "initial_documents": len(initial),
        "added_documents": len(added),
        "total_documents": len(prepared),
        "realized_update_fraction": len(added) / len(prepared),
        "question_count": len(questions),
        "initial_document_ids": sorted(initial_ids),
        "added_document_ids": sorted(added_ids),
        "passages": passage_checks,
        "checks": checks,
        "valid": all(checks.values()),
        "limitations": provenance.get("limitations", []),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
