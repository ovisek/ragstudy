#!/usr/bin/env python3
"""Validate the frozen GraphRAG-Bench Medical RQ3 inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prepared" / "graphrag_bench_medical"
OUTPUT = ROOT / "results" / "rq3_medical_input_validation.json"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(name: str) -> str:
    return hashlib.sha256((DATA / name).read_bytes()).hexdigest()


def contains(records: list[dict], phrase: str) -> bool:
    phrase = " ".join(phrase.lower().split())
    return any(phrase in " ".join(str(row.get("text", "")).lower().split()) for row in records)


def main() -> int:
    initial = rows("update_initial_documents.jsonl")
    added = rows("update_added_documents.jsonl")
    prepared = rows("documents.jsonl")
    questions = rows("update_set.jsonl")
    split = json.loads((DATA / "update_split.json").read_text(encoding="utf-8"))
    initial_ids = {row["id"] for row in initial}
    added_ids = {row["id"] for row in added}
    prepared_ids = {row["id"] for row in prepared}
    phrase_checks = {
        "Medical-422500d5": ["areas exposed to the sun", "face, head"],
        "Medical-73586ddc": ["most common type of skin cancer"],
        "Medical-a8bad1cf": ["basal cells", "lower part of the epidermis"],
    }
    checks = {
        "initial_count_is_5": len(initial) == 5,
        "added_count_is_1": len(added) == 1,
        "question_count_is_3": len(questions) == 3,
        "partition_ids_are_disjoint": not bool(initial_ids & added_ids),
        "partitions_reconstruct_prepared_corpus": initial_ids | added_ids == prepared_ids,
        "split_counts_match": split.get("initial_documents") == len(initial) and split.get("added_documents") == len(added),
        "question_ids_are_unique": len({row["id"] for row in questions}) == len(questions),
        "all_evidence_ids_are_added_only": all(
            set(row.get("evidence_document_ids", [])) <= added_ids and bool(row.get("evidence_document_ids"))
            for row in questions
        ),
    }
    evidence_checks = {}
    for question_id, phrases in phrase_checks.items():
        evidence_checks[question_id] = {
            phrase: {"present_in_added": contains(added, phrase), "present_in_initial": contains(initial, phrase)}
            for phrase in phrases
        }
    checks["defining_phrases_added_only"] = all(
        item["present_in_added"] and not item["present_in_initial"]
        for question in evidence_checks.values() for item in question.values()
    )
    report = {
        "dataset": "graphrag_bench_medical",
        "grain": "one deterministic corpus split with three matched update-sensitive questions",
        "initial_documents": len(initial),
        "added_documents": len(added),
        "total_documents": len(prepared),
        "realized_update_fraction": len(added) / len(prepared),
        "question_count": len(questions),
        "initial_document_ids": sorted(initial_ids),
        "added_document_ids": sorted(added_ids),
        "hashes": {name: digest(name) for name in (
            "documents.jsonl", "update_initial_documents.jsonl", "update_added_documents.jsonl",
            "update_split.json", "update_set.jsonl",
        )},
        "checks": checks,
        "evidence_phrase_checks": evidence_checks,
        "valid": all(checks.values()),
        "limitations": [
            "The realized update is one of six documents (16.7%), not an exact 10% partition.",
            "The three questions share one added source document and are not independent corpus updates.",
            "Correct pre-update answers may arise from model parametric knowledge rather than retrieval.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
