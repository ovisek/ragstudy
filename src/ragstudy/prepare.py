from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Config
from .io import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl


def _validate_document(row: dict[str, Any], row_number: int) -> None:
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        raise ValueError(f"Document row {row_number} requires non-empty string id")
    if not isinstance(row.get("text"), str) or not row["text"].strip():
        raise ValueError(f"Document {row['id']} requires non-empty text")


def _validate_question(row: dict[str, Any], row_number: int) -> None:
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        raise ValueError(f"Question row {row_number} requires non-empty string id")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError(f"Question {row['id']} requires non-empty question")


def _stratum(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    values = row.get("strata") or {}
    if not isinstance(values, dict):
        raise ValueError(f"Question {row.get('id')} has non-object strata")
    return tuple(sorted((str(key), str(value)) for key, value in values.items())) or (("all", "all"),)


def stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Allocate as evenly as possible across declared strata, deterministically."""
    if n >= len(rows):
        return sorted(rows, key=lambda item: item["id"])
    groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_stratum(row)].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        group.sort(key=lambda item: item["id"])
        rng.shuffle(group)
    group_keys = sorted(groups)
    allocation = {key: 0 for key in group_keys}
    # Round-robin preserves representation; surplus follows seed-stable order.
    while sum(allocation.values()) < n:
        candidates = [key for key in group_keys if allocation[key] < len(groups[key])]
        if not candidates:
            break
        smallest = min(allocation[key] / len(groups[key]) for key in candidates)
        tied = [key for key in candidates if allocation[key] / len(groups[key]) == smallest]
        allocation[rng.choice(tied)] += 1
    selected = [row for key in group_keys for row in groups[key][: allocation[key]]]
    return sorted(selected, key=lambda item: item["id"])


def prepare_dataset(config: Config, dataset_name: str, documents_path: Path, questions_path: Path) -> dict[str, Any]:
    definition = config.dataset(dataset_name)
    documents = read_jsonl(documents_path)
    questions = read_jsonl(questions_path)
    for index, row in enumerate(documents, 1):
        _validate_document(row, index)
    for index, row in enumerate(questions, 1):
        _validate_question(row, index)
    if len({row["id"] for row in documents}) != len(documents):
        raise ValueError("Document IDs must be unique before de-duplication")
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")

    kept: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for row in sorted(documents, key=lambda item: item["id"]):
        text_hash = sha256_text(row["text"])
        if text_hash in seen_text:
            continue
        seen_text.add(text_hash)
        kept.append({**row, "text_sha256": text_hash})

    seed = int(config.data["study"]["seed"])
    full_sample = stratified_sample(questions, int(definition["sample_size"]), seed)
    pilot = stratified_sample(full_sample, min(int(definition["pilot_size"]), len(full_sample)), seed + 1)
    dataset_dir = config.data_root / "prepared" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Copy sources first so raw-to-prepared lineage remains inspectable.
    shutil.copy2(documents_path, dataset_dir / "source_documents.jsonl")
    shutil.copy2(questions_path, dataset_dir / "source_questions.jsonl")
    write_jsonl(dataset_dir / "documents.jsonl", kept)
    write_jsonl(dataset_dir / "evaluation_set.jsonl", full_sample)
    write_jsonl(dataset_dir / "pilot_set.jsonl", pilot)
    strata = Counter(" | ".join(f"{key}={value}" for key, value in _stratum(row)) for row in full_sample)
    report = {
        "dataset": dataset_name,
        "source_url": definition["source_url"],
        "licence": definition["licence"],
        "documents_source_sha256": sha256_file(documents_path),
        "questions_source_sha256": sha256_file(questions_path),
        "document_count_source": len(documents),
        "document_count_deduplicated": len(kept),
        "question_count_source": len(questions),
        "evaluation_count": len(full_sample),
        "pilot_count": len(pilot),
        "evaluation_strata": dict(sorted(strata.items())),
        "selection_seed": seed,
    }
    write_json(dataset_dir / "provenance.json", report)
    return report


def make_update_split(config: Config, dataset_name: str) -> dict[str, Any]:
    dataset_dir = config.data_root / "prepared" / dataset_name
    documents = read_jsonl(dataset_dir / "documents.jsonl")
    seed = int(config.data["study"]["seed"])
    rng = random.Random(seed + 9)
    documents = sorted(documents, key=lambda item: item["id"])
    rng.shuffle(documents)
    cut = max(1, round(len(documents) * 0.9))
    initial, update = documents[:cut], documents[cut:]
    write_jsonl(dataset_dir / "update_initial_documents.jsonl", initial)
    write_jsonl(dataset_dir / "update_added_documents.jsonl", update)
    report = {"dataset": dataset_name, "initial_documents": len(initial), "added_documents": len(update), "seed": seed + 9}
    write_json(dataset_dir / "update_split.json", report)
    return report

