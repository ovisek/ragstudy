from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .config import Config
from .io import read_jsonl, write_json


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _bootstrap_ci(values: list[float], seed: int, iterations: int = 10_000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iterations))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _bh(rows: list[dict[str, Any]]) -> None:
    valid = sorted((row for row in rows if row.get("p_value") is not None), key=lambda row: row["p_value"])
    total = len(valid)
    adjusted = 1.0
    for index in range(total - 1, -1, -1):
        raw = valid[index]["p_value"]
        adjusted = min(adjusted, raw * total / (index + 1))
        valid[index]["p_value_bh"] = adjusted


def analyse(config: Config) -> dict[str, int]:
    traces = [row for row in read_jsonl(config.output_root / "query_traces.jsonl") if row.get("status") == "ok"]
    metrics_path = config.output_root / "metric_results.jsonl"
    metrics = read_jsonl(metrics_path) if metrics_path.exists() else []
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for metric in metrics:
        if metric.get("coverage") == "available" and isinstance(metric.get("value"), (int, float)):
            grouped[(metric["dataset"], metric["system"], metric["metric"])].append(float(metric["value"]))
    seed = int(config.data["study"]["seed"])
    for (dataset, system, metric), values in sorted(grouped.items()):
        low, high = _bootstrap_ci(values, seed)
        summaries.append({"dataset": dataset, "system": system, "measure": metric, "n": len(values), "mean": mean(values), "median": median(values), "ci95_low": low, "ci95_high": high, "unit": "score"})
    timing: dict[tuple[str, str], list[float]] = defaultdict(list)
    costs: dict[tuple[str, str], list[float]] = defaultdict(list)
    calls: dict[tuple[str, str], list[float]] = defaultdict(list)
    for trace in traces:
        key = trace["dataset"], trace["system"]
        timing[key].append(float(trace["elapsed_seconds"]))
        costs[key].append(float(trace.get("metadata", {}).get("estimated_generator_cost_usd", 0)))
        calls[key].append(float(trace.get("usage", {}).get("api_calls", 0)))
    for key, values in sorted(timing.items()):
        latency_low, latency_high = _bootstrap_ci(values, seed)
        cost_low, cost_high = _bootstrap_ci(costs[key], seed)
        calls_low, calls_high = _bootstrap_ci(calls[key], seed)
        summaries.extend([
            {"dataset": key[0], "system": key[1], "measure": "query_latency_seconds", "n": len(values), "mean": mean(values), "median": median(values), "ci95_low": latency_low, "ci95_high": latency_high, "p95": _quantile(values, .95), "unit": "seconds"},
            {"dataset": key[0], "system": key[1], "measure": "generator_cost_usd", "n": len(costs[key]), "mean": mean(costs[key]), "median": median(costs[key]), "ci95_low": cost_low, "ci95_high": cost_high, "unit": "USD"},
            {"dataset": key[0], "system": key[1], "measure": "api_calls", "n": len(calls[key]), "mean": mean(calls[key]), "median": median(calls[key]), "ci95_low": calls_low, "ci95_high": calls_high, "unit": "calls"},
        ])
    output = config.output_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "summary.csv", summaries)
    comparisons = paired_comparisons(config, metrics)
    _write_csv(output / "paired_comparisons.csv", comparisons)
    write_json(output / "analysis_manifest.json", {"bootstrap_iterations": 10000, "seed": seed, "alpha": 0.05, "trace_rows": len(traces), "metric_rows": len(metrics)})
    return {"summary_rows": len(summaries), "comparison_rows": len(comparisons)}


def paired_comparisons(config: Config, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in metrics:
        if row.get("coverage") == "available" and isinstance(row.get("value"), (int, float)):
            lookup[(row["dataset"], row["metric"], row["question_id"])][row["system"]] = float(row["value"])
    rows = []
    systems = list(config.data["systems"])
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    for first_index, first in enumerate(systems):
        for second in systems[first_index + 1 :]:
            grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
            for (dataset, metric, _), scores in lookup.items():
                if first in scores and second in scores:
                    grouped[(dataset, metric)].append(scores[first] - scores[second])
            for (dataset, metric), differences in grouped.items():
                p_value = None
                if wilcoxon and any(value != 0 for value in differences):
                    p_value = float(wilcoxon(differences, zero_method="wilcox").pvalue)
                rows.append({"dataset": dataset, "metric": metric, "system_a": first, "system_b": second, "n_paired": len(differences), "mean_difference": mean(differences), "p_value": p_value, "test": "paired_wilcoxon" if wilcoxon else "scipy_not_installed"})
    _bh(rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
