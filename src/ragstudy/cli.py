from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .analysis import analyse
from .acquire import acquire
from .config import load_config, validate_run_config
from .evaluation import export_evaluation_inputs, import_metrics, score_retrieval_contract
from .manifest import freeze_manifest
from .prepare import make_update_split, prepare_dataset
from .runner import run, run_update
from .thesis import render_thesis_outputs


def _systems(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ragstudy", description="Auditable GraphRAG / LightRAG thesis experiment runner")
    root.add_argument("--config", default="config/experiment.toml", help="Path to frozen study TOML")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Freeze protocol and environment manifest")
    acquire_parser = commands.add_parser("acquire", help="Explicitly download one official dataset snapshot")
    acquire_parser.add_argument("--dataset", required=True)
    acquire_parser.add_argument("--confirm-download", action="store_true", help="Required: some sources are multi-gigabyte downloads")
    prepare = commands.add_parser("prepare", help="Validate, deduplicate, and sample canonical JSONL")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--documents", required=True, type=Path)
    prepare.add_argument("--questions", required=True, type=Path)
    commands.add_parser("validate", help="Check data gates before an API run")
    run_parser = commands.add_parser("run", help="Run pilot or full paired evaluation")
    run_parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    run_parser.add_argument("--systems", default="naive,graphrag,lightrag")
    run_parser.add_argument("--repetition", default=1, type=int)
    update = commands.add_parser("run-update", help="Run auditable 90/10 update workflow")
    update.add_argument("--systems", default="naive,graphrag,lightrag")
    score = commands.add_parser("score", help="Export evaluator input, compute deterministic proxies, or import scores")
    score.add_argument("--input", default=None, type=Path, help="Trace JSONL; defaults to results/query_traces.jsonl")
    score.add_argument("--import-metrics", default=None, type=Path)
    score.add_argument("--backend", default="external")
    commands.add_parser("analyse", help="Create bootstrapped summaries and paired tests")
    commands.add_parser("thesis", help="Generate a thesis-ready results draft from analysis")
    return root


def readiness(config) -> list[str]:
    errors: list[str] = []
    for dataset in config.data["datasets"]:
        directory = config.data_root / "prepared" / dataset
        for filename in ("documents.jsonl", "evaluation_set.jsonl", "pilot_set.jsonl", "provenance.json", "update_initial_documents.jsonl", "update_added_documents.jsonl"):
            if not (directory / filename).exists():
                errors.append(f"{dataset}: missing {directory / filename}")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "init":
        output = freeze_manifest(config)
        print(json.dumps({"manifest": str(config.output_root / "experiment_manifest.json"), "protocol_version": output["protocol_version"]}, indent=2))
        return 0
    if args.command == "acquire":
        if not args.confirm_download:
            print("Acquisition is intentionally opt-in. Re-run with --confirm-download.", file=sys.stderr)
            return 2
        if args.dataset not in config.data["datasets"]:
            print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
            return 2
        print(json.dumps(acquire(config, args.dataset), indent=2))
        return 0
    if args.command == "prepare":
        report = prepare_dataset(config, args.dataset, args.documents, args.questions)
        split = make_update_split(config, args.dataset)
        print(json.dumps({"prepare": report, "update_split": split}, indent=2))
        return 0
    if args.command == "validate":
        errors = readiness(config)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 2
        print("Data gates passed. Use `run` only after reviewing the manifest and provider pricing snapshot.")
        return 0
    if args.command in {"run", "run-update"}:
        systems = _systems(args.systems)
        missing = [name for name in systems if name not in config.data["systems"]]
        errors = readiness(config) + [f"Unknown system: {item}" for item in missing] + validate_run_config(config, systems)
        if errors:
            print("Run blocked:\n" + "\n".join(f"- {item}" for item in errors), file=sys.stderr)
            return 2
        if args.command == "run":
            result = run(config, args.phase, systems, args.repetition)
        else:
            result = run_update(config, systems)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "score":
        source = args.input or (config.output_root / "query_traces.jsonl")
        if not source.exists():
            print(f"Trace input missing: {source}", file=sys.stderr)
            return 2
        result = export_evaluation_inputs(config, source)
        result.update(score_retrieval_contract(config, source))
        if args.import_metrics:
            result["imported_metric_rows"] = import_metrics(config, args.import_metrics, args.backend)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "analyse":
        print(json.dumps(analyse(config), indent=2))
        return 0
    if args.command == "thesis":
        print(render_thesis_outputs(config))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
