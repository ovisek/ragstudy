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
from .pilot_collection import collect_pilot, parse_run_selections
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
    run_parser.add_argument("--resume", action="store_true", help="Resume the latest compatible incomplete run")
    run_parser.add_argument("--allow-partial-index", action="store_true", help="Run queries against an incomplete index and mark their traces")
    run_parser.add_argument("--index-retries", default=2, type=int, help="Retries per failed index checkpoint unit (default: 2)")
    run_parser.add_argument("--concurrent-workers", type=int, help="Override worker concurrency for GraphRAG and LightRAG for this launch/resume")
    run_parser.add_argument("--requests-per-minute", type=int, help="Override the provider RPM budget for this launch/resume")
    collect = commands.add_parser("collect-pilot", help="Collect exact independent pilot runs into an immutable comparative report")
    collect.add_argument("--dataset", required=True)
    collect.add_argument("--run", action="append", required=True, metavar="SYSTEM=RUN_ID", help="Exact single-system pilot run; repeat for naive, graphrag, and lightrag")
    collect.add_argument("--allow-short-pilot", action="store_true", help="Explicitly accept fewer prepared questions than datasets.<name>.pilot_size")
    update = commands.add_parser("run-update", help="Run auditable initial/pre-query/update/post-query workflow")
    update.add_argument("--dataset", action="append", help="Configured dataset to run; repeat to select more than one")
    update.add_argument("--systems", default="graphrag,lightrag")
    update.add_argument("--index-retries", default=2, type=int, help="Retries per failed index checkpoint unit (default: 2)")
    update.add_argument("--concurrent-workers", type=int, help="Override worker concurrency for this update run")
    update.add_argument("--requests-per-minute", type=int, help="Override the provider RPM budget for this update run")
    update.add_argument("--resume-run-id", help="Resume this exact update run and reuse completed indexes, caches, and traces")
    score = commands.add_parser("score", help="Export evaluator input, compute deterministic proxies, or import scores")
    score.add_argument("--input", default=None, type=Path, help="Trace JSONL; defaults to results/query_traces.jsonl")
    score.add_argument("--import-metrics", default=None, type=Path)
    score.add_argument("--backend", default="external")
    score_mode = score.add_mutually_exclusive_group()
    score_mode.add_argument("--export-only", action="store_true", help="Export evaluator inputs without writing metric results")
    score_mode.add_argument("--import-only", action="store_true", help="Import metrics without re-exporting or recomputing deterministic proxies")
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


def apply_run_overrides(config, args) -> None:
    if args.requests_per_minute is not None:
        config.data["provider"]["requests_per_minute"] = args.requests_per_minute
    if args.concurrent_workers is not None:
        config.data["common"]["concurrent_workers"] = args.concurrent_workers


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
    if args.command == "collect-pilot":
        try:
            selections = parse_run_selections(args.run)
            result = collect_pilot(config, args.dataset, selections, allow_short_pilot=args.allow_short_pilot)
        except (RuntimeError, ValueError) as exc:
            print(f"Pilot collection blocked: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0
    if args.command in {"run", "run-update"}:
        systems = _systems(args.systems)
        apply_run_overrides(config, args)
        missing = [name for name in systems if name not in config.data["systems"]]
        errors = readiness(config) + [f"Unknown system: {item}" for item in missing] + validate_run_config(config, systems)
        if errors:
            print("Run blocked:\n" + "\n".join(f"- {item}" for item in errors), file=sys.stderr)
            return 2
        if args.command == "run":
            try:
                result = run(config, args.phase, systems, args.repetition, resume=args.resume, allow_partial_index=args.allow_partial_index, index_retries=args.index_retries)
            except KeyboardInterrupt:
                print("Run interrupted; resume with the same command plus --resume.", file=sys.stderr)
                return 130
            except (RuntimeError, ValueError) as exc:
                print(f"Run failed: {exc}", file=sys.stderr)
                return 1
        else:
            try:
                result = run_update(
                    config,
                    systems,
                    datasets=args.dataset,
                    index_retries=args.index_retries,
                    resume_run_id=args.resume_run_id,
                )
            except KeyboardInterrupt:
                print("Update run interrupted; its manifest and completed records were preserved.", file=sys.stderr)
                return 130
            except (RuntimeError, ValueError) as exc:
                print(f"Update run failed: {exc}", file=sys.stderr)
                return 1
        print(json.dumps(result, indent=2))
        return 0 if not isinstance(result, dict) or result.get("complete", True) else 1
    if args.command == "score":
        if args.import_only:
            if not args.import_metrics:
                print("--import-only requires --import-metrics", file=sys.stderr)
                return 2
            try:
                imported = import_metrics(config, args.import_metrics, args.backend)
            except ValueError as exc:
                print(f"Metric import blocked: {exc}", file=sys.stderr)
                return 1
            print(json.dumps({"imported_metric_rows": imported}, indent=2))
            return 0
        if args.export_only and args.import_metrics:
            print("--export-only cannot be combined with --import-metrics", file=sys.stderr)
            return 2
        source = args.input or (config.output_root / "query_traces.jsonl")
        if not source.exists():
            print(f"Trace input missing: {source}", file=sys.stderr)
            return 2
        result = export_evaluation_inputs(config, source)
        if args.export_only:
            print(json.dumps(result, indent=2))
            return 0
        result.update(score_retrieval_contract(config, source))
        if args.import_metrics:
            try:
                result["imported_metric_rows"] = import_metrics(config, args.import_metrics, args.backend)
            except ValueError as exc:
                print(f"Metric import blocked: {exc}", file=sys.stderr)
                return 1
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
