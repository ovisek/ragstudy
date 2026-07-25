#!/usr/bin/env python3
"""Bridge the official Microsoft GraphRAG CLI to ragstudy's JSON contract.

Pinned against graphrag==3.1.0. The bridge creates a self-contained GraphRAG
workspace beneath the harness run directory and never writes an API key to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def config(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Command failed:\n" + " ".join(command) + "\n" + result.stderr[-4000:])
    return result


def environment(study: dict) -> dict[str, str]:
    provider = study["provider"]
    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing {provider['api_key_env']}")
    env = os.environ.copy()
    # GraphRAG uses its own environment-variable name; keep the source key out
    # of .env files and pass it only to the child CLI process.
    env["GRAPHRAG_API_KEY"] = api_key
    env["GRAPHRAG_LLM_API_KEY"] = api_key
    ca_bundle = provider.get("ca_bundle")
    if ca_bundle:
        ca_path = Path(ca_bundle).expanduser()
        if not ca_path.is_file():
            raise RuntimeError(f"Configured provider.ca_bundle is not a readable file: {ca_path}")
        env["SSL_CERT_FILE"] = str(ca_path)
        env["REQUESTS_CA_BUNDLE"] = str(ca_path)
    return env


def write_inputs(path: Path, source: Path) -> int:
    input_dir = path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in (json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", row["id"])
        # Prefix metadata makes the GraphRAG text-unit lineage auditable.
        header = f"Document ID: {row['id']}\nTitle: {row.get('title') or ''}\n\n"
        (input_dir / f"{safe_id}.txt").write_text(header + row["text"], encoding="utf-8")
        count += 1
    return count


def configure_settings(workspace: Path, study: dict) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("GraphRAG installation must include PyYAML") from exc
    path = workspace / "settings.yaml"
    settings = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = settings.setdefault("models", {})
    provider, common = study["provider"], study["common"]
    chat = models.setdefault("default_chat_model", {})
    chat.update({"type": "openai_chat", "model": provider["generator_model"], "api_key": "${GRAPHRAG_API_KEY}", "api_base": provider["base_url"], "encoding_model": "cl100k_base"})
    embedding = models.setdefault("default_embedding_model", {})
    embedding.update({"type": "openai_embedding", "model": provider["embedding_model"], "api_key": "${GRAPHRAG_API_KEY}", "api_base": provider["base_url"], "encoding_model": "cl100k_base"})
    # The v3 generated defaults own the rest of the workflow configuration.
    chunks = settings.setdefault("chunks", {})
    chunks["size"] = int(common["chunk_tokens"])
    chunks["overlap"] = int(common["chunk_overlap_tokens"])
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")


def index(args: argparse.Namespace, study: dict) -> None:
    workspace, source, output = Path(args.workspace), Path(args.input), Path(args.output)
    workspace.mkdir(parents=True, exist_ok=True)
    env = environment(study)
    if not (workspace / "settings.yaml").exists():
        run(["graphrag", "init", "--root", str(workspace), "--force"], env)
        configure_settings(workspace, study)
    documents = write_inputs(workspace, source)
    completed = run(["graphrag", "index", "--root", str(workspace), "--method", "standard"], env)
    artifacts = list((workspace / "output").rglob("*.parquet")) if (workspace / "output").exists() else []
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"implementation": "microsoft/graphrag", "documents": documents, "parquet_artifacts": len(artifacts), "stdout_tail": completed.stdout[-2000:]}, indent=2) + "\n", encoding="utf-8")


def query(args: argparse.Namespace, study: dict) -> None:
    workspace, output = Path(args.workspace), Path(args.output)
    env = environment(study)
    method = {"hybrid": "local", "local": "local", "global": "global"}.get(args.method, "local")
    completed = run(["graphrag", "query", "--root", str(workspace), "--method", method, "--query", args.question], env)
    answer = completed.stdout.strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"answer": answer, "contexts": [], "usage": {}, "metadata": {"implementation": "microsoft/graphrag==3.1.0", "requested_method": args.method, "executed_method": method, "context_trace": "unavailable_from_official_CLI"}}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("index", "query"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--question")
    parser.add_argument("--method", default="local")
    args = parser.parse_args()
    study = config(Path(args.config))
    if args.action == "index":
        if not args.input:
            parser.error("index requires --input")
        index(args, study)
    else:
        if not args.question:
            parser.error("query requires --question")
        query(args, study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

