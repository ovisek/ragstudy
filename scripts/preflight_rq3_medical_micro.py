#!/usr/bin/env python3
"""Run the non-measured provider and LightRAG preflight for the RQ3 micro-run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def normalized_keyword_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(str.maketrans({
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    }))
    return normalized.upper()


def usage_record(response: object) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def run_streamed(command: list[str], *, environment: dict[str, str]) -> tuple[int, float, str]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        captured.append(line)
        print(line, end="", flush=True)
    process.stdout.close()
    return process.wait(), time.monotonic() - started, "".join(captured)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/rq3_medical_micro.toml")
    parser.add_argument("--consecutive-calls", type=int, default=3)
    args = parser.parse_args()
    if args.consecutive_calls < 2:
        parser.error("--consecutive-calls must be at least 2")

    config_path = (ROOT / args.config).resolve()
    study = tomllib.loads(config_path.read_text(encoding="utf-8"))
    provider = study["provider"]
    load_dotenv(ROOT / ".env", override=False)
    key_name = provider["api_key_env"]
    api_key = os.getenv(key_name)
    if not api_key:
        print(f"[FAIL] {key_name} is unavailable", flush=True)
        return 1

    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    root = ROOT / "results" / "preflight" / f"rq3-medical-micro-{stamp}"
    workspace = root / "lightrag_workspace"
    root.mkdir(parents=True, exist_ok=False)
    report_path = root / "preflight_report.json"
    report: dict[str, object] = {
        "preflight_id": root.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "measured_experiment": False,
        "actual_api_expenditure_usd": 0.0,
        "generator_calls": [],
        "embedding_calls": [],
        "tiny_lightrag": {},
        "status": "running",
    }

    client = OpenAI(api_key=api_key, base_url=provider["base_url"], max_retries=0)
    interval = 60.0 / int(provider["requests_per_minute"])
    next_request = 0.0

    def pace() -> None:
        nonlocal next_request
        delay = next_request - time.monotonic()
        if delay > 0:
            print(f"Pacing provider requests for {delay:.1f}s", flush=True)
            time.sleep(delay)
        next_request = time.monotonic() + interval

    try:
        for position in range(1, args.consecutive_calls + 1):
            pace()
            print(f"[{position}/{args.consecutive_calls}] embedding endpoint", flush=True)
            started = time.monotonic()
            response = client.embeddings.create(
                model=provider["embedding_model"],
                input=[f"RQ3 micro-experiment preflight embedding {position}"],
                timeout=float(provider["embedding_timeout_seconds"]),
            )
            dimension = len(response.data[0].embedding)
            record = {
                "position": position,
                "status": "ok",
                "elapsed_seconds": time.monotonic() - started,
                "embedding_dimension": dimension,
                "usage": usage_record(response),
            }
            report["embedding_calls"].append(record)  # type: ignore[union-attr]
            if dimension != int(provider["embedding_dimension"]):
                raise RuntimeError(f"Embedding dimension {dimension} does not match configured {provider['embedding_dimension']}")

        for position in range(1, args.consecutive_calls + 1):
            pace()
            print(f"[{position}/{args.consecutive_calls}] generator endpoint", flush=True)
            started = time.monotonic()
            response = client.chat.completions.create(
                model=provider["generator_model"],
                messages=[{"role": "user", "content": "Do not explain. Output exactly the word OK."}],
                # GPT-OSS may use part of this allowance for reasoning before
                # emitting visible content; eight tokens can therefore yield
                # a valid but visibly empty completion.
                max_tokens=256,
                temperature=0.0,
                timeout=float(provider["generator_request_timeout_seconds"]),
            )
            reply = (response.choices[0].message.content or "").strip()
            report["generator_calls"].append({  # type: ignore[union-attr]
                "position": position,
                "status": "ok" if reply else "empty",
                "elapsed_seconds": time.monotonic() - started,
                "nonempty_reply": bool(reply),
                "usage": usage_record(response),
            })
            if not reply:
                raise RuntimeError("Generator returned an empty response")

        # Give the configured global RPM window room before the bridge starts.
        pace()
        tiny_input = root / "tiny_document.jsonl"
        tiny_output = root / "tiny_index_output.json"
        tiny_query_output = root / "tiny_query_output.json"
        tiny_document = {
            "id": "rq3-preflight::medical-hand-hygiene",
            "title": "Disposable RQ3 preflight document",
            "text": (
                "Hand hygiene is an important infection-control practice in clinical settings. "
                "Cleaning hands with soap and water or an appropriate alcohol-based hand rub can "
                "reduce transmission of microorganisms. This short passage exists only to verify "
                "that the LightRAG indexing and query processes complete and exit normally. "
                "The verification keyword for this disposable preflight is ORCHID-47."
            ),
            "metadata": {"preflight_only": True},
        }
        tiny_input.write_text(json.dumps(tiny_document, sort_keys=True) + "\n", encoding="utf-8")
        environment = dict(os.environ)
        environment["RAGSTUDY_CONCURRENT_WORKERS"] = "1"
        environment["RAGSTUDY_REQUESTS_PER_MINUTE"] = str(provider["requests_per_minute"])
        index_command = [
            str(ROOT / ".venv" / "bin" / "python"), "scripts/lightrag_bridge.py", "index",
            "--config", str(config_path), "--workspace", str(workspace),
            "--input", str(tiny_input), "--output", str(tiny_output),
        ]
        print("Running disposable LightRAG index", flush=True)
        index_code, index_seconds, index_log = run_streamed(index_command, environment=environment)
        report["tiny_lightrag"] = {
            "index_exit_code": index_code,
            "index_elapsed_seconds": index_seconds,
            "index_log_tail": index_log[-4000:],
        }
        if index_code != 0 or not tiny_output.exists():
            raise RuntimeError("Disposable LightRAG indexing failed")

        query_command = [
            str(ROOT / ".venv" / "bin" / "python"), "scripts/lightrag_bridge.py", "query",
            "--config", str(config_path), "--workspace", str(workspace),
            "--question", "What is the verification keyword?", "--method", "local",
            "--output", str(tiny_query_output),
        ]
        print("Running disposable LightRAG query", flush=True)
        query_code, query_seconds, query_log = run_streamed(query_command, environment=environment)
        report["tiny_lightrag"].update({  # type: ignore[union-attr]
            "query_exit_code": query_code,
            "query_elapsed_seconds": query_seconds,
            "query_log_tail": query_log[-4000:],
        })
        if query_code != 0 or not tiny_query_output.exists():
            raise RuntimeError("Disposable LightRAG query failed")
        query_result = json.loads(tiny_query_output.read_text(encoding="utf-8"))
        answer = str(query_result.get("answer", ""))
        report["tiny_lightrag"].update({  # type: ignore[union-attr]
            "answer_nonempty": bool(answer.strip()),
            "verification_keyword_retrieved": "ORCHID-47" in normalized_keyword_text(answer),
            "index_output": str(tiny_output),
            "query_output": str(tiny_query_output),
        })
        if "ORCHID-47" not in normalized_keyword_text(answer):
            raise RuntimeError("Disposable LightRAG query did not retrieve the verification keyword")

        report["status"] = "ok"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[OK] preflight passed; report: {report_path}", flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[FAIL] {exc}; report: {report_path}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
