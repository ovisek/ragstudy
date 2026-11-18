#!/usr/bin/env python3
"""Run pinned RAGChecker through BAS's OpenAI-compatible provider client."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

from dotenv import load_dotenv

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    provider = config["provider"]
    key = os.environ.get(provider["api_key_env"])
    if not key:
        raise SystemExit(f"Missing {provider['api_key_env']}")

    from openai import OpenAI
    from ragchecker import RAGChecker
    from ragchecker.container import RAGResults

    client = OpenAI(api_key=key, base_url=provider["base_url"], timeout=900.0, max_retries=2)
    usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
    prompt_counter = 0

    def call_provider(prompts: list[str]) -> list[str]:
        nonlocal prompt_counter
        outputs: list[str] = []
        for prompt in prompts:
            prompt_counter += 1
            label = f"RAGChecker prompt {prompt_counter}"
            started = monotonic()
            stop = Event()

            def heartbeat() -> None:
                while not stop.wait(30):
                    print(f"[{timestamp()}] ACTIVE {label}; elapsed={monotonic() - started:.1f}s", flush=True)

            print(f"[{timestamp()}] START {label}; characters={len(prompt)}", flush=True)
            thread = Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                last_error: Exception | None = None
                for attempt in range(1, 4):
                    try:
                        response = client.chat.completions.create(
                            model=provider["judge_model"],
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=4096,
                        )
                        content = response.choices[0].message.content
                        if not content:
                            raise RuntimeError("provider returned no final answer content")
                        token_usage = response.usage
                        usage["api_calls"] += 1
                        usage["input_tokens"] += int(getattr(token_usage, "prompt_tokens", 0) or 0)
                        usage["output_tokens"] += int(getattr(token_usage, "completion_tokens", 0) or 0)
                        outputs.append(content)
                        print(f"[{timestamp()}] COMPLETE {label}; attempt={attempt}; elapsed={monotonic() - started:.1f}s", flush=True)
                        break
                    except Exception as exc:
                        last_error = exc
                        print(f"[{timestamp()}] RETRY {label}; attempt={attempt}; error={type(exc).__name__}: {exc}", flush=True)
                        if attempt < 3:
                            sleep(10)
                else:
                    raise RuntimeError(f"{label} failed after 3 attempts") from last_error
            finally:
                stop.set()
                thread.join(timeout=1)
        return outputs

    source_path = args.output if args.output.exists() else args.input
    rag_results = RAGResults.from_json(source_path.read_text(encoding="utf-8"))
    evaluator = RAGChecker(
        extractor_name=provider["judge_model"],
        checker_name=provider["judge_model"],
        extractor_max_new_tokens=4096,
        batch_size_extractor=1,
        batch_size_checker=1,
        joint_check=True,
        joint_check_num=5,
        custom_llm_api_func=call_provider,
    )
    started = monotonic()
    evaluator.evaluate(rag_results, metrics=["overall_metrics"], save_path=str(args.output))
    args.output.write_text(rag_results.to_json(indent=2), encoding="utf-8")

    manifest = {
        "backend": "ragchecker",
        "evaluator_model": provider["judge_model"],
        "evaluator_version": importlib.metadata.version("ragchecker"),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output),
        "result_rows": len(rag_results.results),
        "metrics": ["precision", "recall", "f1"],
        "elapsed_seconds_this_invocation": monotonic() - started,
        "judge_usage": usage,
        "actual_expenditure_usd": 0.0,
        "notional_cost_usd": None,
    }
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result_rows": len(rag_results.results), "metrics": rag_results.metrics, "manifest": str(manifest_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
