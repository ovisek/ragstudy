#!/usr/bin/env python3
"""Smoke-test the provider endpoint using settings from experiment.toml."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from openai import OpenAI

CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.toml"

# Keep SOCKS disabled by default unless explicitly requested.
if os.getenv("RAGSTUDY_KEEP_ALL_PROXY") not in {"1", "true", "TRUE", "yes", "YES"}:
    os.environ.pop("all_proxy", None)
    os.environ.pop("ALL_PROXY", None)


def load_provider() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))["provider"]


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f": {detail}" if detail else ""))


def main() -> int:
    p = load_provider()
    api_key = os.getenv(p["api_key_env"])
    if not api_key:
        print(f"[FAIL] {p['api_key_env']} is not set")
        return 1

    client = OpenAI(api_key=api_key, base_url=p["base_url"])

    # --- generator ---
    try:
        resp = client.chat.completions.create(
            model=p["generator_model"],
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=8,
            temperature=0.0,
            timeout=30.0,
        )
        reply = resp.choices[0].message.content or ""
        check("generator", True, repr(reply.strip()))
    except Exception as exc:
        check("generator", False, str(exc))
        return 1

    # --- embedding ---
    try:
        eresp = client.embeddings.create(
            model=p["embedding_model"],
            input=["hello world"],
            timeout=30.0,
        )
        dim = len(eresp.data[0].embedding)
        expected = p.get("embedding_dimension", 0)
        ok = dim == expected
        check("embedding", ok, f"dim={dim} (expected {expected})")
        if not ok:
            return 1
    except Exception as exc:
        check("embedding", False, str(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
