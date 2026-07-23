from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .io import write_json


def _command_version(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False, timeout=10).stdout.strip() or None
    except OSError:
        return None


def freeze_manifest(config: Config) -> dict[str, Any]:
    package_names = ("openai", "graphrag", "lightrag-hku", "ragas", "ragchecker", "datasets")
    installed = {}
    for name in package_names:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    provider = config.data["provider"]
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config.data["study"]["protocol_version"],
        "study": config.data["study"],
        "common_controls": config.data["common"],
        "provider": {key: value for key, value in provider.items() if key != "api_key_env"},
        "api_key_environment_variable": provider.get("api_key_env"),
        "systems": config.data["systems"],
        "datasets": config.data["datasets"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "docker_version": _command_version(["docker", "--version"]),
            "neo4j_version": _command_version(["neo4j", "--version"]),
            "packages": installed,
        },
    }
    write_json(config.output_root / "experiment_manifest.json", payload)
    return payload

