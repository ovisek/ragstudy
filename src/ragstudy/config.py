from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


OLLAMA_PLACEHOLDER_API_KEY = "ollama"


def is_ollama_provider(provider: dict[str, Any]) -> bool:
    return str(provider.get("type", "openai")).strip().lower() == "ollama"


def provider_api_key(provider: dict[str, Any]) -> str | None:
    if is_ollama_provider(provider):
        return OLLAMA_PLACEHOLDER_API_KEY
    key_name = str(provider.get("api_key_env", "")).strip()
    return os.environ.get(key_name) if key_name else None


def _is_local_ollama_endpoint(value: object) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _is_set(value: object) -> bool:
    return bool(str(value).strip()) and not str(value).startswith("SET_")


@dataclass(frozen=True)
class Config:
    path: Path
    root: Path
    data: dict[str, Any]

    @property
    def output_root(self) -> Path:
        return self.root / self.data["study"]["output_root"]

    @property
    def data_root(self) -> Path:
        return self.root / self.data["study"]["data_root"]

    def system(self, name: str) -> dict[str, Any]:
        return self.data["systems"][name]

    def dataset(self, name: str) -> dict[str, Any]:
        return self.data["datasets"][name]


def load_config(path: str | Path) -> Config:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    path = Path(path).resolve()
    with path.open("rb") as source:
        data = tomllib.load(source)
    for required in ("study", "common", "provider", "datasets", "systems"):
        if required not in data:
            raise ValueError(f"Missing [{required}] in {path}")
    return Config(path=path, root=path.parent.parent, data=data)


def validate_run_config(config: Config, systems: list[str]) -> list[str]:
    errors: list[str] = []
    provider = config.data["provider"]
    native_needed = any(config.system(item).get("kind") == "native" for item in systems)
    if is_ollama_provider(provider):
        if not _is_local_ollama_endpoint(provider.get("base_url")):
            errors.append("provider.base_url must be an http:// localhost, 127.0.0.1, or ::1 Ollama endpoint")
        for item in ("generator_model", "embedding_model", "judge_model"):
            if not _is_set(provider.get(item, "")):
                errors.append(f"provider.{item} must be set for Ollama")
        dimensions = provider.get("embedding_dimensions")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            errors.append("provider.embedding_dimensions must be a positive integer for Ollama")
    elif native_needed:
        for item in ("base_url", "generator_model", "embedding_model", "api_key_env"):
            if not _is_set(provider.get(item, "")):
                errors.append(f"provider.{item} must be set for native NaiveRAG")
        key_name = str(provider.get("api_key_env", "")).strip()
        if key_name and not os.environ.get(key_name):
            errors.append(f"Environment variable {key_name} is not set")
    for system in systems:
        item = config.system(system)
        if item.get("kind") == "command":
            for field in ("index_command", "query_command"):
                if not str(item.get(field, "")).strip():
                    errors.append(f"systems.{system}.{field} must be set")
    return errors

