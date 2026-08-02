from __future__ import annotations

import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


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
    requests_per_minute = provider.get("requests_per_minute", 30)
    if isinstance(requests_per_minute, bool) or not isinstance(requests_per_minute, int) or requests_per_minute < 1:
        errors.append("provider.requests_per_minute must be a positive integer")
    concurrent_workers = config.data["common"].get("concurrent_workers", 1)
    if isinstance(concurrent_workers, bool) or not isinstance(concurrent_workers, int) or concurrent_workers < 1:
        errors.append("common.concurrent_workers must be a positive integer (defaults to 1)")
    native_needed = any(config.system(item).get("kind") == "native" for item in systems)
    if native_needed:
        for item in ("generator_model", "embedding_model", "api_key_env"):
            if not str(provider.get(item, "")).strip() or str(provider[item]).startswith("SET_"):
                errors.append(f"provider.{item} must be set for native NaiveRAG")
        key_name = provider.get("api_key_env", "")
        if key_name and not os.environ.get(key_name):
            errors.append(f"Environment variable {key_name} is not set")
    for system in systems:
        item = config.system(system)
        if item.get("kind") == "command":
            for field in ("index_command", "query_command"):
                if not str(item.get(field, "")).strip():
                    errors.append(f"systems.{system}.{field} must be set")
        if system == "lightrag":
            dimension = provider.get("embedding_dimension")
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
                errors.append("provider.embedding_dimension must be a positive integer for LightRAG")
    return errors
