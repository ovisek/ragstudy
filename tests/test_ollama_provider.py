from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ragstudy.adapters import NaiveRagAdapter
from ragstudy.config import Config, OLLAMA_PLACEHOLDER_API_KEY, provider_api_key, validate_run_config


def make_config(provider: dict[str, object]) -> Config:
    return Config(
        path=Path("experiment.toml"),
        root=Path("."),
        data={"provider": provider, "common": {}, "systems": {"naive": {"kind": "native"}}},
    )


def ollama_provider(**overrides: object) -> dict[str, object]:
    provider: dict[str, object] = {
        "type": "ollama",
        "base_url": "http://localhost:11434/v1",
        "generator_model": "llama3.2",
        "embedding_model": "nomic-embed-text",
        "judge_model": "llama3.2",
        "embedding_dimensions": 768,
    }
    provider.update(overrides)
    return provider


class OllamaProviderTest(unittest.TestCase):
    def test_ollama_config_needs_no_environment_key(self) -> None:
        config = make_config(ollama_provider())

        self.assertEqual(validate_run_config(config, ["naive"]), [])
        self.assertEqual(provider_api_key(config.data["provider"]), OLLAMA_PLACEHOLDER_API_KEY)

    def test_ollama_requires_local_endpoint_and_dimension(self) -> None:
        config = make_config(ollama_provider(base_url="https://ollama.example.com/v1", embedding_dimensions=0))

        self.assertEqual(
            validate_run_config(config, ["naive"]),
            [
                "provider.base_url must be an http:// localhost, 127.0.0.1, or ::1 Ollama endpoint",
                "provider.embedding_dimensions must be a positive integer for Ollama",
            ],
        )

    def test_existing_provider_still_requires_configured_environment_key(self) -> None:
        config = make_config({
            "base_url": "https://api.example.com/v1",
            "api_key_env": "RAGSTUDY_TEST_API_KEY",
            "generator_model": "generator",
            "embedding_model": "embedding",
        })

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(validate_run_config(config, ["naive"]), ["Environment variable RAGSTUDY_TEST_API_KEY is not set"])

    def test_naive_adapter_uses_placeholder_key_for_ollama(self) -> None:
        created: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs: object) -> None:
                created.update(kwargs)

        config = make_config(ollama_provider())
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            NaiveRagAdapter(config, "naive", Path("workspace"))

        self.assertEqual(created, {"api_key": OLLAMA_PLACEHOLDER_API_KEY, "base_url": "http://localhost:11434/v1"})