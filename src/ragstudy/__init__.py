"""Auditable thesis experiment runner for graph-based RAG systems."""

__version__ = "0.1.0"

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
