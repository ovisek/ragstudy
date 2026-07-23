from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_ultradomain_jsonl.py"
SPEC = importlib.util.spec_from_file_location("convert_ultradomain_jsonl", SCRIPT)
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class UltraDomainConversionTest(unittest.TestCase):
    def test_convert_deduplicates_contexts_and_preserves_question_lineage(self) -> None:
        source = [
            {
                "_id": "q2", "input": "Second question", "answers": ["Second answer"],
                "context": "Shared source text", "context_id": "c1", "label": "agriculture",
                "meta": {"title": "Source title", "authors": "An Author"},
            },
            {
                "_id": "q1", "input": "First question", "answers": ["First answer"],
                "context": "Shared source text", "context_id": "c1", "label": "agriculture",
                "meta": {"title": "Source title", "authors": "An Author"},
            },
        ]

        documents, questions, repeated = converter.convert(source, "https://example.test/source")

        self.assertEqual(repeated, 1)
        self.assertEqual(documents, [{
            "id": "c1", "text": "Shared source text", "title": "Source title",
            "metadata": {
                "source": "https://example.test/source", "source_context_id": "c1",
                "authors": "An Author", "domain": "agriculture",
            },
        }])
        self.assertEqual([question["id"] for question in questions], ["q1", "q2"])
        self.assertEqual(questions[0]["evidence"], ["c1"])

    def test_convert_rejects_conflicting_context_text(self) -> None:
        source = [
            {"_id": "q1", "input": "Question", "answers": ["Answer"], "context": "A", "context_id": "c1", "label": "agriculture"},
            {"_id": "q2", "input": "Question", "answers": ["Answer"], "context": "B", "context_id": "c1", "label": "agriculture"},
        ]

        with self.assertRaisesRegex(ValueError, "different text"):
            converter.convert(source, "https://example.test/source")


if __name__ == "__main__":
    unittest.main()
