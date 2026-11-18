from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_graphrag_bench_json.py"
SPEC = importlib.util.spec_from_file_location("convert_graphrag_bench_json", SCRIPT)
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class GraphRAGBenchConversionTest(unittest.TestCase):
    def test_convert_partitions_source_without_using_evidence_as_document_text(self) -> None:
        corpus = [{"corpus_name": "Medical", "context": "Alpha beta gamma delta epsilon zeta eta theta."}]
        questions = [{
            "id": "Medical-q1", "question": "Question?", "answer": "Answer.",
            "question_type": "Fact Retrieval", "evidence": ["Published evidence."],
            "evidence_relations": "Published relation",
        }]

        documents, converted_questions = converter.convert(corpus, questions, 2, 1, "https://example.test/source")

        self.assertEqual("".join(document["text"] for document in documents), corpus[0]["context"])
        self.assertEqual(len(documents), 2)
        self.assertEqual(converted_questions[0]["id"], "Medical-q1")
        self.assertEqual(converted_questions[0]["evidence"], ["Published evidence."])
        self.assertEqual(converted_questions[0]["evidence_relations"], "Published relation")

    def test_convert_rejects_more_questions_than_released(self) -> None:
        corpus = [{"corpus_name": "Medical", "context": "Alpha beta"}]
        questions = [{"id": "q1", "question": "Q", "answer": "A", "question_type": "Fact", "evidence": ["E"]}]

        with self.assertRaisesRegex(ValueError, "published questions"):
            converter.convert(corpus, questions, 1, 2, "https://example.test/source")


if __name__ == "__main__":
    unittest.main()
