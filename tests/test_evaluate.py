import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from legal_rag.evaluate import BM25, embedding_input, metrics, validate_dataset, embed


class EvaluationTests(unittest.TestCase):
    def test_bm25_relevant_and_unknown(self):
        index = BM25(["개인정보 접근 권한 관리", "과태료 납부 기한"])
        scores = index.scores("접근권한")
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(index.scores("zzzzzz"), [0, 0])

    def test_query_prefixes(self):
        self.assertTrue(embedding_input("qwen3-embedding:0.6b", "질문", "query").startswith("Instruct:"))
        self.assertEqual(embedding_input("qwen3-embedding:0.6b", "본문", "document"), "본문")
        self.assertEqual(embedding_input("multilingual-e5-large", "본문", "document"), "passage: 본문")
        self.assertEqual(embedding_input("bge-m3", "질문", "query"), "질문")

    def test_metrics_groups_and_parent_expansion(self):
        question = {"answerable": True, "evidence": [
            {"child_ids": ["a", "b"], "parent_id": "p"},
            {"child_ids": ["c"], "parent_id": "q"}]}
        lookup = {"a": {"parent_id": "p"}, "b": {"parent_id": "p"},
                  "c": {"parent_id": "q"}, "x": {"parent_id": "q"}}
        result = metrics(["x", "a", "b"], question, lookup)
        self.assertEqual(result["mrr@10"], 0.5)
        self.assertEqual(result["evidence_recall@5"], 0.5)
        self.assertEqual(result["all_evidence@5"], 0)
        self.assertEqual(result["parent_recall@5"], 1)
        self.assertIsNone(metrics([], {"answerable": False}, {}))

    def test_stale_evidence_fails(self):
        question = {"id": "q", "answerable": True, "evidence": [{"parent_id": "p", "quote": "근거", "child_ids": ["a"]}]}
        parent = {"id": "p", "text": "근거"}
        validate_dataset([question], [{"id": "a", "parent_id": "p", "text": "근거"}], {"p": parent})
        with self.assertRaisesRegex(ValueError, "child 근거 불일치"):
            validate_dataset([question], [{"id": "b", "parent_id": "p", "text": "근거"}], {"p": parent})

    def test_embedding_cache_and_no_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache.jsonl"
            with patch("legal_rag.evaluate.request_json", return_value={"embeddings": [[3, 4]]}) as request:
                vectors = embed(["본문"], "bge-m3", {"digest": "one"}, "http://localhost", cache, "document", 8)
                self.assertAlmostEqual(float(vectors[0, 0]), 0.6)
                self.assertFalse(request.call_args.args[1]["truncate"])
            with patch("legal_rag.evaluate.request_json", side_effect=AssertionError("cache missed")):
                embed(["본문"], "bge-m3", {"digest": "one"}, "http://localhost", cache, "document", 8)
            with patch("legal_rag.evaluate.request_json", return_value={"embeddings": [[0, 1]]}) as request:
                embed(["본문"], "bge-m3", {"digest": "two"}, "http://localhost", cache, "document", 8)
                self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
