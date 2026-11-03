import unittest
from legal_rag.rerank import compare, reorder


class RerankTests(unittest.TestCase):
    def test_order_stable_and_invalid_scores(self):
        self.assertEqual(reorder(["a", "b", "c"], [0, 2, 2]), ["b", "c", "a"])
        for values in ([1], [float("nan"), 0], [float("inf"), 0]):
            with self.assertRaises(ValueError):
                reorder(["a", "b"], values)

    def test_rerank_uses_only_selected_children_and_restores_parents(self):
        children = [{"id": "a", "parent_id": "p", "search_text": "무관한 내용"},
                    {"id": "b", "parent_id": "q", "search_text": "정답 근거"},
                    {"id": "c", "parent_id": "q", "search_text": "다른 문맥"}]
        question = {"id": "q1", "question": "질문", "split": "dev", "category": "direct", "answerable": True,
                    "evidence": [{"child_ids": ["b"], "parent_id": "q"}]}
        class Scorer:
            def scores(self, query, texts):
                self_test.assertEqual(query, "질문")
                self_test.assertEqual(texts, ["무관한 내용", "정답 근거"])
                return [0, 5]
        self_test = self
        before, after, details = compare([question], [["a", "b", "c"]], children,
            {"p": {"id": "p"}, "q": {"id": "q"}}, Scorer(), 2)
        self.assertEqual(before[0]["metrics"]["mrr@10"], 0.5)
        self.assertEqual(after[0]["metrics"]["mrr@10"], 1)
        self.assertEqual(details[0]["restored_parent_ids"], ["q", "p"])
        self.assertNotIn("c", details[0]["candidate_ids"])

    def test_missing_candidate_cannot_be_recovered(self):
        children = [{"id": "a", "parent_id": "p", "search_text": "다른 문맥"},
                    {"id": "b", "parent_id": "p", "search_text": "정답"}]
        question = {"id": "q1", "question": "질문", "split": "dev", "category": "multi", "answerable": True,
                    "evidence": [{"child_ids": ["b"], "parent_id": "p"}]}
        class Scorer:
            def scores(self, query, texts):
                return [9]
        _, after, details = compare([question], [["a", "b"]], children, {"p": {"id": "p"}}, Scorer(), 1)
        self.assertEqual(after[0]["metrics"]["mrr@10"], 0)
        self.assertEqual(details[0]["candidate_evidence_recall"], 0)
        self.assertEqual(after[0]["metrics"]["parent_recall@5"], 1)

    def test_unanswerable_is_not_scored(self):
        class Scorer:
            def scores(self, query, texts):
                return []
        question = {"id": "q1", "question": "질문", "split": "test", "category": "unanswerable", "answerable": False, "evidence": []}
        before, after, details = compare([question], [[]], [], {}, Scorer(), 50)
        self.assertIsNone(before[0]["metrics"])
        self.assertIsNone(after[0]["metrics"])
        self.assertIsNone(details[0]["candidate_evidence_recall"])


if __name__ == "__main__":
    unittest.main()
