import json
from pathlib import Path
import tempfile
import unittest

from legal_rag.pipeline import Block, parse, chunk, split_prose, article_start, run, extract
from legal_rag.retrieval import rrf, expand_parents


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "법.pdf"
        self.source.write_bytes(b"fixture")

    def doc(self, pages):
        return parse(self.source, pages, [])

    def test_toc_deleted_branch_and_reference(self):
        doc = self.doc([[Block(1, "법"), Block(1, "제1조(목적)"), Block(1, "제2조의2")],
                        [Block(2, "법"), Block(2, "제1조(목적) ① 이 법은 다음을 정한다."),
                         Block(2, "제58조에 따른 인용은 새 조문이 아니다."), Block(2, "제2조의2 삭제")]])
        articles = [s for s in doc["sections"] if s["kind"] == "body"]
        self.assertEqual([s["article"] for s in articles], ["제1조", "제2조의2"])
        self.assertEqual(doc["validation"]["toc_missing_in_body"], [])
        self.assertIn("제58조", articles[0]["blocks"][1]["text"])
        self.assertFalse(article_start("제12조제2항에 따라"))

    def test_short_article_stays_whole(self):
        doc = self.doc([[Block(1, "법"), Block(1, "제1조(목적) ① 첫째\n② 둘째")]])
        parents, children = chunk(doc)
        self.assertEqual(len(children), 1)
        self.assertIn(children[0]["parent_id"], {p["id"] for p in parents})

    def test_nested_unit_context_and_atomic_overflow(self):
        parts = split_prose("제1조(시험) ① 다음 기준을 따른다.\n1. 첫 기준\n가. 세부 기준" + "가" * 100 + "\n나. 다른 기준\n2. 둘째 기준\n② 예외", 20)
        child = next(text for text, path in parts if path == ["①", "1.", "나."])
        self.assertIn("다음 기준", child)
        self.assertIn("첫 기준", child)
        self.assertNotIn("세부 기준", child)
        self.assertTrue(any(len(text) > 20 for text, _ in parts))

    def test_versions_and_local_dates_do_not_leak(self):
        doc = self.doc([[Block(1, "법"), Block(1, "[시행 2026. 9. 11.] [법률 제123호, 2026. 9. 10., 개정]"),
            Block(1, "부칙 <제1호,2020.1.1.>"), Block(1, "제1조(시행일) 이 법은 2020년 2월 1일부터 시행한다."),
            Block(1, "부칙 <제2호,2021.1.1.>"), Block(1, "제1조(시행일) 공포 후 6개월이 경과한 날부터 시행한다. 다만, 제2조는 2023년 1월 1일부터 시행한다.")]])
        _, children = chunk(doc)
        self.assertEqual(doc["metadata"]["effective_date"], "2026-09-11")
        old = [c for c in children if c["metadata"]["version"] == "제1호"]
        relative = [c for c in children if c["metadata"]["version"] == "제2호"]
        self.assertTrue(old)
        self.assertEqual(old[0]["metadata"]["effective_date"], "2020-02-01")
        self.assertIsNone(relative[0]["metadata"]["effective_date"])
        self.assertTrue(relative[0]["metadata"]["supplement"]["has_exceptions"])

    def test_multilevel_headers_and_next_page_continuation(self):
        doc = self.doc([[Block(1, "법"), Block(1, "■ 법 [별표 1] 기준"),
            Block(1, "", "table", rows=[["행위", "금액", None], [None, "1회", "2회"], ["가", "1", "2"], ["나", "3", "4"]])],
            [Block(2, "", "table", rows=[["다", "5", "6"], ["라", "7", "8"]])]])
        _, children = chunk(doc, table_rows=1)
        tables = [c for c in children if c["kind"] == "table"]
        self.assertEqual(len(tables), 4)
        self.assertTrue(all("금액 / 1회" in c["text"] for c in tables))
        self.assertTrue(any("다 | 5 | 6" in c["text"] for c in tables))

    def test_missing_toc_is_reported(self):
        doc = self.doc([[Block(1, "법"), Block(1, "제1조(목적)"), Block(1, "제2조(기준)")],
                        [Block(2, "제1조(목적) 내용")]])
        self.assertEqual(doc["validation"]["toc_missing_in_body"], ["제2조"])

    def test_appendix_list_is_not_indexed(self):
        doc = self.doc([[Block(1, "법"), Block(1, "[별표 1] 기준"), Block(1, "[별표 2] 금액")]])
        self.assertTrue(doc["appendix_list"])
        self.assertEqual(chunk(doc)[1], [])

    def test_stable_ids_and_metadata_override(self):
        pages = [[Block(1, "법"), Block(1, "제1조(목적) 내용")]]
        a, b = self.doc(pages), self.doc(pages)
        self.assertEqual(chunk(a), chunk(b))
        changed = parse(self.source, pages, [], metadata={"version": "reviewed"})
        self.assertNotEqual(a["id"], changed["id"])

    def test_rrf_and_parent_dedup(self):
        self.assertEqual(rrf(["a", "b"], ["b", "c"])[0], "b")
        self.assertEqual(expand_parents(["a", "b"], [{"id": "a", "parent_id": "p"}, {"id": "b", "parent_id": "p"}], {"p": {"text": "원문"}}), [{"text": "원문"}])

    def test_empty_input_and_invalid_limits(self):
        with self.assertRaises(ValueError):
            run(self.tmp.name, Path(self.tmp.name) / "out", 0)
        self.source.unlink()
        with self.assertRaises(ValueError):
            run(self.tmp.name, Path(self.tmp.name) / "out")

    def test_pdf_extraction_and_ocr_hook(self):
        # Real blank PDF tests the extension path without a remote service.
        from reportlab.pdfgen.canvas import Canvas
        c = Canvas(str(self.source)); c.showPage(); c.save()
        class Enhancer:
            called = False
            def enhance(inner, *, source, page, blocks, reasons):
                inner.called = True
                return [Block(1, "OCR로 복원한 내용", method="test-ocr")]
        enhancer = Enhancer()
        pages, warnings = extract(self.source, enhancer)
        self.assertTrue(enhancer.called)
        self.assertEqual(pages[0][0].method, "test-ocr")
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
