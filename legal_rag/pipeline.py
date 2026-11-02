"""Deterministic local PDF pipeline. No network calls or inferred legal validity."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from datetime import date
import hashlib
import importlib
import json
from pathlib import Path
import re
from typing import Protocol

ARTICLE = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(\([^\n]*?\))?")
CHAPTER = re.compile(r"^제\s*\d+\s*[편장절관](?:의\d+)?(?:\s|$)")
SUPPLEMENT = re.compile(r"^부\s*칙(?:\s|<|\[|$)")
APPENDIX = re.compile(r"\[\s*별(?:표|지)\s*[^\]]+\]")
UNIT = re.compile(r"^([①-⑳㉑-㉟㊱-㊿]|\d+\s*[.](?!\d)|[가-하]\s*[.]|\d+\))\s*")
DATE = r"(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?"
KOREAN_DATE = r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"


def uid(*values):
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()[:24]


def iso(match):
    if not match:
        return None
    try:
        return date(*map(int, match.groups()[:3])).isoformat()
    except ValueError:
        return None


@dataclass
class Block:
    page: int
    text: str
    kind: str = "text"
    bbox: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    method: str = "pdfplumber"
    header_rows: int = 1


class PageEnhancer(Protocol):
    """Called for sparse pages/ambiguous tables. Return replacement blocks or None.

    An implementation may render pdfplumber's page to an image and call local OCR
    or a VLM. It must return original page coordinates, table rows, and method.
    No remote upload is performed by the default pipeline.
    """
    def enhance(self, *, source: Path, page, blocks: list[Block], reasons: list[str]) -> list[Block] | None: ...


def table_md(rows):
    rows = [[str(c or "").replace("|", "\\|").replace("\n", "<br>") for c in r] for r in rows]
    if not rows:
        return ""
    width = max(map(len, rows))
    rows = [r + [""] * (width - len(r)) for r in rows]
    line = lambda r: "| " + " | ".join(r) + " |"
    return "\n".join([line(rows[0]), line(["---"] * width)] + [line(r) for r in rows[1:]])


def extract(source: Path, enhancer: PageEnhancer | None = None):
    import pdfplumber
    pages, warnings = [], []
    with pdfplumber.open(source) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            # Font borders can look like one-row tables. Keep genuine grids;
            # rejected regions remain ordinary prose, so no content disappears.
            tables = [t for t in page.find_tables() if len(t.extract()) >= 2 and
                      max((len(r) for r in t.extract()), default=0) >= 2]
            blocks = []
            for t in tables:
                rows = t.extract()
                blocks.append(Block(number, table_md(rows), "table", list(t.bbox), rows))
            # Exclude table glyphs from prose to avoid indexing them twice.
            outside = page.filter(lambda o: not (o.get("object_type") == "char" and any(
                t.bbox[0] <= (o["x0"] + o["x1"]) / 2 <= t.bbox[2] and
                t.bbox[1] <= (o["top"] + o["bottom"]) / 2 <= t.bbox[3] for t in tables)))
            for line in outside.extract_text_lines():
                blocks.append(Block(number, line["text"], bbox=[line["x0"], line["top"], line["x1"], line["bottom"]]))
            blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            reasons = []
            if sum(len(b.text) for b in blocks) < 40:
                reasons.append("sparse_text_requires_ocr")
            if any(any(c is None for c in row) for t in tables for row in t.extract()):
                reasons.append("merged_or_empty_cells_require_review")
            if tables:
                reasons.append("table_headers_inferred_require_review")
            if reasons and enhancer:
                replacement = enhancer.enhance(source=source, page=page, blocks=blocks, reasons=reasons)
                if replacement is not None:
                    if any(b.page != number for b in replacement):
                        raise ValueError("Enhancer returned wrong page number")
                    blocks = replacement
            if reasons:
                warnings.append({"page": number, "reasons": reasons, "enhancer_called": bool(enhancer)})
            pages.append(blocks)
    return pages, warnings


def article_key(text):
    m = ARTICLE.match(text)
    return f"제{m[1]}조" + (f"의{m[2]}" if m[2] else "") if m else None


def article_start(text):
    m = ARTICLE.match(text)
    return m if m and (m[3] or not text[m.end():].strip() or text[m.end():].lstrip().startswith("삭제")) else None


def is_list_page(blocks):
    """Law.go.kr exports have unlabelled TOC pages before the repeated title.

    Only heading-only pages are classified as lists; deleted body articles have
    a trailing 삭제 and are therefore kept as body.
    """
    candidates = [b.text.strip() for b in blocks if ARTICLE.match(b.text.strip())]
    if not candidates or any(b.kind == "table" for b in blocks):
        return False
    heading_only = sum(bool(ARTICLE.fullmatch(s)) for s in candidates)
    other = [b.text for b in blocks if not (ARTICLE.match(b.text) or CHAPTER.match(b.text)
             or b.text.startswith("[") or "법제처" in b.text)]
    return heading_only == len(candidates) and len(other) <= 3


def parse(source: Path, pages, warnings, *, metadata=None):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    flat = [b for page in pages for b in page]
    top = "\n".join(b.text for b in flat[:30])
    title = next((b.text.strip() for b in flat if b.text.strip()), source.stem)
    meta = {"source": str(source.resolve()), "source_sha256": source_hash, "law_title": title,
            "effective_date": iso(re.search(r"\[시행\s*" + DATE, top)),
            "promulgation_date": None, "version": None, "validity": "unreviewed"}
    version = re.search(r"\[([^\]\n]*제\s*[\d-]+\s*호),\s*" + DATE, top)
    if version:
        meta["version"] = version[1]
        meta["promulgation_date"] = iso(re.search(DATE, version[0]))
    meta.update(metadata or {})
    docid = uid(source_hash, json.dumps(meta, sort_keys=True, ensure_ascii=False))
    sections, toc, appendix_list = [], [], []
    current, chapter, supplement = None, [], None
    section_kind = "body"

    def start(kind, heading, block, **extra):
        nonlocal current
        current = {"id": uid(docid, len(sections), kind, heading), "kind": kind,
                   "heading": heading, "chapter": list(chapter), "blocks": [],
                   "supplement": dict(supplement) if supplement else None, **extra}
        sections.append(current)

    for page in pages:
        list_page = meta.get("document_kind") == "toc" or is_list_page(page)
        # Appendix list pages contain bracketed headings only, not actual annexes
        # which usually start with '■ 법령명 [별표 ...]'.
        annexes = [b for b in page if APPENDIX.match(b.text.strip())]
        annex_list = meta.get("document_kind") == "appendix_list" or (len(annexes) >= 2 and not any(b.kind == "table" for b in page))
        for b in page:
            text = b.text.strip()
            if (not text and b.kind != "table") or text == title or re.match(r"^법제처\s*\d+\s*국가법령정보센터", text):
                continue
            if list_page:
                if ARTICLE.match(text):
                    toc.append({"article": article_key(text), "heading": text, "page": b.page})
                continue
            if annex_list:
                appendix_list.append({"text": text, "page": b.page})
                continue
            if SUPPLEMENT.match(text):
                section_kind = "supplement"
                supplement = {"heading": text, "version": None, "promulgation_date": iso(re.search(DATE, text)),
                              "effective_date": None, "effective_date_text": None, "validity": "unreviewed"}
                v = re.search(r"제\s*[\d-]+\s*호", text)
                if v:
                    supplement["version"] = v[0]
                chapter = []
                start("supplement", text, b)
            elif APPENDIX.search(text) and (text.startswith("■") or APPENDIX.match(text)):
                section_kind, supplement, chapter = "appendix", None, []
                start("form" if "별지" in text else "appendix", text, b)
            elif section_kind != "appendix" and CHAPTER.match(text):
                rank = next((i for i, x in enumerate("편장절관") if x in text.split()[0]), 1)
                chapter = [x for x in chapter if x[0] < rank] + [(rank, text)]
                current = None
                continue
            elif section_kind != "appendix" and article_start(text):
                m = article_start(text)
                start(section_kind, m[0].strip(), b, article=article_key(text))
            elif current is None:
                start("preamble", "문서 정보", b)
            current["blocks"].append(asdict(b))
    # Supplement effective dates are local; do not inherit the current law date.
    supplements = {}
    for s in sections:
        if s["supplement"]:
            supplements.setdefault(s["supplement"]["heading"], []).append(s)
    for group in supplements.values():
        text = "\n".join(b["text"] for s in group for b in s["blocks"])
        effective_section = next((s for s in group if "(시행일)" in s["heading"]), group[0])
        clause = "\n".join(b["text"] for b in effective_section["blocks"])
        # Preserve all exceptions and relative-date wording. A date below is
        # merely the base commencement date, never an assertion of validity.
        base_clause = clause.split("다만", 1)[0]
        value = iso(re.search(KOREAN_DATE, base_clause)) or iso(re.search(DATE, base_clause))
        if not value and re.search(r"(?:공포|고시|발령)한 날\s*부터 시행", base_clause):
            value = group[0]["supplement"]["promulgation_date"]
        for s in group:
            s["supplement"].update(effective_date=value, effective_date_text=clause or None,
                                   has_exceptions="다만" in clause, date_scope="base_only")
    actual = {s.get("article") for s in sections if s["kind"] == "body"}
    expected = {t["article"] for t in toc}
    return {"schema_version": 1, "id": docid, "metadata": meta, "toc": toc,
            "appendix_list": appendix_list, "sections": sections, "warnings": warnings,
            "validation": {"toc_missing_in_body": sorted(expected - actual),
                           "body_missing_in_toc": sorted(actual - expected) if expected else []}}


def split_prose(text, limit):
    """Split only at legal markers; repeat ancestor lead-ins for each child.

    Oversize atomic clauses remain whole and are flagged instead of cut mid-rule.
    """
    if len(text) <= limit:
        return [(text, [])]
    m = article_start(text)
    if m and text[m.end():].strip():
        text = text[:m.end()] + "\n" + text[m.end():].lstrip()
    lines = text.splitlines()
    result, ancestors, buffer, path = [], {}, [], []
    def flush():
        if buffer:
            result.append(("\n".join(buffer), list(path)))
    for line in lines:
        m = UNIT.match(line)
        if m:
            marker = m[1]
            level = 1 if marker[0] in "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿" else (3 if re.match(r"[가-하]", marker) else 2)
            flush()
            ancestors = {k: v for k, v in ancestors.items() if k < level}
            buffer = [v[1] for k, v in sorted(ancestors.items())] + [line]
            ancestors[level] = (marker, line)
            path = [v[0] for k, v in sorted(ancestors.items())]
        else:
            buffer.append(line)
            if ancestors:
                k = max(ancestors)
                ancestors[k] = (ancestors[k][0], ancestors[k][1] + "\n" + line)
    flush()
    return result


def chunk(doc, max_chars=1000, table_rows=8):
    parents, children = [], []
    for section in doc["sections"]:
        text = "\n".join(b["text"] for b in section["blocks"])
        parent = {**section, "text": text, "document_id": doc["id"], "metadata": doc["metadata"]}
        parents.append(parent)
        if section["kind"] == "preamble":
            continue
        context = " / ".join([doc["metadata"]["law_title"]] + [c[1] for c in section["chapter"]] +
                               ([section["supplement"]["heading"]] if section["supplement"] else []) + [section["heading"]])
        pieces, prose = [], []
        previous_header, previous_width, previous_page = [], None, None
        def flush():
            if prose:
                pieces.extend((t, p, "text", sorted({b["page"] for b in prose})) for t, p in
                              split_prose("\n".join(b["text"] for b in prose), max_chars))
                prose.clear()
        for block in section["blocks"]:
            if block["kind"] != "table":
                prose.append(block)
                continue
            flush()
            rows = block["rows"]
            # Preserve original cell matrix in parent, repeat header + caption.
            caption = "\n".join(b["text"] for b in section["blocks"][:section["blocks"].index(block)] if b["kind"] == "text")[-600:]
            width = max(map(len, rows), default=0)
            # Law.go.kr tables often continue on the next page without headers.
            # Reuse only within the same annex and same column count. Flagged
            # in extraction report for visual review (not guaranteed semantics).
            explicit_header = bool(re.search(r"위반행위|근거|구분|학\s*위|자격|기준|항목|금액", " ".join(str(x or "") for x in rows[0])))
            continuation = (section["kind"] in {"appendix", "form"} and previous_header and
                            width == previous_width and block["page"] == previous_page + 1 and not explicit_header)
            header_count = 0 if continuation else block.get("header_rows", 1)
            if not continuation and len(rows) > 1 and any(c is None for c in rows[0]) and any(c is None for c in rows[1]):
                header_count = max(header_count, 2)
            headers = previous_header if continuation else rows[:header_count]
            # Flatten multi-level headers for Markdown; original matrix stays
            # untouched in the parent. Carry merged top labels horizontally.
            filled = []
            for row in headers:
                last, out = "", []
                for cell in row:
                    last = str(cell) if cell is not None else last
                    out.append(last)
                filled.append(out)
            combined = [" / ".join(dict.fromkeys(r[i].replace("\n", " ") for r in filled if i < len(r) and r[i])) for i in range(width)]
            for i in range(header_count, max(header_count+1, len(rows)), table_rows):
                pieces.append((caption + "\n" + table_md([combined] + rows[i:i+table_rows]),
                               [f"table_rows:{i}-{min(i+table_rows-1,len(rows)-1)}"], "table", [block["page"]]))
            previous_header, previous_width, previous_page = headers, width, block["page"]
        flush()
        for i, (body, path, kind, pages) in enumerate(pieces):
            local = section["supplement"]
            metadata = {**doc["metadata"], "section_kind": section["kind"], "article": section.get("article"),
                        "chapter": section["chapter"], "unit_path": path, "pages": pages,
                        "supplement": local, "oversize": len(body) > max_chars}
            metadata["deleted"] = bool(re.match(r"^제\s*\d+\s*조(?:의\d+)?\s+삭제", text))
            if local:
                metadata.update(effective_date=local["effective_date"], version=local["version"],
                                promulgation_date=local["promulgation_date"])
            children.append({"id": uid(section["id"], i), "parent_id": section["id"],
                             "document_id": doc["id"], "kind": kind, "text": body,
                             "search_text": context + "\n" + body, "metadata": metadata})
    return parents, children


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, values):
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")


def run(input_dir, output_dir, max_chars=1000, table_rows=8, enhancer=None, overrides=None):
    if max_chars < 1 or table_rows < 1:
        raise ValueError("max_chars and table_rows must be positive")
    sources = sorted(Path(input_dir).rglob("*.pdf"))
    if not sources:
        raise ValueError(f"No PDFs found: {input_dir}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parents, children, report = [], [], []
    for source in sources:
        pages, warnings = extract(source, enhancer)
        relative = source.relative_to(input_dir).as_posix()
        doc = parse(source, pages, warnings, metadata=(overrides or {}).get(relative))
        folder = output / "documents" / doc["id"]
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "document.json", doc)
        write_json(folder / "raw_pages.json", [[asdict(b) for b in p] for p in pages])
        (folder / "document.md").write_text("# " + doc["metadata"]["law_title"] + "\n\n" +
            "\n\n".join("## " + s["heading"] + "\n\n" + "\n".join(b["text"] for b in s["blocks"]) for s in doc["sections"]), encoding="utf-8")
        p, c = chunk(doc, max_chars, table_rows)
        parents.extend(p)
        children.extend(c)
        report.append({"source": str(source), "document_id": doc["id"], "pages": len(pages),
                       "parents": len(p), "children": len(c), "tables": sum(b.kind == "table" for pg in pages for b in pg),
                       "warnings": warnings, "validation": doc["validation"]})
    write_jsonl(output / "parents.jsonl", parents)
    write_jsonl(output / "children.jsonl", children)
    write_jsonl(output / "bm25.jsonl", ({"id": c["id"], "parent_id": c["parent_id"],
                   "text": c["search_text"], "metadata": c["metadata"]} for c in children))
    write_jsonl(output / "dense.jsonl", ({"id": c["id"], "parent_id": c["parent_id"],
                   "text": c["search_text"], "metadata": c["metadata"]} for c in children))
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/legal_rag"))
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--table-rows", type=int, default=8)
    parser.add_argument("--metadata", type=Path, help="JSON mapping of relative PDF paths to metadata overrides")
    parser.add_argument("--enhancer", help="module:object implementing PageEnhancer (object or zero-argument class)")
    args = parser.parse_args()
    enhancer = None
    if args.enhancer:
        module, name = args.enhancer.split(":", 1)
        enhancer = getattr(importlib.import_module(module), name)
        if isinstance(enhancer, type):
            enhancer = enhancer()
    report = run(args.input, args.output, args.max_chars, args.table_rows, enhancer,
                 json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else None)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
