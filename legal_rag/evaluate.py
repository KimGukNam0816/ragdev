"""BM25 / Ollama dense / RRF 검색 평가. 실행: uv run --extra eval python -m legal_rag.evaluate"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from .retrieval import rrf


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compact(text):
    return re.sub(r"\s+", "", text.replace("<br>", " "))


def tokens(text):
    # 형태소 분석기 없이 재현 가능한 기준선: 단어 + 한글 2글자 조각.
    words = re.findall(r"[가-힣]+|[a-z0-9]+", text.lower())
    return words + ["~" + word[i:i+2] for word in words if re.fullmatch(r"[가-힣]+", word) for i in range(len(word)-1)]


class BM25:
    def __init__(self, texts):
        self.counts = [Counter(tokens(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(texts)) or 1
        df = Counter(term for count in self.counts for term in count)
        self.idf = {term: math.log(1 + (len(texts)-n+0.5)/(n+0.5)) for term, n in df.items()}

    def scores(self, query):
        scores = []
        for count, length in zip(self.counts, self.lengths):
            score = 0.0
            for term in set(tokens(query)):
                freq = count[term]
                score += self.idf.get(term, 0) * freq * 2.5 / (freq + 1.5*(0.25+0.75*length/self.average))
            scores.append(score)
        return scores


def request_json(url, value=None, timeout=180):
    data = json.dumps(value).encode() if value is not None else None
    try:
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Ollama HTTP {error.code}: {error.read().decode(errors='replace')}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama 접속 실패: {url} / {error.reason}") from error


def model_info(base_url, model):
    models = request_json(base_url + "/api/tags")["models"]
    name = model if ":" in model else model + ":latest"
    found = next((m for m in models if m["name"] == name), None)
    if found is None:
        raise ValueError(f"설치되지 않은 모델: {name}. ollama pull {name}")
    return found


def embedding_input(model, text, role):
    # Qwen은 질문에만 검색 지시문, E5는 질문/문서 접두어가 필요하다.
    if model.startswith("qwen3-embedding") and role == "query":
        return "Instruct: Given a Korean legal question, retrieve passages that answer the question.\nQuery: " + text
    if "e5" in model:
        return ("query: " if role == "query" else "passage: ") + text
    return text


def embed(texts, model, info, base_url, cache_path, role, batch_size):
    import numpy as np
    cache = {r["key"]: r["vector"] for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    inputs = [embedding_input(model, text, role) for text in texts]
    keys = [hashlib.sha256((info["digest"] + "\0" + text).encode()).hexdigest() for text in inputs]
    missing = list(dict.fromkeys(key for key in keys if key not in cache))
    lookup = dict(zip(keys, inputs))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as file:
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start+batch_size]
            result = request_json(base_url + "/api/embed", {
                "model": model, "input": [lookup[k] for k in batch], "truncate": False,
                "keep_alive": "10m", "options": {"temperature": 0}})["embeddings"]
            if len(result) != len(batch):
                raise ValueError("Ollama 응답 벡터 수 불일치")
            for key, vector in zip(batch, result):
                cache[key] = vector
                file.write(json.dumps({"key": key, "vector": vector}) + "\n")
            file.flush()  # 중간에 멈춰도 완료한 배치는 다음 실행에서 재사용한다.
            print(f"{model} {role}: {min(start+batch_size,len(missing))}/{len(missing)}", flush=True)
    vectors = np.asarray([cache[key] for key in keys], dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError("유효하지 않은 임베딩 벡터")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def validate_dataset(questions, children, parents):
    # 정답 근거가 사라지면 점수를 내지 않는다. 재청킹 후 평가셋 재연결이 필요하다.
    child_map = {c["id"]: c for c in children}
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("평가 질문 ID 중복")
    for q in questions:
        if not q["answerable"]:
            if q["evidence"]:
                raise ValueError(f"{q['id']}: 답 없음 문항에 근거가 있음")
            continue
        if not q["evidence"]:
            raise ValueError(f"{q['id']}: 근거 누락")
        for evidence in q["evidence"]:
            parent = parents.get(evidence["parent_id"])
            if not parent or compact(evidence["quote"]) not in compact(parent["text"]):
                raise ValueError(f"{q['id']}: parent 근거 불일치")
            expected = {c["id"] for c in children if c["parent_id"] == evidence["parent_id"] and compact(evidence["quote"]) in compact(c["text"])}
            if not expected or expected != set(evidence["child_ids"]):
                raise ValueError(f"{q['id']}: child 근거 불일치; 평가셋 재연결 필요")
            if any(child_map[cid]["parent_id"] != parent["id"] for cid in evidence["child_ids"]):
                raise ValueError(f"{q['id']}: parent 연결 불일치")


def metrics(ranking, question, child_map):
    if not question["answerable"]:
        return None  # 답 없음은 정답 청크가 없으므로 Recall/MRR 평균에서 제외한다.
    groups = [set(e["child_ids"]) for e in question["evidence"]]
    relevant = set.union(*groups)
    parent_gold = {e["parent_id"] for e in question["evidence"]}
    first = next((i for i, cid in enumerate(ranking[:10], 1) if cid in relevant), None)
    values = {"mrr@10": 1/first if first else 0.0}
    for k in (5, 10):
        hits = set(ranking[:k])
        covered = sum(bool(hits & group) for group in groups)
        # 같은 근거가 여러 청크에 반복되어도 근거 하나로 계산한다.
        values[f"evidence_recall@{k}"] = covered / len(groups)
        values[f"hit@{k}"] = float(bool(hits & relevant))
        values[f"all_evidence@{k}"] = float(covered == len(groups))
        retrieved_parents = {child_map[cid]["parent_id"] for cid in ranking[:k]}
        values[f"parent_recall@{k}"] = len(retrieved_parents & parent_gold) / len(parent_gold)
    return values


def summarize(rows):
    groups = {"all": rows}
    for field in ("split", "category"):
        for value in sorted({row[field] for row in rows}):
            groups[f"{field}:{value}"] = [r for r in rows if r[field] == value]
    result = {}
    for name, group in groups.items():
        answered = [r for r in group if r["metrics"] is not None]
        result[name] = {"questions": len(group), "answerable": len(answered),
            "unanswerable_inspect_only": len(group)-len(answered),
            "metrics": {key: sum(r["metrics"][key] for r in answered)/len(answered) for key in answered[0]["metrics"]} if answered else {},
            "mean_query_ms": sum(r["query_ms"] for r in group)/len(group)}
    return result


def evaluate(args):
    import numpy as np
    children = read_jsonl(args.artifacts / "children.jsonl")
    parents = {p["id"]: p for p in read_jsonl(args.artifacts / "parents.jsonl")}
    questions = read_jsonl(args.dataset)
    validate_dataset(questions, children, parents)
    questions = [q for q in questions if args.split == "all" or q["split"] == args.split]
    if not questions or not children:
        raise ValueError("질문 또는 검색 청크가 없음")
    if any(c["parent_id"] not in parents for c in children):
        raise ValueError("청크의 parent가 없음")
    ids = [c["id"] for c in children]
    child_map = {c["id"]: c for c in children}
    texts = [c["search_text"] for c in children]
    args.output.mkdir(parents=True, exist_ok=True)
    lexical = BM25(texts)
    lexical_rankings, lexical_times = [], []
    for q in questions:
        started = time.perf_counter()
        scores = lexical.scores(q["question"])
        lexical_rankings.append([ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i])) if scores[i] > 0][:args.candidates])
        lexical_times.append((time.perf_counter()-started)*1000)
    summaries = {}

    def record(name, rankings, times, extra=None):
        rows = [{"id": q["id"], "question": q["question"], "answerable": q["answerable"],
                 "split": q["split"], "category": q["category"], "metrics": metrics(r, q, child_map),
                 "query_ms": ms, "top10": [{"id": cid, "parent_id": child_map[cid]["parent_id"],
                    "text": child_map[cid]["search_text"]} for cid in r[:10]]}
                for q, r, ms in zip(questions, rankings, times)]
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        with (folder / "details.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {"groups": summarize(rows), **(extra or {})}
        save_json(folder / "summary.json", summary)
        summaries[name] = summary

    record("bm25", lexical_rankings, lexical_times, {"tokenizer": "word+korean_character_bigram"})
    for model in args.models:
        info = model_info(args.ollama, model)
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
        started = time.perf_counter()
        vectors = embed(texts, model, info, args.ollama, args.output / "cache" / (safe+".jsonl"), "document", args.batch_size)
        index_seconds = time.perf_counter()-started
        folder = args.output / safe
        folder.mkdir(exist_ok=True)
        # Qdrant 적재를 위해 벡터 순서와 child 레코드를 함께 보관한다.
        np.save(folder / "vectors.npy", vectors)
        with (folder / "records.jsonl").open("w", encoding="utf-8") as f:
            for child in children:
                f.write(json.dumps(child, ensure_ascii=False) + "\n")
        manifest = {"model": model, "digest": info["digest"], "dimensions": vectors.shape[1],
                    "distance": "Cosine", "normalized": True, "records": len(ids),
                    "input_format_version": 1, "index_seconds_including_cache": index_seconds}
        save_json(folder / "manifest.json", manifest)
        rankings, timings, hybrid_rankings, hybrid_times = [], [], [], []
        for i, q in enumerate(questions):
            started = time.perf_counter()
            query = embed([q["question"]], model, info, args.ollama, args.output / "cache" / (safe+"_queries.jsonl"), "query", 1)
            scores = vectors @ query[0]
            ranking = [ids[j] for j in np.argsort(-scores, kind="stable")[:args.candidates]]
            rankings.append(ranking)
            timings.append((time.perf_counter()-started)*1000)
            started_rrf = time.perf_counter()
            hybrid_rankings.append(rrf(lexical_rankings[i], ranking))
            hybrid_times.append(timings[-1]+lexical_times[i]+(time.perf_counter()-started_rrf)*1000)
        record(safe, rankings, timings, manifest)
        record(safe+"_hybrid", hybrid_rankings, hybrid_times, {"rrf_k": 60, "candidates_per_engine": args.candidates})
    save_json(args.output / "summary.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "children_sha256": hashlib.sha256((args.artifacts/"children.jsonl").read_bytes()).hexdigest(),
        "notes": ["검색만 평가하며 답변 생성과 답변 거부 판정은 평가하지 않음.", "query_ms includes cache lookup or Ollama query embedding; first load may be included", "parent_recall counts parents reached by top-k children"],
        "runs": summaries})
    lines = ["# 검색 평가 결과", "", "|검색 방식|답변 가능 문항|Hit@5|근거 Recall@10|MRR@10|Parent Recall@10|", "|---|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        group = summary["groups"]["all"]
        m = group["metrics"]
        lines.append(f"|{name}|{group['answerable']}|{m['hit@5']:.3f}|{m['evidence_recall@10']:.3f}|{m['mrr@10']:.3f}|{m['parent_recall@10']:.3f}|")
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/legal_rag"))
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--models", nargs="*", default=["qwen3-embedding:0.6b"])
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    args = parser.parse_args()
    if args.batch_size < 1 or args.candidates < 10:
        parser.error("batch-size는 1 이상, candidates는 10 이상")
    args.ollama = args.ollama.rstrip("/")
    evaluate(args)


if __name__ == "__main__":
    main()

