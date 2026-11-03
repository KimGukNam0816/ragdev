"""저장된 하이브리드 검색 후보에 cross-encoder를 적용하고 전후를 비교한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .evaluate import BM25, embedding_input, metrics, read_jsonl, save_json, summarize, validate_dataset
from .retrieval import expand_parents, rrf


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_rankings(base, model, questions, children, dataset, children_file):
    # 이전 평가와 입력이 같아야 같은 후보로 rerank 효과만 비교할 수 있다.
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    if summary["dataset_sha256"] != fingerprint(dataset) or summary["children_sha256"] != fingerprint(children_file):
        raise ValueError("기준선과 평가셋/청크가 다릅니다. 기존 검색 평가부터 다시 실행하세요.")
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    folder = base / safe
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    records = read_jsonl(folder / "records.jsonl")
    if [c["id"] for c in records] != [c["id"] for c in children]:
        raise ValueError("벡터와 현재 child 순서가 다릅니다.")
    vectors = np.load(folder / "vectors.npy", allow_pickle=False)
    if vectors.shape != (len(children), manifest["dimensions"]) or not np.isfinite(vectors).all():
        raise ValueError("기준선 벡터의 크기 또는 값이 잘못되었습니다.")
    cached_queries = {row["key"]: row["vector"] for row in read_jsonl(base / "cache" / (safe+"_queries.jsonl"))}
    original = {row["id"]: row for row in read_jsonl(base / (safe+"_hybrid") / "details.jsonl")}
    count = summary["runs"][safe+"_hybrid"]["candidates_per_engine"]
    fusion_k = summary["runs"][safe+"_hybrid"]["rrf_k"]
    index = BM25([c["search_text"] for c in children])
    ids, rankings = [c["id"] for c in children], []
    for question in questions:
        text = embedding_input(model, question["question"], "query")
        key = hashlib.sha256((manifest["digest"]+"\0"+text).encode()).hexdigest()
        if key not in cached_queries or question["id"] not in original:
            raise ValueError(f"{question['id']}: 기준선 질문 캐시/결과가 없습니다.")
        query = np.asarray(cached_queries[key], dtype=np.float32)
        norm = np.linalg.norm(query)
        if query.shape != (vectors.shape[1],) or not np.isfinite(query).all() or norm == 0:
            raise ValueError("기준선 질문 벡터가 잘못되었습니다.")
        scores = index.scores(question["question"])
        lexical = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i])) if scores[i] > 0][:count]
        dense = [ids[i] for i in np.argsort(-(vectors @ (query/norm)), kind="stable")[:count]]
        ranking = rrf(lexical, dense, k=fusion_k)
        if ranking[:10] != [hit["id"] for hit in original[question["id"]]["top10"]]:
            raise ValueError(f"{question['id']}: 복원한 검색 결과가 이전 기준선과 다릅니다.")
        rankings.append(ranking)
    return rankings, {"model": model, "digest": manifest["digest"], "rrf_k": fusion_k,
                      "candidates_per_engine": count, "baseline_top10_verified": True}


def reorder(candidate_ids, scores):
    # 같은 점수면 기존 검색 순서를 유지한다. 정답 정보는 모델에 전달하지 않는다.
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(candidate_ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("reranker 점수의 개수 또는 값이 잘못되었습니다.")
    order = np.argsort(-scores, kind="stable")
    return [candidate_ids[i] for i in order]


class LocalReranker:
    def __init__(self, name, revision, device, max_length, batch_size, cache_path):
        import torch
        from huggingface_hub import model_info
        from sentence_transformers import CrossEncoder

        # 모델 버전을 고정하고, 파일은 프로젝트 artifacts 아래에 내려받는다.
        self.revision = model_info(name, revision=revision).sha
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = CrossEncoder(name, revision=self.revision, device=self.device,
            max_length=max_length, cache_folder=str(cache_path.parent / "models"),
            trust_remote_code=False,
            model_kwargs={"torch_dtype": torch.float16 if self.device.startswith("cuda") else torch.float32})
        self.model.eval()
        self.batch_size, self.max_length = batch_size, max_length
        self.cache_path = cache_path
        self.cache = {row["key"]: row["score"] for row in read_jsonl(cache_path)} if cache_path.exists() else {}
        self.config = {"model": name, "revision": self.revision, "max_length": max_length,
                       "device": self.device, "dtype": str(next(self.model.parameters()).dtype),
                       "score": "raw_logit", "input_format_version": 1}

    def scores(self, question, texts):
        import torch
        keys = [hashlib.sha256(json.dumps([self.config, question, text], sort_keys=True).encode()).hexdigest() for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        lookup = dict(zip(keys, texts))
        pairs = [(question, lookup[key]) for key in missing]
        if pairs:
            # 긴 법령의 뒷부분이 몰래 잘리지 않도록 토큰 길이를 먼저 검사한다.
            tokenizer = self.model.tokenizer
            encoded = tokenizer([q for q, _ in pairs], [t for _, t in pairs], truncation=False)
            longest = max(map(len, encoded["input_ids"]))
            if longest > self.max_length:
                raise ValueError(f"reranker 입력 {longest}토큰 > {self.max_length}. --max-length를 늘리세요.")
            values = self.model.predict(pairs, batch_size=self.batch_size,
                activation_fn=torch.nn.Identity(), show_progress_bar=False, convert_to_numpy=True)
            values = np.asarray(values).reshape(-1)
            reorder(missing, values)  # 점수 유효성 검사
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as file:
                for key, value in zip(missing, values):
                    self.cache[key] = float(value)
                    file.write(json.dumps({"key": key, "score": float(value)})+"\n")
        return [self.cache[key] for key in keys]


def compare(questions, rankings, children, parents, scorer, candidates):
    child_map = {child["id"]: child for child in children}
    before, after, details = [], [], []
    for question, ranking in zip(questions, rankings):
        selected = ranking[:candidates]
        started = time.perf_counter()
        scores = scorer.scores(question["question"], [child_map[cid]["search_text"] for cid in selected])
        reranked = reorder(selected, scores)
        elapsed = (time.perf_counter()-started)*1000
        base_metrics, new_metrics = metrics(ranking, question, child_map), metrics(reranked, question, child_map)
        common = {key: question[key] for key in ("id", "question", "split", "category", "answerable")}
        before.append({**common, "metrics": base_metrics, "query_ms": 0})
        after.append({**common, "metrics": new_metrics, "query_ms": elapsed})
        # 후보 안에 없는 근거는 reranker로 복구할 수 없으므로 한계도 기록한다.
        evidence = question["evidence"]
        ceiling = sum(bool(set(selected) & set(e["child_ids"])) for e in evidence)/len(evidence) if evidence else None
        def hits(ids):
            return [{"id": cid, "parent_id": child_map[cid]["parent_id"], "text": child_map[cid]["search_text"]} for cid in ids[:10]]
        details.append({**common, "candidate_ids": selected,
            "candidate_scores": dict(zip(selected, map(float, scores))), "candidate_evidence_recall": ceiling,
            "before": base_metrics, "after": new_metrics, "rerank_ms_including_cache": elapsed,
            "before_top10": hits(ranking), "after_top10": hits(reranked),
            "restored_parent_ids": [p["id"] for p in expand_parents(reranked[:10], children, parents, limit=10)]})
        print(f"rerank {question['id']}: {len(selected)} candidates", flush=True)
    return before, after, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/legal_rag"))
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/rerank"))
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    args = parser.parse_args()
    if args.candidates < 10 or args.max_length < 32 or args.batch_size < 1:
        parser.error("candidates >= 10, max-length >= 32, batch-size >= 1 필요")
    if args.output.resolve() == args.baseline.resolve():
        parser.error("기준선과 새 결과의 출력 폴더는 달라야 합니다.")
    children_file = args.artifacts / "children.jsonl"
    children = read_jsonl(children_file)
    parents = {p["id"]: p for p in read_jsonl(args.artifacts / "parents.jsonl")}
    questions = read_jsonl(args.dataset)
    validate_dataset(questions, children, parents)
    questions = [q for q in questions if args.split == "all" or q["split"] == args.split]
    if not questions or not any(q["answerable"] for q in questions):
        raise ValueError("평가할 답변 가능 질문이 없습니다.")
    rankings, base_info = restore_rankings(args.baseline, args.embedding_model, questions, children, args.dataset, children_file)
    args.output.mkdir(parents=True, exist_ok=True)
    print("기존 검색 top10 일치 확인 완료. reranker를 불러옵니다.", flush=True)
    scorer = LocalReranker(args.model, args.revision, args.device, args.max_length, args.batch_size, args.output / "scores.jsonl")
    before, after, details = compare(questions, rankings, children, parents, scorer, args.candidates)
    before_summary, after_summary = summarize(before), summarize(after)
    delta = {group: {key: after_summary[group]["metrics"][key]-value for key, value in record["metrics"].items()}
             for group, record in before_summary.items()}
    ceilings = [row["candidate_evidence_recall"] for row in details if row["candidate_evidence_recall"] is not None]
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "baseline": base_info,
        "reranker": scorer.config, "dataset_sha256": fingerprint(args.dataset), "children_sha256": fingerprint(children_file),
        "candidates": args.candidates, "candidate_evidence_recall": sum(ceilings)/len(ceilings),
        "before": before_summary, "after": after_summary, "delta": delta,
        "notes": ["답 없음은 점수에서 제외. 답변 생성/거부 판단은 미평가.",
                  "before query_ms=0은 재검색 지연시간 미측정 표시. after는 rerank/캐시 읽기 시간만 측정."]}
    save_json(args.output / "summary.json", summary)
    with (args.output / "details.jsonl").open("w", encoding="utf-8") as file:
        for row in details:
            file.write(json.dumps(row, ensure_ascii=False)+"\n")
    keys = ["hit@5", "evidence_recall@5", "evidence_recall@10", "mrr@10", "all_evidence@5"]
    lines = ["# Rerank 전후 비교", "", "|방식|Hit@5|근거 Recall@5|근거 Recall@10|MRR@10|전체 근거@5|", "|---|---:|---:|---:|---:|---:|"]
    for name, result in [("기존 하이브리드", before_summary), ("BGE rerank", after_summary)]:
        lines.append("|"+name+"|"+"|".join(f"{result['all']['metrics'][key]:.3f}" for key in keys)+"|")
    lines += ["", f"후보 {args.candidates}개 안의 근거 확보율: {summary['candidate_evidence_recall']:.3f}",
              "", "dev/test·유형별 결과와 차이는 summary.json, 문항별 전후 순위와 점수는 details.jsonl을 확인하세요.",
              "", "답 없음 5문항은 검색 지표 평균에서 제외하며, 이 결과는 현재 PDF/평가셋에만 해당합니다."]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
