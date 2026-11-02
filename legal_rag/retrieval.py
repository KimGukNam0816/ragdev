"""Small integration helpers; dense vectors are supplied by your embedding model."""
import json
from pathlib import Path


def load_parents(path):
    return {p["id"]: p for p in (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines())}


def rrf(bm25_ids, dense_ids, k=60):
    """Fuse ordered child ID lists; use the same metadata filters in both engines."""
    scores = {}
    for ranking in (bm25_ids, dense_ids):
        for rank, child_id in enumerate(dict.fromkeys(ranking), 1):
            scores[child_id] = scores.get(child_id, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda child_id: (-scores[child_id], child_id))


def expand_parents(ranked_child_ids, children, parents, limit=5):
    """Deduplicate parents after child search/reranking."""
    lookup = {child["id"]: child["parent_id"] for child in children}
    result, seen = [], set()
    for child_id in ranked_child_ids:
        parent_id = lookup[child_id]
        if parent_id not in seen:
            result.append(parents[parent_id])
            seen.add(parent_id)
            if len(result) >= limit:
                break
    return result
