# Cross-encoder rerank 전후 비교

프로젝트 루트 `C:\project\reg_project`에서 실행합니다.

```powershell
uv run --inexact --extra rerank python -m legal_rag.rerank
```

기본 구성: **BM25 + BGE-M3 dense → RRF → 상위 50개 child → BGE-Reranker-v2-M3 → 상위 10개 → parent 복원**.

`legal_rag/rerank.py`가 핵심 코드이며 주요 로직에 한글 주석을 달았습니다. 이전 검색 평가의 벡터와 질문 캐시를 읽으므로 Ollama 서버를 다시 호출하거나 문서를 재임베딩하지 않습니다. 처음에는 Hugging Face에서 reranker 가중치를 내려받습니다. CUDA 사용이 가능하면 GPU, 아니면 CPU로 실행합니다. 이 프로젝트의 PyTorch 추가 의존성은 RTX 5060에서 사용할 CUDA 12.8 빌드로 설정했습니다.

## 결과

2026-11-02 최초 실측: 고정 50문항 중 답변 가능한 45문항 기준, 근거 Recall@5 **0.856 → 1.000**, Recall@10 **0.978 → 1.000**, MRR@10 **0.784 → 0.974**. 현재 자료/평가셋에 대한 결과이며 일반적인 법률 검색 성능을 보장하는 값은 아닙니다. dev/test별 수치도 summary.json에 보존했습니다.

RTX 5060 Laptop GPU에서 FP16, 배치 2로 실행했고 rerank 추가 시간은 질문당 평균 약 386ms였습니다(모델 로딩·이전 검색 제외). 테스트 20개 통과, 50문항의 이전 top10 순위 일치, 2,500쌍의 점수 캐시 재사용을 검증했습니다. 모델 파일은 공식 SHA-256과 대조했습니다.

- `artifacts/rerank/REPORT.md`: 전체 전후 비교표.
- `artifacts/rerank/summary.json`: dev/test·질문 유형별 점수와 변화량, 모델 revision, 후보 내 근거 확보율.
- `artifacts/rerank/details.jsonl`: 문항별 기존/새 순위, 후보 50개와 원점수, 복원한 parent ID.
- `artifacts/rerank/scores.jsonl`: 질문-청크 점수 캐시. 같은 설정으로 재실행하면 재사용.
- `artifacts/rerank/models/`: 다운로드한 모델 파일.

기존 `artifacts/evaluation` 결과와 청킹 산출물은 수정하지 않습니다. 이전 평가셋/청크의 해시가 다르거나 복원한 top10이 기존 결과와 다르면 중단합니다. 따라서 같은 후보에 reranker를 추가했을 때의 차이를 비교할 수 있습니다. 기존 기준선 폴더에 출력을 쓰는 것도 차단합니다.

## 다른 설정으로 실행

```powershell
# 개선 실험은 dev 30문항으로 실행
uv run --inexact --extra rerank python -m legal_rag.rerank --split dev --output artifacts/rerank_dev

# 설정을 정한 후 test 20문항으로 확인
uv run --inexact --extra rerank python -m legal_rag.rerank --split test --output artifacts/rerank_test

# 후보 수 비교: 30개로 줄이면 속도가 좋아질 수 있지만 근거가 빠질 수도 있음
uv run --inexact --extra rerank python -m legal_rag.rerank --candidates 30 --output artifacts/rerank_30

# 기존 Qwen3 임베딩 하이브리드에 같은 BGE reranker 적용
uv run --inexact --extra rerank python -m legal_rag.rerank --embedding-model qwen3-embedding:0.6b --output artifacts/rerank_qwen

# CPU 실행 / GPU 메모리가 부족할 때 배치 크기 조정
uv run --inexact --extra rerank python -m legal_rag.rerank --device cpu --batch-size 1 --output artifacts/rerank_cpu

# 테스트 (모델 다운로드 없이 20개 테스트)
uv run --inexact --extra eval --extra test python -m unittest discover -s tests -v
```

같은 출력 폴더에 여러 프로세스를 동시에 실행하지 마세요. `--inexact`는 프로젝트 밖에서 별도로 설치한 Python 패키지를 정리하지 않도록 지정합니다. 다른 모델이나 후보 수를 비교할 때는 출력 폴더도 바꾸세요. 가중치 revision은 실행 시 commit SHA로 기록하며 같은 버전 고정은 `--revision <SHA>`로 지정할 수 있습니다.

## 평가 기준

기존과 같은 50문항, 같은 정답 근거로 평가합니다. 핵심 지표는 **MRR@10, 근거 Recall@5, 전체 근거 확보율@5**입니다. Recall@10도 함께 기록하며 reranker가 언제나 개선된다는 가정은 하지 않습니다. 답 없음 5문항은 검색 점수에서 제외하고 결과만 남깁니다. 답변 생성/거부 판정 평가는 포함하지 않습니다.

후보 안에 없는 정답은 reranker가 복구할 수 없습니다. `candidate_evidence_recall`은 선택한 후보 수로 달성 가능한 근거 확보율의 한계를 나타냅니다. 정답 정보는 지표 계산에만 사용하고 reranker 입력에는 질문과 `search_text`만 전달합니다.

기본 최대 입력은 질문+문서 합계 4096토큰입니다. 먼저 토큰 길이를 검사하여 넘으면 오류로 알려주며 몰래 자르지 않습니다. 이 경우 `--max-length`를 늘리되 모델 최대 길이와 GPU 메모리를 확인하세요. 배치 기본값은 2입니다.

점수는 sigmoid 확률이 아닌 모델의 raw logit이며 답변 가능 확률로 해석하지 않습니다. 점수 캐시는 모델 commit, 장치, 정밀도, 입력 길이, 질문과 본문에 연결되어 있습니다. `rerank_ms_including_cache`는 모델 로딩과 이전 검색 시간을 제외한 rerank 또는 캐시 읽기 시간입니다. 기존 결과의 before query_ms=0은 지연시간을 다시 측정하지 않았다는 표시이므로 속도 비교에 쓰지 마세요.

모델 사용 방식: [BGE 공식 모델 카드](https://huggingface.co/BAAI/bge-reranker-v2-m3), [CrossEncoder 공식 문서](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).
