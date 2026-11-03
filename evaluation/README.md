# 평가 실행

Cross-encoder reranker와 기존 결과 비교는 `RERANK.md`를 참고하세요. 실행 명령은 `uv run --inexact --extra rerank python -m legal_rag.rerank`입니다.

프로젝트 루트 `C:\project\reg_project`에서 실행합니다. Ollama 서버가 켜져 있어야 합니다.

```powershell
# Qwen3 기준선: BM25 + dense + BM25/dense RRF를 한 번에 평가
uv run --extra eval python -m legal_rag.evaluate

# 설치된 두 모델 비교
uv run --extra eval python -m legal_rag.evaluate --models qwen3-embedding:0.6b bge-m3

# Ollama 호출 없이 BM25만 평가 (기존 결과를 보존하도록 다른 출력 폴더)
uv run --extra eval python -m legal_rag.evaluate --models --output artifacts/evaluation_bm25

# 개선 실험은 dev 30문항으로 수행
uv run --extra eval python -m legal_rag.evaluate --split dev --output artifacts/evaluation_dev

# 최종 비교는 고정된 test 20문항으로 수행
uv run --extra eval python -m legal_rag.evaluate --split test --output artifacts/evaluation_test

# 테스트 (Ollama 없이 실행 가능)
uv run --extra eval --extra test python -m unittest discover -s tests -v
```

## 코드와 평가셋

- `legal_rag/evaluate.py`: BM25, Ollama 임베딩, 코사인 검색, RRF, 지표 계산. 핵심 부분에 한글 주석.
- `questions.jsonl`: 질문, 기대 답변, 근거 원문, PDF 파일/페이지/해시, 정답 parent/child ID.
- `QUESTIONS.md`: 사람이 읽는 50문항과 기대 답변/근거.
- `source_audit.json`: 근거 51곳이 원본 PDF 페이지의 텍스트 또는 표에서 확인됐는지 기록.

문항 구성: 조문 직접 질문 15 / 표현을 바꾼 질문 10 / 표 질문 10 / 부칙 질문 5 / 복수 근거 질문 5 / 답 없음 5. dev 30 / test 20으로 고정했습니다. 각 질문은 보유한 PDF 버전의 내용을 묻습니다. 현재 시점의 법률 적용 여부를 판정하는 평가셋은 아닙니다. 기대 답변은 검색 근거 설명용이며 자동 생성 답변 채점은 아직 하지 않습니다.

문항은 자동 생성한 초안을 그대로 사용하지 않고 원문 구절과 연결해 작성했습니다. 독립된 법률 전문가가 검수한 공인 벤치마크는 아니므로, 실사용 질문과 추가 정답 근거를 누적하세요. 동일 답을 뒷받침하는 다른 근거가 발견되면 evidence를 보완해야 합니다.

## 결과 읽기

`artifacts/evaluation/REPORT.md`가 전체 비교표입니다. `summary.json`에는 dev/test와 질문 유형별 세부 평균도 있습니다. 방식별 `details.jsonl`에서 문항별 점수와 실제 상위 10개 검색 청크를 확인합니다.

| 지표 | 의미 |
|---|---|
| Hit@5, @10 | 상위 k개에서 정답 근거 청크를 하나라도 찾은 질문 비율 |
| evidence_recall@5, @10 | 필요한 근거 그룹 중 직접 답을 담은 child를 찾은 비율 |
| all_evidence@5, @10 | 여러 근거가 필요한 질문에서 근거를 모두 찾았는지 |
| MRR@10 | 첫 정답 child 순위의 역수 평균 |
| parent_recall@5, @10 | 상위 k개 child의 parent를 복원했을 때 정답 parent 확보 비율 |

같은 근거를 여러 child가 반복해 담을 수 있어, Recall은 개별 child 개수가 아니라 근거별 그룹으로 계산합니다. 정답과 같은 parent의 다른 child를 찾은 것은 child 근거 검색 성공으로 처리하지 않습니다. parent 확장 지표에서만 성공으로 잡힙니다. parent_recall은 parent 본문 전체가 모델 입력에 들어간다는 가정이며 컨텍스트 길이를 제한하는 최종 RAG 성능과 다릅니다.

답 없음 5문항은 Recall/MRR의 분모에서 제외하고 검색 결과만 저장합니다. 검색은 무관한 자료도 반환할 수 있으므로 답변 거부 성능은 생성/판정 단계에서 별도로 평가해야 합니다. 아직 거부 임계값을 조정하거나 거부 정확도를 보고하지 않습니다.

BM25는 의존성을 줄이기 위해 단어+한글 2글자 조각 분석을 사용한 간단한 기준선입니다. 형태소 분석기 기반 운영 BM25와 점수를 혼동하지 마세요. dense는 Qdrant 없이 전체 벡터를 정확 코사인 검색합니다. 이 기준선으로 나중에 Qdrant ANN 검색이 근거를 놓치는지도 비교할 수 있습니다. Hybrid는 각 검색 상위 50개를 RRF(k=60)로 합칩니다.

## 임베딩과 재실행

- Qwen3는 질문에만 검색 지시문을 붙이고, BGE-M3는 별도 접두어 없이 사용합니다. E5 계열도 선택하면 query/passage 접두어를 적용합니다.
- 입력은 `children.jsonl`의 `search_text`입니다 (`dense.jsonl`의 text와 같은 내용). parent는 임베딩하지 않습니다.
- `truncate=false`: 긴 텍스트를 몰래 자르지 않고 Ollama 오류로 알립니다. 길이 오류 시 해당 청크 구조를 확인하거나 모델/컨텍스트 설정을 조정해야 합니다.
- 모델 digest와 실제 입력 해시로 캐시합니다. 완료한 배치를 계속 저장하므로 같은 명령을 다시 실행하면 이어서 처리합니다. 같은 출력 폴더에 여러 실행을 동시에 쓰지 마세요.
- `query_ms`는 질문 임베딩 또는 캐시 읽기와 검색을 포함합니다. 첫 모델 로딩, 캐시 유무가 섞이므로 순수 추론 속도 비교로 사용하지 마세요. 동일 환경에서 별도 속도 측정이 필요합니다.
- 재청킹하여 ID/근거가 달라지면 검증에서 중단합니다. 원문 근거를 새 child와 다시 연결한 뒤 평가하세요. 평가셋을 바꾸면 이전 기준선과 직접 비교하지 말고 새 버전으로 기록하세요.

## Qdrant에 나중에 올릴 파일

모델별 폴더(예: `artifacts/evaluation/qwen3-embedding_0.6b`)에 다음을 저장합니다.

- `vectors.npy`: L2 정규화한 float32 벡터, 행 하나가 child 하나.
- `records.jsonl`: 벡터와 동일 순서의 child ID, parent ID, 텍스트, 메타데이터.
- `manifest.json`: 모델명/digest, 벡터 차원, 거리 설정(Cosine), 청크 개수.

Qdrant point ID는 uint64 또는 UUID를 쓰므로 현재 24자리 child 해시를 그대로 point ID로 쓰지 마세요. 나중에 `uuid.uuid5(uuid.NAMESPACE_URL, child_id)` 등으로 변환하고 원래 child ID와 parent ID는 payload에 보존하면 됩니다. Qdrant 업로드는 이번 코드에서 수행하지 않습니다.
