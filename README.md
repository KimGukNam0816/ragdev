# 한국 법령 PDF RAG 청킹

로컬 `ragdev` 프로젝트는 Codex 등록명이며 실제 경로는 `C:\project\reg_project`입니다. 조사 시 기존 코드는 없고 `data/`에 PDF 6개가 있었습니다. 원본 PDF를 수정하지 않는 별도 패키지입니다. Python 3.10 이상을 사용합니다.

## 실행

프로젝트 루트에서 PowerShell:

```powershell
uv venv
uv pip install --python .venv/Scripts/python.exe -e '.[test]'
.venv/Scripts/python.exe -m legal_rag --input data --output artifacts/legal_rag
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

uv가 없으면 `py -m venv .venv` 후 `.venv/Scripts/python.exe -m pip install -e '.[test]'`를 실행합니다. 설치 후 `legal-rag` 명령도 사용할 수 있습니다. 기본 입력은 `data`, 출력은 `artifacts/legal_rag`입니다. 다른 폴더 구조도 `--input`으로 지정할 수 있고 하위 폴더의 소문자 확장자 `.pdf`를 재귀 탐색합니다.

```powershell
.venv/Scripts/python.exe -m legal_rag --input data --output artifacts/legal_rag --max-chars 1000 --table-rows 8
```

`max-chars`는 본문 청크의 **문자 수** 목표치입니다. 토큰 수 제한이 아니며, 긴 항·호·목이나 표 행은 의미 보존을 위해 초과할 수 있습니다. 초과 청크에는 `oversize=true`를 기록합니다. 임베딩 모델에 넣기 전에 모델의 토크나이저로 길이를 검사하세요.

## 산출물

| 경로 | 내용 |
|---|---|
| `documents/<document_id>/raw_pages.json` | 페이지별 원시 텍스트·표, 좌표, 추출 방식 |
| `documents/<document_id>/document.json` | 장·조·부칙·별표 구조, 조문목록, 검증 결과 |
| `documents/<document_id>/document.md` | 사람이 검수할 정규화 Markdown |
| `parents.jsonl` | 조문·부칙 조항·별표 단위 원문 저장소; `id`가 키 |
| `children.jsonl` | 검색 청크, 상위 문맥, parent 연결, 메타데이터 |
| `bm25.jsonl` | BM25 엔진에 전달할 `id`, `parent_id`, `text`, 메타데이터 |
| `dense.jsonl` | 같은 ID와 텍스트를 갖는 임베딩 입력; **벡터는 미포함** |
| `report.json` | PDF별 페이지·표·청크 수, 조문목록 대조, 검토할 페이지 |

`search_text` 및 각 검색 입력의 `text`에는 법령명·장·조문 제목·부칙 버전을 붙입니다. `children.text`는 해당 조각입니다. `metadata.pages`는 1부터 시작하는 출처 페이지이며 텍스트 조각은 소속 텍스트 구간의 페이지 집합을 기록합니다. 좌표의 정확한 근거는 parent의 blocks와 raw_pages에 있습니다.

문서 ID는 PDF 바이트와 메타데이터의 해시, parent ID는 문서 ID와 구조 순서, child ID는 parent ID와 조각 순서입니다. 같은 파일·설정은 재실행해도 같은 ID가 나옵니다. 청킹 설정을 바꾸면 같은 child ID의 내용이 바뀔 수 있으므로 **전체 인덱스를 교체**하세요. 출력 JSONL은 매번 전체 교체합니다. 이전 실행의 `documents/` 하위 폴더는 남을 수 있으므로 현재 report의 document_id만 사용하거나 새 출력 폴더를 쓰세요. 여러 작업이 같은 출력 폴더에 동시에 쓰지 않도록 하세요.

## 구조와 청킹 기준

- PDF의 표 영역 안 문자는 본문 추출에서 제외하여 이중 색인을 줄입니다. 한 행뿐인 가짜 표 후보는 본문으로 보존합니다.
- 법제처 머리말·꼬리말, 제목 반복을 제거합니다. 실제 샘플의 제목 없는 조문목록 페이지는 제목만 나열되는 패턴으로 판별합니다. 목록은 별도 검색 청크로 넣지 않고 본문 조문 번호와 대조합니다.
- 조문 제목 또는 삭제 표시가 있는 줄만 조문 경계로 인정합니다. `제58조에 따른...` 같은 줄바꿈된 인용문은 새 조문으로 분리하지 않습니다.
- 짧은 조문은 하나의 child, 긴 조문은 항(①~㊿)·호(1.)·목(가.) 경계로 나눕니다. 하위 청크에는 상위 항·호의 도입문을 반복하여 조건을 유지합니다. 번호 없는 긴 문장은 강제로 자르지 않습니다.
- 부칙은 공포 버전마다 별도 구간으로 보존하고 조항 단위로 parent를 만듭니다. 과거 부칙도 검색 입력에 남깁니다. 날짜만으로 경과조치의 효력을 추정하지 않습니다.
- 별표/별지서식은 각각 parent를 만듭니다. 표는 행 묶음으로 나누며 헤더를 반복합니다. 다단 헤더는 Markdown에서 합치고 JSON의 원래 셀 행렬과 `null` 병합 정보를 그대로 보존합니다.
- 같은 별표에서 연속 페이지·같은 열 수·새 헤더가 없는 표는 앞 표의 헤더를 이어받습니다. 표 제목과 직전 본문 문맥도 포함합니다. 이는 추론이므로 report에 표 검토 경고를 남깁니다.

## 날짜와 버전

법령 전체의 `[시행 ...]`, 공포일, 공포번호를 추출합니다. 부칙에는 해당 공포번호·공포일을 별도로 기록하고 `effective_date`는 그 부칙의 기본 시행일로 덮어씁니다. 알 수 없으면 현재 법령 시행일을 대신 넣지 않고 `null`로 둡니다.

`effective_date_text`에는 시행일 조항 전체(다만, 예외 포함)를 저장합니다. 명시 날짜나 “공포/고시/발령한 날부터”는 기본 시행일을 추출하고, “공포 후 6개월” 같은 상대 날짜는 계산하지 않습니다. `date_scope=base_only`, `has_exceptions`, `validity=unreviewed`를 확인하세요. 시행일은 조문별 적용 여부나 경과조치 유효성의 확정 값이 아닙니다. 시점별 검색에는 검수한 조문별 적용 기간을 별도로 추가해야 합니다.

파일명에서 시행일을 추정하지 않습니다. 별도 목차 파일이나 추출 정보 보정은 상대 경로를 키로 하는 JSON을 사용합니다.

```json
{
  "별도_조문목록.pdf": {"document_kind": "toc", "law_title": "개인정보 보호법"},
  "별표목록.pdf": {"document_kind": "appendix_list", "law_title": "개인정보 보호법"},
  "법령본문.pdf": {"law_title": "검수한 법령명", "version": "검수한 버전"}
}
```

`--metadata metadata.json`으로 전달합니다. 별도 파일 사이 조문목록 자동 결합은 하지 않으므로 별도 목차는 JSON 결과를 이용해 대조하거나 본문과 같은 PDF를 쓰세요. 현재 6개 PDF는 목차와 본문이 같은 파일에 있습니다.

## OCR / VLM 확장

기본 실행은 네트워크나 외부 모델을 호출하지 않습니다. 텍스트가 드문 페이지, 표가 있는 페이지에서 `PageEnhancer`를 호출할 수 있습니다.

```python
# my_enhancer.py
from legal_rag.pipeline import Block

class Enhancer:
    def enhance(self, *, source, page, blocks, reasons):
        # page.to_image(resolution=180).original 로 PIL 이미지 생성 가능.
        # 로컬 OCR 또는 승인된 VLM으로 읽고 실제 결과를 Block으로 반환.
        # 문서에 없는 내용을 생성하지 말고 원본 좌표와 페이지를 유지.
        # return [Block(page.page_number, text, method="my-ocr"), ...]
        # 표는 kind="table", rows=[...], header_rows=2 등 지정.
        return None  # None이면 기본 결과 유지
```

```powershell
.venv/Scripts/python.exe -m legal_rag --enhancer my_enhancer:Enhancer
```

실제 OCR/VLM 구현과 모델 인증·비용 정책은 연결 지점으로 남겨두었습니다. 기본 추출은 스캔 문자를 복원하지 않습니다. report의 `sparse_text_requires_ocr`를 반드시 확인하세요. 병합 셀, 표 열 수가 바뀌는 페이지, 복잡한 서식은 원본 이미지와 JSON을 대조하고 필요하면 enhancer로 보정하세요. report에 enhancer_called가 있어도 해결 완료라는 뜻은 아닙니다.

## BM25 + dense 연결

BM25 엔진에 `bm25.jsonl`의 text를 한국어 분석기로 색인하고, 임베딩 모델로 `dense.jsonl`의 text를 벡터화해 벡터 DB에 저장합니다. 양쪽 모두 동일한 child ID를 사용하고 version, section_kind, supplement 등 메타데이터 필터도 동일하게 적용하세요. 이 패키지는 검색 서버·임베딩 모델·reranker를 강제 설치하지 않습니다.

```python
import json
from pathlib import Path
from legal_rag.retrieval import rrf, load_parents, expand_parents

root = Path("artifacts/legal_rag")
children = [json.loads(line) for line in (root / "children.jsonl").read_text(encoding="utf-8").splitlines()]
parents = load_parents(root / "parents.jsonl")
# bm25_ids = BM25 검색 결과 child ID 순위
# dense_ids = 벡터 검색 결과 child ID 순위
# ranked = rrf(bm25_ids, dense_ids)
# selected = reranker로 ranked의 child 본문 재정렬
# context = expand_parents(selected, children, parents, limit=5)
```

검색 child를 먼저 결합하고 rerank한 후 parent를 복원합니다. 긴 별표 parent는 모델 컨텍스트 제한에 맞게 검색 행과 관련 본문만 선택할 수 있습니다.

## 검증

2026-11-02 실제 입력 6개(총 118페이지)를 처리했습니다. `VALIDATION.md`에 실행 결과를 기록했습니다. 목차 대조는 조문 번호 집합의 일치 검사이며 모든 문장·표 셀의 정확성을 보장하는 검사는 아닙니다. 테스트는 삭제·가지조문, 줄바꿈 인용, 항·호·목 문맥, 부칙 버전/날짜 격리, 표 다단 헤더/연속 페이지, 목차 제외, 안정 ID, parent 복원, 빈 입력, OCR 호출을 확인합니다.
