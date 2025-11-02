import json
import os
import re
import datetime as dt
from typing import List, Optional, Tuple
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
# (중요) main.py에 있던 styled_from_output.py를 import
from routers.styled_from_output import llm_report_text_to_pdf


# (수정) app 대신 router 사용
router = APIRouter()

# (수정) main.py의 유틸리티 함수들
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # main_combined.py 기준
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)
client = genai.Client()

REPORT_PROMPT = """
[역할]
너는 보안 정찰 리포트 자동 작성 도우미다.
입력 JSON의 각 항목은 '탐지 확정 샷'이며 다음 키를 가진다:
(id, timestamp, lat, lon, score, caption)

[목표]
- 한국어로 간결하고 정확한 분석 리포트를 작성한다.
- 모든 수치/사실은 반드시 입력 JSON으로부터 계산하여 사용한다(임의 생성 금지).
- 과도한 단정은 피하고 “~로 보임/가능성” 수준으로 기술한다.

[데이터 처리/집계 규칙]
1) 입력은 서버에서 사전 필터링된 레코드 배열(filtered)만 사용한다.
   - 총건수 = len(filtered)이며, 보고서에는 반드시 이 값을 사용한다.
   - 총건수는 서버가 주입한 "__TOTAL_COUNT__"와 일치해야 한다(직접 계산/추정 금지).
2) 시간대/요일 집계는 KST 기준으로 수행한다.
   - 날짜 포함 범위: {period_from} ~ {period_to} (양끝 포함).
   - 시간 히스토그램 bin: 정시 단위의 반개구간 [h, h+1) (h=0..23).
   - 요일 정의: 월/화/수/목/금/토/일 (KST).
3) 합계 일관성:
   - 시간별 합계의 총합 = __TOTAL_COUNT__
   - 요일별 합계의 총합 = __TOTAL_COUNT__
   - 만약 계산 결과 불일치가 발생하면 “검증/불일치 보고” 섹션에 불일치 값을 명시하되,
     표에는 재계산·추정으로 맞추지 말고 원계산 결과를 그대로 표기한다.
4) 반올림 규칙:
   - 평균 확신도(score)는 소수점 2자리 “반올림”으로 표기(예: 0.734 → 0.73, 0.735 → 0.74).
5) 좌표 요약:
   - lat/lon 최소값/최대값, 평균(소수점 5자리)을 보고.
6) 결측/파싱 실패:
   - timestamp/score 파싱 실패나 필수 키 누락 레코드는 집계에서 제외된 것으로 간주하며
     “검증/제외 통계”에 제외 건수만 보고(개별 레코드 나열 금지).
7) 중복 id:
   - 특별 지시 없으면 “중복 의심”만 보고하고 집계에는 그대로 포함한다(배제/병합 금지).


[출력 형식: Markdown]
# 기간 요약
- 기간: {period_from} ~ {period_to} (KST)
- 필터: 시간대(KST)={hour_range} / 최소 확신도={min_conf:.2f}
- 데이터 표준시: 입력 timestamp는 ISO8601(대개 UTC) 기반이며, 해석 과정에서 KST를 병기

# 핵심 지표
- 총 건수: __TOTAL_COUNT__  ← 그대로 출력(LLM 임의 계산/추정 금지)
- 평균 확신도(score): (입력/필터링 결과로부터 계산, 소수점 2자리)
- 피크 시간대(KST): (시간 히스토그램으로 계산)
- 좌표 분포 요약: (lat, lon의 범위 및 중심 경향 간단 요약)

# 패턴 분석
## 시간 분포 (KST) — 고정 형식
- 아래 표는 **반드시 24행(00시~23시)**을 모두 포함한다.
- **오름차순(00→23)**으로 정렬한다.
- **값은 정수만** 표기한다(예: 2). “2건”처럼 단위를 붙이지 않는다.
- 모든 행의 합은 **__TOTAL_COUNT__**와 정확히 일치해야 한다(맞추기 위해 임의 변경 금지, 불일치 시 아래 ‘검증/품질 체크’에서 보고).

| 시간대 (KST) | 탐지 건수 |
| :----------- | --------: |
| 00시 | 0 |
| 01시 | 0 |
| 02시 | 0 |
| 03시 | 0 |
| 04시 | 0 |
| 05시 | 0 |
| 06시 | 0 |
| 07시 | 0 |
| 08시 | 0 |
| 09시 | 0 |
| 10시 | 0 |
| 11시 | 0 |
| 12시 | 0 |
| 13시 | 0 |
| 14시 | 0 |
| 15시 | 0 |
| 16시 | 0 |
| 17시 | 0 |
| 18시 | 0 |
| 19시 | 0 |
| 20시 | 0 |
| 21시 | 0 |
| 22시 | 0 |
| 23시 | 0 |

## 요일 분포 (KST) — 고정 형식
- 아래 표는 **반드시 7행(월~일)**을 모두 포함한다.
- **월→화→수→목→금→토→일** 순으로 정렬한다.
- **값은 정수만** 표기한다(예: 14). “14건”처럼 단위를 붙이지 않는다.
- 모든 행의 합은 **__TOTAL_COUNT__**와 정확히 일치해야 한다(맞추기 위해 임의 변경 금지, 불일치 시 아래 ‘검증/품질 체크’에서 보고).

| 요일 (KST) | 탐지 건수 |
| :--------- | --------: |
| 월요일 | 0 |
| 화요일 | 0 |
| 수요일 | 0 |
| 목요일 | 0 |
| 금요일 | 0 |
| 토요일 | 0 |
| 일요일 | 0 |

## 확신도 분포(요약)
- 평균/최소/최대 확신도 등 간단 요약(필요시 구간별 개수도 제시하되 자유 형식).

## 장소적 시사점(선택)
- 좌표 범위/밀집 경향/지형 키워드 등 간단 기술.

# 대표 사례
- 아래에는 id(이미지명)만 백틱으로 3~6개 나열할 것. (예: `result_002870`)
- UTC/KST/score/caption 등은 출력하지 말 것. 서버가 후처리로 채운다.
- 예:
  - `result_002870`
  - `result_002948`
  - `result_002911`
  
# 위험/권고
- 관측 강화/배치/사각지 개선 등의 실행적 제안
- (예) 특정 시간대에 건수/평균 score가 높음 → 해당 시간대 관측 강화
- (예) 특정 좌표 범위에 밀집 → 그 구역 추가 관측 포인트 고려
- (예) 우천/야간 가정치 등 데이터 한계 명시
- (예) 데이터 한계(UTC→KST 변환, 캡션 해석 등) 명시

# 주의/한계
- 좌표 정밀도, 캡션 작성 방식 편향 등


# 검증/품질 체크
- 시간별 합계: Σ = (계산값) / 기대치 __TOTAL_COUNT__ → (일치/불일치)
- 요일별 합계: Σ = (계산값) / 기대치 __TOTAL_COUNT__ → (일치/불일치)
- 제외 통계: 파싱 실패/결측 등으로 제외된 레코드 건수 n
- 중복 의심: 동일 id 중복 발견 시 개수만 보고(세부 나열 금지)

[데이터]
다음은 JSON 배열이다. 이 데이터로 위 모든 수치/항목을 계산하여 작성하라.
"""


class GenerateRequest(BaseModel):
    period_from: str = Field(..., example="2025-10-10")
    period_to: str = Field(..., example="2025-10-16")
    hour_range: str = Field("02:00-04:00", example="02:00-04:00")
    min_conf: float = Field(0.70, ge=0.0, le=1.0)
    data_path: Optional[str] = Field(None, example="./data/detections_from_manifest.json")
    title: Optional[str] = Field(None, example="위장군인 자동 보고서(LLM 응답 기반)")
    # =================임장빈 수정 =====================
    detection_ids: Optional[List[str]] = Field(None, description="Streamlit에서 ES 캡션 검색 등으로 필터링된 ID 목록")
    # ================================================

class GenerateResponse(BaseModel):
    report_text_path: str
    report_pdf_path: str
    report_pdf_url: str
    created_at: str

def _force_total_count(report_text: str, total_count: int) -> str:
    # 1) 정상 경로: 플레이스홀더 치환
    if "__TOTAL_COUNT__" in report_text:
        return report_text.replace("__TOTAL_COUNT__", str(total_count))

    # 2) 백업 경로: "총 건수:" 줄을 정규식으로 교체
    # - 불릿/공백/한글 콜론 변형까지 허용
    pattern = r"(?im)^(?:[-*]\s*)?총\s*건수\s*[:：]\s*.*$"
    replacement = f"총 건수: {total_count}"
    if re.search(pattern, report_text):
        return re.sub(pattern, replacement, report_text, count=1)


    # 3) 그래도 없으면 '핵심 지표' 섹션 바로 아래에 삽입
    anchor = re.search(r"(?im)^#\s*핵심\s*지표\s*$", report_text)
    if anchor:
        idx = anchor.end()
        return report_text[:idx] + f"\n- 총 건수: {total_count}\n" + report_text[idx:]

    # 4) 마지막 안전망: 맨 앞에 한 줄 추가
    return f"- 총 건수: {total_count}\n\n{report_text}"

def _read_json(path: str):
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        if text.endswith("\n"):
            f.write(text)
        else:
            f.write(text + "\n")


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_hour_range(hour_range: str) -> Tuple[int, int]:
    try:
        left, right = hour_range.split("-")
        sh = int(left.split(":")[0])
        eh = int(right.split(":")[0])
        return sh, eh
    except Exception:
        return 0, 24


KST = dt.timezone(dt.timedelta(hours=9))


def _to_dt(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d


def _in_date_range_kst(d: dt.datetime, start_date: dt.date, end_date: dt.date) -> bool:
    d_kst = d.astimezone(KST)
    return start_date <= d_kst.date() <= end_date


def _in_hour_range_kst(d: dt.datetime, start_h: int, end_h: int) -> bool:
    h = d.astimezone(KST).hour
    if start_h <= end_h:
        return start_h <= h < end_h
    else:
        return (h >= start_h) or (h < end_h)


def _filter_records(records: List[dict], period_from: str, period_to: str, hour_range: str, min_conf: float) -> List[
    dict]:
    sd = dt.date.fromisoformat(period_from)
    ed = dt.date.fromisoformat(period_to)
    sh, eh = _parse_hour_range(hour_range)
    out = []
    for r in records:
        try:
            ts = _to_dt(r["timestamp"])
            score = float(r.get("score", 0))
        except Exception:
            continue
        if score < min_conf:
            continue
        if not _in_date_range_kst(ts, sd, ed):
            continue
        if not _in_hour_range_kst(ts, sh, eh):
            continue
        out.append(r)
    return out


@router.get("/health")
def health():
    return {"status": "ok", "time": dt.datetime.now().isoformat(timespec="seconds")}


# (수정) @app.post -> @router.post
@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    1) CSV-파생 JSON 읽기
    2) 서버 측 필터링
    3) Gemini 호출로 report_text 생성
    4) report.txt 저장
    5) styled_from_output으로 PDF 생성
    6) PDF 직접 반환(FileResponse)
    """
    try:
        data_path = req.data_path or os.path.join(DATA_DIR, "detections_from_manifest.json")
        detections = _read_json(data_path)
        if not isinstance(detections, list):
            raise ValueError("데이터는 JSON 배열이어야 합니다.")

        filtered = _filter_records(
            detections,
            period_from=req.period_from,
            period_to=req.period_to,
            hour_range=req.hour_range,
            min_conf=req.min_conf,
        )
        if len(filtered) == 0:
            raise HTTPException(status_code=400, detail="필터 결과가 비었습니다. 기간/시간대/min_conf를 완화해보세요.")
        # (신규) Streamlit이 전달한 ID 목록이 있으면, 2차 필터링 수행
        # ===============임장빈 수정 ================
        if req.detection_ids is not None:
            # ID 목록을 Set으로 만들어 검색 속도 향상
            id_set = set(req.detection_ids)
            # filtered 목록에서 id가 id_set에 포함된 항목만 남김
            filtered = [r for r in filtered if str(r.get('id')) in id_set]
        # ===========================================
        prompt_text = REPORT_PROMPT.format(
            period_from=req.period_from,
            period_to=req.period_to,
            hour_range=req.hour_range,
            min_conf=req.min_conf,
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_text, json.dumps(filtered, ensure_ascii=False)],
        )

        if not hasattr(response, "text") or not response.text:
            raise RuntimeError("LLM이 빈 응답을 반환했습니다.")

        report_text = response.text

        ts = _timestamp()
        base = f"report_{ts}"
        txt_path = os.path.join(OUT_DIR, f"{base}.txt")
        pdf_path = os.path.join(OUT_DIR, f"{base}_styled.pdf")

        _write_text(txt_path, report_text)

        title = req.title
        first = report_text.strip().splitlines()[0] if report_text.strip() else ""
        if not title and first.startswith("#"):
            title = first.lstrip("#").strip()
        if not title:
            title = "위장군인 자동 보고서(LLM 응답 기반)"

        llm_report_text_to_pdf(report_text, out_path=pdf_path, title=title, records=filtered)

        pdf_url = f"/files/{os.path.basename(pdf_path)}"

        # (수정) response_model 대신 FileResponse 반환
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(pdf_path),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"보고서 생성 실패: {e}")
