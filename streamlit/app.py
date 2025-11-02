"""
졸프1 서비스(위장군 탐지 결과 대시보드) – Streamlit 구현

기능
1) GPS 기반 지도 시각화 (구역/히트맵/클러스터)
2) 탐지 이력(그리드+표)
3) 자동 보고서 요약 + 통계적 시각화 + 다운로드
4) 캡션 기반 검색(간단 TF‑IDF 유사도)

백엔드(API)는 이미 구현 중이라고 하셔서, 본 프런트는 다음 스키마를 가정합니다.
- GET {API_BASE_URL}/detections?from=YYYY-MM-DD&to=YYYY-MM-DD&limit=...&offset=...
  응답(JSON list): [
    {
      "id": str,
      "timestamp": "2025-10-25T13:45:00Z",
      "lat": float,
      "lon": float,
      "score": float,          # 신뢰도 0~1
      "class": "soldier",     # 또는 다른 클래스
      "image_url": "https://...",
      "thumbnail_url": "https://..." ,
      "caption": "숲길 위에 위장무늬 군인이 웅크려 있음",  # 백엔드 캡셔닝 결과
      "extra": {"model":"yolov8n","device":"binocular"}
    }, ...
  ]

로컬 데모가 필요할 경우, API 호출 실패 시 자동으로 목업 데이터를 생성합니다.
"""

from __future__ import annotations
import os
import io
import math
import json
import time
import random
import string
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
from pathlib import Path
from streamlit_pdf_viewer import pdf_viewer

import base64
import streamlit.components.v1 as components

import requests
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit.components.v1 import html

# =========================
# 0) 전역 설정
# =========================
ROOT = Path(__file__).parent
IMG_DIR    = ROOT / "result"
MANIFEST   = ROOT / "manifest.csv"

st.set_page_config(
    page_title="CamoSoldier Dashboard",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 👉 필요에 맞게 API 주소 수정하세요.
API_BASE_URL = os.environ.get("CAMO_API_BASE", "http://localhost:8000")

SEARCH_API_BASE_URL = os.environ.get("SEARCH_API_BASE", "http://localhost:8080")

# 지리 구역 해상도(도 단위 bin 크기) – 너무 작으면 타일 많아지고, 너무 크면 해상도 낮아짐
GEO_BIN = 0.005  # 약 0.005° ≈ 550m (위도 기준) 정도로 가정

# =========================
# 커스텀 필터 컴포넌트
# =========================
from dataclasses import dataclass
from datetime import date, time
from typing import Optional, Tuple, Dict, Any

@dataclass
class FilterParams:
    date_from: date
    date_to: date
    use_hour_filter: bool
    start_time: time
    end_time: time
    hour_range: str  # "ALL" 또는 "HH:MM-HH:MM"
    score_th: float
    search_query: str
    topk: int


def _to_hour_range(use_hour: bool, stime: time, etime: time) -> str:
    return "ALL" if not use_hour else f"{stime.strftime('%H:%M')}-{etime.strftime('%H:%M')}"


def _hr(use_hour: bool, stime: time, etime: time) -> str:
    return "ALL" if not use_hour else f"{stime.strftime('%H:%M')}-{etime.strftime('%H:%M')}"


def render_section_filter(
    key_prefix: str,
    base: FilterParams,
    title: str = "⚙️ 커스텀 필터",
    collapsed: bool = True,
) -> Tuple[FilterParams, bool]:
    """
    체크박스를 눌러야만 커스텀 필터가 활성화되는 섹션 전용 필터 패널.
    - key_prefix: 섹션 구분용 고유 키(예: "report", "grid", "table", "map")
    - base: 현재 글로벌 필터(기본값으로 사용)

    반환:
      (active_params, is_custom_enabled)
      - is_custom_enabled == False → active_params == base (글로벌 그대로)
      - is_custom_enabled == True  → 사용자 입력으로 덮어쓴 값
    """
    import streamlit as st

    with st.expander(title, expanded=not collapsed):
        use_custom = st.checkbox(
            "이 섹션에서 커스텀 필터 사용",
            key=f"{key_prefix}-use-custom",
            value=False,
            help="활성화: 커스텀 필터 적용 | 비활성화: 글로벌 필터(사이드바) 적용",
        )

        if not use_custom:
            # 참고용 요약만 보여주고 글로벌 그대로 리턴
            with st.container(border=True):
                st.caption("현재 적용 중인 글로벌 필터")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("기간"); st.code(f"{base.date_from} ~ {base.date_to}", language="text")
                with c2:
                    st.write("시간대"); st.code(base.hour_range, language="text")
                with c3:
                    st.write("최소 신뢰도"); st.code(f"{base.score_th:.2f}", language="text")
                c4, c5 = st.columns([2,1])
                with c4:
                    st.write("검색어"); st.code(base.search_query or "(없음)", language="text")
                with c5:
                    st.write("검색 상위 N"); st.code(str(base.topk), language="text")
            return base, False

        # --- 커스텀 UI (체크박스가 켜졌을 때만 노출) ---
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                dfrom = st.date_input("시작일(커스텀)", key=f"{key_prefix}-date-from", value=base.date_from)
            with c2:
                dto   = st.date_input("종료일(커스텀)", key=f"{key_prefix}-date-to", value=base.date_to)

            st.markdown("---")
            st.markdown("##### 🕒 시간대 필터(커스텀)")
            ch_use_hour = st.checkbox(
                "특정 시간대 지정",
                key=f"{key_prefix}-use-hour",
                value=base.use_hour_filter,
            )
            if ch_use_hour:
                col_a, col_b = st.columns(2)
                with col_a:
                    stime = st.time_input("시작 시간", key=f"{key_prefix}-start-time", value=base.start_time, step=300)
                with col_b:
                    etime = st.time_input("종료 시간", key=f"{key_prefix}-end-time", value=base.end_time, step=300)
                hour_range = _hr(True, stime, etime)
            else:
                stime, etime = base.start_time, base.end_time
                hour_range = "ALL"

            st.markdown("---")
            col_s, col_q, col_k = st.columns([1, 2, 1])
            with col_s:
                score = st.slider("최소 신뢰도(커스텀)", 0.0, 1.0, float(base.score_th), 0.05, key=f"{key_prefix}-score")
            with col_q:
                query = st.text_input("캡션 검색(커스텀)", key=f"{key_prefix}-query", value=base.search_query or "", placeholder="예) 숲 속, 바위, 눈 덮인, 덤풀 등")
            with col_k:
                k     = st.number_input("검색 상위 N(커스텀)", key=f"{key_prefix}-topk", min_value=5, max_value=200, value=int(base.topk), step=5)

        # 적용 버튼 없이도 현재 값이 즉시 반영되도록 설계(원하면 버튼 추가 가능)
        custom = FilterParams(
            date_from=dfrom,
            date_to=dto,
            use_hour_filter=ch_use_hour,
            start_time=stime,
            end_time=etime,
            hour_range=hour_range,
            score_th=float(score),
            search_query=query,
            topk=int(k),
        )
        return custom, True

def apply_filters(df: pd.DataFrame, params: FilterParams) -> pd.DataFrame:
    """FilterParams 기준으로 df를 필터링 (KST 날짜/시간대 + score + 캡션 검색)."""
    ts_kst = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")

    # 1. 기본 필터(날짜, 시간, 점수) 마스크 생성
    mask = (ts_kst.dt.date >= params.date_from) & (ts_kst.dt.date <= params.date_to)
    if params.use_hour_filter:
        mask &= _in_hour_mask(ts_kst, params.start_time, params.end_time)
    mask &= (df["score"] >= params.score_th)

    # 2. 기본 필터 우선 적용
    out = df[mask].copy()

    # 3. (수정) 캡션 검색어가 있으면, TF-IDF 대신 ES API 호출
    if params.search_query.strip():

        # 3-1. (수정) ES 캡션 검색 API 호출 경로 변경
        # main.py의 prefix="/searching"과 searching.py의 "/search"를 조합
        url = f"{SEARCH_API_BASE_URL}/searching/search"

        # ES API의 limit. searching.py의 기본값(200)을 사용하거나,
        # topk보다 넉넉하게 (e.g., 200~500) 가져와서 조인 후보군 확보
        es_limit = 200
        es_params = {"q": params.search_query, "limit": es_limit}

        try:
            # _safe_get은 List[Dict]를 반환함
            es_results_list = _safe_get(url, params=es_params, timeout=5)

            if not es_results_list:
                st.warning(f"'{params.search_query}'에 대한 캡션 검색 결과가 없습니다.")
                # 일치하는 ID가 없으므로 빈 DataFrame 반환
                return out.iloc[0:0]

                # 3-2. ES 결과를 DataFrame으로 변환: [ {"id": "id1", "_sim_score": 1.2}, ... ]
            df_es = pd.DataFrame(es_results_list)
            # searching.py에서 반환하는 키: "id", "_sim_score"
            # _sim으로 이름 변경
            df_es = df_es.rename(columns={"_sim_score": "_sim"})

            if df_es.empty or 'id' not in df_es.columns:
                st.warning(f"'{params.search_query}'에 대한 캡션 검색 결과가 없습니다 (API 반환 형식 오류).")
                return out.iloc[0:0]

            # 3-3. (수정) ES 결과와 기본 필터링 결과를 'id' 기준으로 inner join
            # ES 결과의 'id'는 str일 수 있으므로, 'out'의 'id'도 str로 맞춰서 조인
            out['id'] = out['id'].astype(str)
            df_es['id'] = df_es['id'].astype(str)

            joined = out.merge(df_es, on="id", how="inner")

            # 3-4. (수정) ES 유사도(_sim)로 정렬하고 사용자가 요청한 topk 적용
            out = joined.sort_values("_sim", ascending=False).head(params.topk)

        except Exception as e:
            st.error(f"ES 캡션 검색 API ({url}) 호출 중 오류: {e}")
            # 오류 발생 시 빈 결과 반환
            return out.iloc[0:0]

    # 4. 검색어가 없으면 기본 필터링 결과만 반환
    return out

def _in_hour_mask(ts_kst: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """KST datetime 시리즈에서 시간대 마스크 생성. 자정 넘김(예: 22:00~02:00) 지원."""
    hhmm = ts_kst.dt.time
    if start_t <= end_t:
        # 일반 구간 (예: 09:00~18:00)
        return (hhmm >= start_t) & (hhmm <= end_t)
    else:
        # 자정 넘김 (예: 22:00~02:00)
        return (hhmm >= start_t) | (hhmm <= end_t)

# =========================
# 유틸
# =========================

def _safe_get(url: str, params: dict | None = None, timeout: int = 10) -> List[Dict[str, Any]]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _mock_from_folder(center=(36.35, 127.30)) -> list[dict]:
    lat0, lon0 = center
    out = []
    now = datetime.now(timezone.utc)

    # 1) manifest.csv가 있으면 우선 사용
    if MANIFEST.exists():
        df = pd.read_csv(MANIFEST)
        for i, r in df.iterrows():
            fp = IMG_DIR / str(r["filename"])
            if not fp.exists():
                continue
            t = pd.to_datetime(r.get("timestamp", now), utc=True, errors="coerce")
            out.append({
                "id": str(r.get("id", f"mock-{i}")),
                "timestamp": (t or now).isoformat(),
                "lat": float(r.get("lat", lat0 + np.random.normal(0, 0.01))),
                "lon": float(r.get("lon", lon0 + np.random.normal(0, 0.01))),
                "score": float(r.get("score", np.clip(np.random.beta(5,2), 0, 1))),
                "class": str(r.get("class", "soldier")),
                # 로컬 파일은 file:// 대신 Streamlit가 바로 열 수 있도록 절대/상대 경로 문자열로 전달
                "image_url": str(fp),
                "caption": str(r.get("caption", "")),
                "extra": {"model": "mock", "device": "binocular"},
            })
        return out

    # 2) manifest가 없으면 폴더 스캔 + 랜덤 메타 생성
    candidates = sorted([p for p in IMG_DIR.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    captions = [
        "숲길에서 위장무늬 군인이 엎드려 있음",
        "수풀 가장자리에서 관측 장비를 든 인원",
        "바위 지형 사이에 은폐된 병력 추정",
        "위장망 근처에 움직임 포착",
    ]
    for i, fp in enumerate(candidates):
        ts = now - timedelta(minutes=random.randint(0, 60 * 24 * 14))
        out.append({
            "id": f"mock-{i}",
            "timestamp": ts.isoformat(),
            "lat": float(lat0 + np.random.normal(0, 0.01)),
            "lon": float(lon0 + np.random.normal(0, 0.01)),
            "score": float(np.clip(np.random.beta(5,2), 0, 1)),
            "class": "soldier",
            "image_url": str(fp),
            "caption": random.choice(captions),
            "extra": {"model": "mock", "device": "binocular"},
        })
    return out

def _mock_data(n: int = 300, center=(36.35, 127.30)) -> List[Dict[str, Any]]:
    """API 없을 때 데모용 목업."""
    lat0, lon0 = center
    out = []
    classes = ["soldier", "humanoid", "unknown"]
    devices = ["binocular", "dslr", "cctv"]
    models = ["yolov8n", "yolov8s", "yolov8n-int8"]
    captions = [
        "숲길에서 위장무늬 군인이 엎드려 있음",
        "수풀 가장자리에서 관측 장비를 든 인원",
        "바위 지형 사이에 은폐된 병력 추정",
        "위장망 근처에 움직임 포착",
        "멀리서 소규모 대형으로 이동하는 인원",
    ]
    now = datetime.now(timezone.utc)
    for i in range(n):
        lat = lat0 + np.random.normal(0, 0.01)
        lon = lon0 + np.random.normal(0, 0.01)
        score = max(0, min(1, np.random.beta(5, 2)))
        ts = now - timedelta(minutes=random.randint(0, 60 * 24 * 14))
        out.append({
            "id": f"mock-{i}",
            "timestamp": ts.isoformat(),
            "lat": float(lat),
            "lon": float(lon),
            "score": float(score),
            "class": random.choice(classes),
            "image_url": "https://picsum.photos/seed/{}".format(i),
            "thumbnail_url": "https://picsum.photos/seed/{}/256/256".format(i),
            "caption": random.choice(captions),
            "extra": {"model": random.choice(models), "device": random.choice(devices)},
        })
    return out


@st.cache_data(show_spinner=False)
def fetch_detections(date_from: str, date_to: str, limit: int = 3000) -> pd.DataFrame:
    url = f"{API_BASE_URL}/detections"
    data = _safe_get(url, params={"from": date_from, "to": date_to, "limit": limit})
    if not data:
        data = _mock_from_folder()
        # data = _mock_data(800)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # 타입 보정
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ("lat", "lon", "score"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 구역(타일) 라벨 생성
    df["tile_lat"] = (np.floor(df["lat"] / GEO_BIN) * GEO_BIN).round(6)
    df["tile_lon"] = (np.floor(df["lon"] / GEO_BIN) * GEO_BIN).round(6)
    df["tile_id"]  = df["tile_lat"].astype(str) + "," + df["tile_lon"].astype(str)
    # 날짜 단위
        # pandas 버전 호환: tz_convert의 nonexistent/ambiguous 미지원 대비
    _ts = df["timestamp"]
    # tz-naive면 UTC로 가정하여 로컬라이즈
    if _ts.dt.tz is None:
        try:
            _ts = _ts.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        except TypeError:
            # 구버전 호환: 인자 미지원 시 기본 동작
            _ts = _ts.dt.tz_localize("UTC")
    # Asia/Seoul로 변환 (구버전은 nonexistent/ambiguous 인자 미지원)
    try:
        _ts_kst = _ts.dt.tz_convert("Asia/Seoul", nonexistent="shift_forward", ambiguous="NaT")
    except TypeError:
        _ts_kst = _ts.dt.tz_convert("Asia/Seoul")

    df["date"] = _ts_kst.dt.date
    # 결측 제거
    df = df.dropna(subset=["lat", "lon", "timestamp"])
    return df


def make_hex_layer(df: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position='[lon, lat]',
        radius=120,  # meters; pydeck interprets in meters when elevation_scale given
        elevation_scale=5,
        elevation_range=[0, 100],
        extruded=True,
        pickable=True,
    )


def make_scatter_layer(df: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=8,
        get_fill_color=[255, 100, 60],
        pickable=True,
    )


def render_image_grid(df: pd.DataFrame, n_cols: int = 6, caption_with_score: bool = True):
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    cols = st.columns(n_cols)
    for i, row in enumerate(df.itertuples(index=False)):
        col = cols[i % n_cols]
        with col:
            txt = row.caption or "(no caption)"
            if caption_with_score:
                txt = f"{txt}\n(conf={row.score:.2f})"
            st.image(getattr(row, "thumbnail_url", None) or getattr(row, "image_url", None),
                     caption=txt, use_container_width=True)


def render_image_grid_interactive(df: pd.DataFrame, n_cols: int = 6):
    """이미지 그리드 – 모든 카드 높이 동일하게 + 캡션 2줄 제한"""
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    # 한 줄에 들어갈 평균 글자수 (조정 가능)
    per_line = {6: 38, 5: 46, 4: 58, 3: 78, 2: 120}.get(n_cols, 46)
    max_chars = per_line * 2  # 2줄 기준

    cols = st.columns(n_cols)
    for i, row in enumerate(df.itertuples(index=False)):
        col = cols[i % n_cols]
        with col:
            # 1️⃣ 이미지
            st.image(
                getattr(row, "thumbnail_url", None) or getattr(row, "image_url", None),
                use_container_width=True,
            )

            # 2️⃣ 캡션 2줄 제한 (글자 기준)
            raw_cap = getattr(row, "caption", "") or "(no caption)"
            if len(raw_cap) > max_chars:
                short_cap = raw_cap[:max_chars - 1].rstrip() + "…"
            else:
                short_cap = raw_cap

            # 3️⃣ 텍스트 영역 고정 높이 확보 (단차 제거)
            caption_box = st.empty()
            caption_box.markdown(
                f"<div style='height:3.2em; overflow:hidden;'>{short_cap}</div>",
                unsafe_allow_html=True
            )

            # 4️⃣ 신뢰도 / 클래스 표시
            meta = f"conf={getattr(row, 'score', 0):.2f}"
            st.caption(meta)

            # 5️⃣ 상세보기 버튼
            if st.button("상세/편집", key=f"detail-btn-{getattr(row, 'id', i)}"):
                st.session_state["selected_detection"] = {
                    k: getattr(row, k) if hasattr(row, k) else None
                    for k in df.columns
                }
                st.session_state["selected_detection"]["_is_from_grid"] = True
                st.session_state["_scroll_to"] = "sec-detail"
                st.toast("선택됨: 상세 화면으로 이동합니다.")
                st.rerun()


def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if df.empty:
        return out
    out["total"] = int(len(df))
    out["by_date"] = df.groupby("date").size().reset_index(name="count")
    out["by_tile"] = df.groupby(["tile_id", "tile_lat", "tile_lon"]).size().reset_index(name="count")
    out["by_class"] = df.groupby("class").size().reset_index(name="count") if "class" in df else pd.DataFrame()
    out["score_hist"] = df["score"].dropna()
    return out

# 시간대 정규화 헬퍼
def normalize_hour_range(hr: str | None) -> str | None:
    """
    - 'ALL' (대소문자 무시) 또는 None/빈문자열이면 시간 필터 미적용(None 반환)
    - 'HH:MM-HH:MM' 형태만 그대로 통과
    - 그 외 형식이면 시간 필터 미적용(None)
    """
    if hr is None:
        return None
    hr = str(hr).strip()
    if hr == "" or hr.upper() == "ALL":
        return None
    if "-" in hr and ":" in hr:
        return hr  # 예: "02:00-04:00", "23:00-02:00"
    return None

def render_report(stats: Dict[str, Any], global_params: FilterParams):
    if not stats:
        st.info("요약할 데이터가 없습니다.")
        return

    # 1) 섹션용 커스텀 필터 패널 (체크박스를 켜야 활성)
    report_params, is_custom = render_section_filter("stats", global_params)
    section_df = apply_filters(df_all, report_params)

    # 3) 집계 계산
    stats = compute_stats(section_df)
    if not stats:
        st.info("요약할 데이터가 없습니다.")
        return

    # 4) 상태 뱃지
    if is_custom:
        st.caption("✅ 커스텀 필터가 적용된 통계입니다.")
    else:
        st.caption("ℹ️ 글로벌 필터 기준 통계입니다.")

    st.markdown("#### 📊 탐지 통계 요약")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 탐지 수", f"{stats['total']:,}")
    with col2:
        last7 = stats["by_date"].tail(7)["count"].sum() if not stats["by_date"].empty else 0
        st.metric("최근 7일 탐지", f"{last7:,}")
    with col3:
        max_tile = stats["by_tile"].sort_values("count", ascending=False).head(1)
        tile_txt = max_tile["tile_id"].iat[0] if not max_tile.empty else "-"
        st.metric("최다 탐지 구역", tile_txt)

    # 추세
    if isinstance(stats.get("by_date"), pd.DataFrame) and not stats["by_date"].empty:
        fig_trend = px.bar(stats["by_date"], x="date", y="count", title="일자별 탐지 추세")
        st.plotly_chart(fig_trend, config={"width": "stretch"})

    # 점수 히스토그램
    if isinstance(stats.get("score_hist"), pd.Series) and not stats["score_hist"].empty:
        fig_hist = px.histogram(stats["score_hist"], nbins=20, title="신뢰도(score) 분포")
        st.plotly_chart(fig_hist, config={"width": "stretch"})

    # 보고서 텍스트(간단 자동 요약)
    top_tile = "-"
    if not stats["by_tile"].empty:
        t = stats["by_tile"].sort_values("count", ascending=False).iloc[0]
        top_tile = f"{t['tile_id']} (탐지 {int(t['count'])}건)"

    summary_md = f"""
    #### 📄 자동 요약
    - 총 탐지 건수: **{stats['total']:,}건**
    - 최근 7일 탐지: **{last7:,}건**
    - 최다 탐지 구역: **{top_tile}**
    - 일자별 추세 그래프와 클래스 분포, 신뢰도 분포를 참고하세요.
    """
    st.markdown(summary_md)

    # =========================
    # 3-추가) 보고서 생성(LLM) API 호출 + PDF 미리보기/다운로드
    # =========================
    # st.markdown("#### 📑 필터링된 데이터로 보고서 생성")
    # report_params, _ = render_section_filter("report", global_params)

    # 기본값들은 사이드바 필터와 연동
    default_title = "위장군인 탐지 결과 AI 자동 보고서"
    st.markdown("##### 보고서 제목")
    # report_title = st.text_input("보고서 제목", value=default_title)
    report_title = st.text_input(
        label="보고서 제목",          # 접근성용 라벨
        value=default_title,
        key="report_title",
        label_visibility="collapsed"  # 화면에서는 숨김
    )

 
    if st.button("✨ AI 자동 보고서 생성", type="primary"):
        hr = (
            "ALL"
            if not report_params.use_hour_filter
            else f"{report_params.start_time.strftime('%H:%M')}-{report_params.end_time.strftime('%H:%M')}"
        )

        # 현재 필터(캡션 검색 포함)를 적용하여 ID 목록 추출
        # 백단에서 조인해서 보낼 예정
        section_df = apply_filters(df_all, report_params)

        # ID 목록을 문자열 리스트로 변환 (JSON 직렬화를 위해)
        id_list = section_df['id'].astype(str).tolist()

        payload = {
            "period_from": str(report_params.date_from),
            "period_to": str(report_params.date_to),
            # "hour_range": normalize_hour_range(report_params.hour_range),  # "ALL" 또는 "HH:MM-HH:MM"
            "hour_range": hr,  # "ALL" 또는 "HH:MM-HH:MM"
            "min_conf": float(report_params.score_th),
            "data_path": "./data/detections_from_manifest.json",
            "title": report_title,
            "detection_ids": id_list
        }

        with st.spinner("AI가 보고서 생성 중..."):
            try:
                # url = f"{API_BASE_URL}/generate"
                url = "http://127.0.0.1:8080/generate"
                resp = requests.post(url, json=payload, timeout=300)

                # 2xx 확인
                resp.raise_for_status()

                pdf_bytes: bytes | None = None

                # 1) 바이너리 PDF로 바로 내려오는 경우
                ctype = resp.headers.get("Content-Type", "")
                if "application/pdf" in ctype.lower():
                    pdf_bytes = resp.content
                else:
                    # 2) JSON으로 내려오는 다양한 케이스 처리
                    try:
                        j = resp.json()
                    except Exception:
                        j = {}

                    # 2-1) base64 로 반환
                    b64 = j.get("pdf_base64")
                    if b64:
                        pdf_bytes = base64.b64decode(b64)

                    # 2-2) URL 로 반환(로컬 접근 가능한 경우)
                    elif j.get("pdf_url"):
                        try:
                            f = requests.get(j["pdf_url"], timeout=30)
                            f.raise_for_status()
                            if "application/pdf" in f.headers.get("Content-Type", "").lower():
                                pdf_bytes = f.content
                        except Exception as _:
                            pass

                    # 2-3) 파일경로로 반환 (FastAPI가 저장만 하고 경로를 준 경우)
                    elif j.get("pdf_path"):
                        try:
                            with open(j["pdf_path"], "rb") as fp:
                                pdf_bytes = fp.read()
                        except Exception as _:
                            pass

                if not pdf_bytes:
                    st.error("PDF를 받지 못했습니다. FastAPI 응답 형식을 확인하세요.")
                else:
                    # 세션에 저장해두면 rerun 후에도 유지됨
                    st.session_state["last_report_pdf"] = pdf_bytes
                    st.success("보고서 생성 완료! 아래에 미리보기를 표시합니다.")
            except requests.HTTPError as e:
                st.error(f"보고서 생성 실패(HTTP {resp.status_code}): {resp.text[:300]}")
            except Exception as e:
                st.error(f"보고서 생성 중 오류: {e}")

    # 세션에 PDF가 있다면 항상 미리보기 + 다운로드 노출
    pdf_buf = st.session_state.get("last_report_pdf")
    if pdf_buf:
        st.markdown("#### 📄 생성된 보고서 미리보기")
        # bytes 또는 base64 문자열 모두 지원
        pdf_viewer(pdf_buf, width=600, height=450)  # width="stretch" 대신 width=값
        st.download_button("📥 보고서 다운로드 (PDF)", data=pdf_buf,
                           file_name=f"report_{report_params.date_from}_to_{report_params.date_to}.pdf",
                           mime="application/pdf")

# =========================
# 사이드바 – 필터
# =========================
with st.sidebar:
    st.header("🔎 필터")

    today = datetime.now().date()
    default_from = today - timedelta(days=14)

    date_from = st.date_input("시작일", value=default_from)
    date_to   = st.date_input("종료일", value=today)

    st.subheader("🕒 시간대 필터")
    use_hour_filter = st.checkbox("특정 시간대 지정", value=False)

    if use_hour_filter:
        # 기본값: 00:00 ~ 23:59 (하루 전체에서 시작)
        start_time = st.time_input("시작 시간", value=time(0, 0), step=300)  # 5분 단위
        end_time = st.time_input("종료 시간", value=time(23, 59), step=300)

        # 문자열 파라미터(예: "02:00-04:00") – downstream 사용을 위해
        hour_range = f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}"
    else:
        # 필터 미사용 시 전체 시간대
        start_time = time(0, 0)
        end_time = time(23, 59)
        hour_range = "ALL"

    score_th  = st.slider("최소 신뢰도", 0.0, 1.0, 0.4, 0.05)

    search_query = st.text_input("캡션 검색", placeholder="예) 숲 속, 바위, 눈 덮인, 덤풀 등")
    topk = st.number_input("검색 상위 N", min_value=5, max_value=200, value=50, step=5)

    st.divider()

    st.markdown("### ✨ 바로가기")

    # ↘ 스트림릿에서 간혹 해시 스크롤이 막히는 환경을 대비한 버튼/JS 폴백
    def _goto(anchor: str):
        st.session_state["_scroll_to"] = anchor

    st.button("🗺️ 지도 시각화", on_click=_goto, args=("sec-map",))
    st.button("📍 구역별 탐지 집계", on_click=_goto, args=("sec-agg",))
    st.button("🖼️ 탐지 이력",   on_click=_goto, args=("sec-grid",))
    st.button("🧩 상세/편집",    on_click=_goto, args=("sec-detail",))
    st.button("📋 상세 표",    on_click=_goto, args=("sec-table",))
    st.button("📈 탐지 분석 & 보고서",  on_click=_goto, args=("sec-report",))

global_params = FilterParams(
    date_from=date_from,
    date_to=date_to,
    use_hour_filter=use_hour_filter,
    start_time=start_time,
    end_time=end_time,
    hour_range=hour_range,          # "ALL" 또는 "HH:MM-HH:MM"
    score_th=float(score_th),
    search_query=search_query or "",
    topk=int(topk),
)

# =========================
# 데이터 적재
# =========================
with st.spinner("데이터 불러오는 중..."):
    df_all = fetch_detections(str(date_from), str(date_to))

if df_all.empty:
    st.warning("데이터가 없습니다. API를 확인하거나 날짜 범위를 넓혀보세요.")
    st.stop()

# 필터 적용
mask = (df_all["score"] >= score_th)
filtered = df_all[mask].copy()

# 검색 적용(캡션 기반)
filtered_global = apply_filters(df_all, global_params)

# =========================
# API 업데이트/삭제 유틸
# =========================

def api_update_detection(item: Dict[str, Any]) -> Tuple[bool, str]:
    det_id = item.get("id")
    if not det_id:
        return False, "id가 없습니다."
    url = f"{API_BASE_URL}/detections/{det_id}"
    try:
        r = requests.put(url, json=item, timeout=10)
        if r.status_code // 100 == 2:
            return True, "업데이트 성공"
        return False, f"업데이트 실패: {r.status_code}"
    except Exception as e:
        # 목업 환경: 성공한 것으로 처리
        return True, f"(목업) 업데이트 처리: {e}"


def api_delete_detection(det_id: str) -> Tuple[bool, str]:
    url = f"{API_BASE_URL}/detections/{det_id}"
    try:
        r = requests.delete(url, timeout=10)
        if r.status_code // 100 == 2:
            return True, "삭제 성공"
        return False, f"삭제 실패: {r.status_code}"
    except Exception as e:
        # 목업 환경: 성공한 것으로 처리
        return True, f"(목업) 삭제 처리: {e}"

# =========================
# 1) 지도 시각화
# =========================
st.markdown("# 🪖 CamoSoldier 대시보드")
st.markdown("<span id='sec-map'></span>", unsafe_allow_html=True)
st.caption("GPS를 기반으로 구역 별 탐지 밀도를 시각화하고, 탐지 이력 및 자동 요약 보고서를 제공합니다.")

st.subheader("🗺️ 지도 시각화")

report_params, is_custom = render_section_filter("heatmap", global_params)
filtered = apply_filters(df_all, report_params)

mid_lat = float(filtered["lat"].median()) if not filtered.empty else 36.35
mid_lon = float(filtered["lon"].median()) if not filtered.empty else 127.30

# ✅ pydeck 직렬화 호환: datetime/tz -> 문자열로 변환한 얕은 사본 사용
map_cols = [c for c in ["lon", "lat", "class", "caption", "score", "timestamp"] if c in filtered.columns]
df_map = filtered[map_cols].copy()
if "timestamp" in df_map:
    try:
        df_map["timestamp"] = pd.to_datetime(df_map["timestamp"], errors="coerce").dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        df_map["timestamp"] = df_map["timestamp"].astype(str)

layers = [make_hex_layer(df_map), make_scatter_layer(df_map)]
view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12, pitch=40)
map_style = "mapbox://styles/mapbox/dark-v10" if os.getenv("MAPBOX_API_KEY") else None

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip={"text": "[{class}] {caption}\n(conf: {score})\n{timestamp}"},
    map_style=map_style,
)
st.pydeck_chart(deck, use_container_width=True)

# 타일(구역)별 집계 테이블
st.markdown("<span id='sec-agg'></span>", unsafe_allow_html=True)
st.divider()

st.markdown("### 📍 구역별 탐지 집계")
report_params, is_custom = render_section_filter("total per sector", global_params)
filtered = apply_filters(df_all, report_params)
stats = compute_stats(filtered)

if stats and not stats["by_tile"].empty:
    by_tile = stats["by_tile"].sort_values("count", ascending=False).reset_index(drop=True)
    st.dataframe(by_tile, use_container_width=True, height=320)
    st.download_button(
        "구역 집계 CSV 다운로드",
        data=by_tile.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"tile_summary_{date_from}_to_{date_to}.csv",
        mime="text/csv",
    )
st.markdown("<span id='sec-grid'></span>", unsafe_allow_html=True)

st.divider()

# =========================
# 2) 탐지 이력(그리드 + 표)
# =========================
st.subheader("🖼️ 탐지 이력")

report_params, is_custom = render_section_filter("image grid", global_params)
filtered = apply_filters(df_all, report_params)

INITIAL_COUNT = 18     # 기본 표시 개수 (3줄 × 6열)
STEP_COUNT    = 18     # 더보기/접기 단위
N_COLS        = 6

# 세션 상태 초기화
if "grid_visible_count" not in st.session_state:
    st.session_state["grid_visible_count"] = INITIAL_COUNT

visible_count = st.session_state["grid_visible_count"]

# 표시 데이터
visible_df = filtered.head(visible_count)
render_image_grid_interactive(visible_df, n_cols=N_COLS)

can_more      = len(filtered) > visible_count
can_collapse  = visible_count > INITIAL_COUNT

# 버튼 영역

wrap = st.container()
with wrap:
    st.markdown('<div class="camo-wide-row">', unsafe_allow_html=True)

    if can_more and can_collapse:
        # 반반 배치
        col_more, col_collapse_area = st.columns(2)
        with col_more:
            if st.button("🔽 더보기", use_container_width=True, key="grid-more"):
                st.session_state["grid_visible_count"] = min(visible_count + STEP_COUNT, len(filtered))
                st.rerun()

        with col_collapse_area:
            col_collapse, col_collapse_all = st.columns([8, 1])

            with col_collapse:
                if st.button("🔼 접기", use_container_width=True, key="grid-collapse"):
                    st.session_state["grid_visible_count"] = max(visible_count - STEP_COUNT, INITIAL_COUNT)
                    st.rerun()

            with col_collapse_all:
                # [신규] '모두 접기' 버튼 (클릭 시 INITIAL_COUNT로)
                if st.button("⏫", use_container_width=True, key="grid-collapse-all", help="모두 접기", on_click=_goto, args=("sec-grid",)):
                    st.session_state["grid_visible_count"] = INITIAL_COUNT
                    st.rerun()

    elif can_more and not can_collapse:
        # 더보기만 가능 → 100% 폭
        if st.button("🔽 더보기", use_container_width=True, key="grid-more"):
            st.session_state["grid_visible_count"] = min(visible_count + STEP_COUNT, len(filtered))
            st.rerun()

    elif can_collapse and not can_more:
        # [수정] '더보기'가 없을 때, '접기 영역'을 9:1로 분할
        col_collapse, col_collapse_all = st.columns([9, 1])

        with col_collapse:
            if st.button("🔼 접기", use_container_width=True, key="grid-collapse"):
                st.session_state["grid_visible_count"] = max(visible_count - STEP_COUNT, INITIAL_COUNT)
                st.rerun()

        with col_collapse_all:
            # [신규] '모두 접기' 버튼 (클릭 시 INITIAL_COUNT로)
            if st.button("⏫", use_container_width=True, key="grid-collapse-all", help="모두 접기", on_click=_goto, args=("sec-grid",)):
                st.session_state["grid_visible_count"] = INITIAL_COUNT
                st.rerun()


    st.markdown('</div>', unsafe_allow_html=True)
# 안내 문구
if len(filtered) <= INITIAL_COUNT:
    st.caption("표시할 결과가 18개 이하입니다.")
elif visible_count >= len(filtered):
    st.caption("모든 탐지 결과를 표시했습니다.")

# 상세/편집 패널
st.markdown("<span id='sec-detail'></span>", unsafe_allow_html=True)
st.divider()
st.markdown("### 🧩 상세/편집")
sel = st.session_state.get("selected_detection")
if sel:
    with st.container(border=True):
        c1, c2 = st.columns([1,2])
        with c1:
            st.image(sel.get("thumbnail_url") or sel.get("image_url"), use_container_width=True)
            st.caption(sel.get("caption", "(no caption)"))
        with c2:
            with st.form("edit-detection-form", clear_on_submit=False):
                id_val = st.text_input("ID", sel.get("id", ""), disabled=True)
                timestamp_val = st.text_input("timestamp", str(sel.get("timestamp", "")))
                lat_val = st.number_input("lat", value=float(sel.get("lat", 0.0)))
                lon_val = st.number_input("lon", value=float(sel.get("lon", 0.0)))
                score_val = st.number_input("score", min_value=0.0, max_value=1.0, value=float(sel.get("score", 0.0)), step=0.01)
                class_val = st.text_input("class", str(sel.get("class", "")))
                caption_val = st.text_area("caption", sel.get("caption", ""), height=100)
                extra_str = json.dumps(sel.get("extra", {}), ensure_ascii=False, indent=2) if isinstance(sel.get("extra"), (dict, list)) else str(sel.get("extra", ""))
                extra_val = st.text_area("extra(JSON)", extra_str, height=120)

                col_a, col_b, col_c = st.columns([1,1,2])
                save = col_a.form_submit_button("💾 저장")
                del_ = col_b.form_submit_button("🗑️ 삭제")
                cancel = col_c.form_submit_button("취소")

            if save:
                # JSON 파싱
                try:
                    extra_obj = json.loads(extra_val) if extra_val.strip() else {}
                except Exception:
                    st.error("extra는 유효한 JSON이어야 합니다.")
                    extra_obj = sel.get("extra", {})
                payload = {
                    "id": id_val,
                    "timestamp": timestamp_val,
                    "lat": lat_val,
                    "lon": lon_val,
                    "score": score_val,
                    "class": class_val,
                    "caption": caption_val,
                    "extra": extra_obj,
                    "image_url": sel.get("image_url"),
                    "thumbnail_url": sel.get("thumbnail_url"),
                }
                ok, msg = api_update_detection(payload)
                if ok:
                    st.success(msg)
                    # 캐시 무효화 후 새로고침
                    try:
                        fetch_detections.clear()
                    except Exception:
                        pass
                    st.session_state.pop("selected_detection", None)
                    st.rerun()
                else:
                    st.error(msg)

            if del_:
                ok, msg = api_delete_detection(sel.get("id", ""))
                if ok:
                    st.success(msg)
                    try:
                        fetch_detections.clear()
                    except Exception:
                        pass
                    st.session_state.pop("selected_detection", None)
                    st.rerun()
                else:
                    st.error(msg)

            if cancel:
                st.session_state.pop("selected_detection", None)
else:
    st.info("탐지 이력에서 항목의 '상세/편집' 버튼을 누르면 여기 상세 패널이 열립니다.")

st.markdown("<span id='sec-table'></span>", unsafe_allow_html=True)
st.divider()
st.markdown("### 📋 상세 표")
report_params, is_custom = render_section_filter("graph detail", global_params)
filtered = apply_filters(df_all, report_params)

show_cols = [c for c in ["timestamp", "class", "score", "lat", "lon", "caption", "image_url"] if c in filtered.columns]
st.dataframe(filtered[show_cols].sort_values("timestamp", ascending=False), use_container_width=True, height=420)

st.markdown("<span id='sec-report'></span>", unsafe_allow_html=True)
st.divider()

# =========================
# 3) 자동 보고서 요약 + 통계 시각화
# =========================
st.subheader("📈 탐지 분석 & 보고서")
render_report(stats, global_params)

st.divider()

# 폴백: 세션 상태에 목표 앵커가 있으면 스크롤 실행
if st.session_state.get("_scroll_to"):
    target = st.session_state["_scroll_to"]
    html(f"""
    <script>
      const el = window.parent.document.getElementById("{target}");
      if (el) {{
        el.scrollIntoView({{ behavior: "smooth", block: "start" }});
      }} else {{
        // 앵커가 아직 렌더링 중이면 약간 지연 후 재시도
        setTimeout(() => {{
          const e2 = window.parent.document.getElementById("{target}");
          if (e2) e2.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}, 100);
      }}
    </script>
    """, height=0)
    st.session_state["_scroll_to"] = None


st.caption("© 2025 CamoSoldier | Streamlit 대시보드")

