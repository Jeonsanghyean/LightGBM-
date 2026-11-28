"""화면 공통: 출처 선택, 데이터 적재, NULL·정밀도 표시 규칙."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from src import risk
from src.features_ext import add_extended
from src.store import PersonalStore

LABELS = {"us": "미국 Sparkov", "kr": "한국 AI-Hub", "personal": "개인 거래"}
PATHS = {"us": "data/research/us", "kr": "data/research/kr"}
DISPLAY_COLS = ["transaction_id", "card_id", "transaction_date", "transaction_time", "hour_bucket", "time_precision",
                "amount", "currency", "merchant_name", "merchant_category", "merchant_address", "region", "country",
                "latitude", "longitude", "location_precision", "transaction_type", "transaction_status", "is_fraud",
                "fraud_type", "risk_level", "provenance"]
FEAT_COLS = ["dist_prev_km", "gap_prev_min", "speed_kmh", "far_fast", "region_novelty", "region_changed",
             "is_overseas", "count_prev_hour_bucket", "count_same_day_prior", "same_merchant_24h"]
PRECISION_KO = {"second": "초 단위", "minute": "분 단위", "hour": "1시간대 단위", "day": "날짜만",
                "exact": "정확한 좌표", "synthetic_point": "합성 좌표 (실제 매장 위치 아님)",
                "province": "시도 단위", "country": "국가 단위", "none": "위치 없음"}
MISSING = "데이터에 없음"
US_SCORES = "outputs/test_scores.parquet"  # v1 실험이 만든 원본 Test 구간 점수 (화면 4와 같은 값)


def source_picker() -> str:
    cur = st.session_state.get("source", "us")
    src = st.radio("데이터", list(LABELS), index=list(LABELS).index(cur), format_func=LABELS.get,
                   horizontal=True, key=f"src_{st.session_state.get('_page', '')}")
    st.session_state["source"] = src
    return src


@st.cache_data(show_spinner="미국 데이터 불러오는 중…")
def _load_research(src: str) -> pd.DataFrame | None:
    base = PATHS[src]
    if not os.path.exists(f"{base}/transactions.parquet"):
        return None
    df = pd.read_parquet(f"{base}/transactions.parquet", columns=DISPLAY_COLS)
    if os.path.exists(f"{base}/features_v2.parquet"):
        f = pd.read_parquet(f"{base}/features_v2.parquet", columns=["transaction_id"] + FEAT_COLS)
        df = df.merge(f, on="transaction_id", how="left")
    if src == "kr" and os.path.exists(f"{base}/extra.parquet"):
        ex = pd.read_parquet(f"{base}/extra.parquet", columns=["transaction_id", "region_code"])
        df = df.merge(ex, on="transaction_id", how="left")
    if src == "us":
        df = attach_us_scores(df)
    return df


# [v2 변경 2026-11-28] 무엇을: 미국 거래에 v1 LightGBM 원본 Test 점수·위험 수준 연결
#   왜: 화면 2의 위험도가 "미산출 (3주차)"로 남아 있었음 / 어떻게: 화면 4와 같은 점수 파일·임계값 사용.
#   학습·검증 구간 거래는 점수를 내면 학습에 쓴 데이터를 평가하는 셈이라 비워 둔다.
def attach_us_scores(df: pd.DataFrame, path: str = US_SCORES) -> pd.DataFrame:
    df = df.drop(columns=["risk_level", "raw_score"], errors="ignore")
    if not os.path.exists(path):
        df["raw_score"], df["risk_level"] = np.nan, pd.Series(pd.NA, index=df.index, dtype="string")
        return df
    s = pd.read_parquet(path, columns=["trans_num", "score_LightGBM"]).rename(
        columns={"trans_num": "transaction_id", "score_LightGBM": "raw_score"})
    s["transaction_id"] = s["transaction_id"].astype(str)
    t_high, t_watch = risk.thresholds()
    s["risk_level"] = np.select([s["raw_score"] >= t_high, s["raw_score"] >= t_watch], ["높음", "주의"], "낮음")
    return df.merge(s, on="transaction_id", how="left")


def risk_text(v, src: str) -> str:
    """위험 수준이 없을 때 이유를 출처별로 밝힌다."""
    if isinstance(v, str) and v:
        return v
    return {"us": "미산출 (학습·검증 구간)", "personal": "미산출 (화면 6에서 분석 실행)"}.get(src, "미산출")


def load(src: str) -> pd.DataFrame | None:
    """출처별 거래 + 신규 특징. 개인 거래는 캐시하지 않고 매번 로컬 DB에서 읽는다."""
    if src == "personal":
        df = PersonalStore().query()
        if df.empty:
            return df
        return df.merge(add_extended(df), on="transaction_id", how="left")
    return _load_research(src)


def missing_notice(src: str) -> None:
    if src == "kr":
        st.warning("한국 AI-Hub 데이터 미확보 (B안 진행 중). 원본 CSV를 `data/research/kr/raw/`에 넣고 "
                   "`python -m src.sources.aihub_card`, `python -m src.build_features_v2`를 실행하면 이 화면에 표시됩니다.")
    elif src == "personal":
        st.info("저장된 개인 거래가 없습니다. ‘6. 개인 거래 입력’에서 먼저 등록하세요.")
    else:
        st.warning("미국 변환 데이터가 없습니다. `python -m src.sources.sparkov`를 실행하세요.")


def show(v) -> str:
    return MISSING if v is None or (not isinstance(v, str) and pd.isna(v)) or v == "" else str(v)


def time_text(r) -> str:
    if r["time_precision"] in ("second", "minute") and pd.notna(r["transaction_time"]):
        return str(r["transaction_time"])
    if r["time_precision"] == "hour" and pd.notna(r["hour_bucket"]):
        return f"{int(r['hour_bucket'])}시대"
    return MISSING


def money(amount, cur) -> str:
    if pd.isna(amount):
        return MISSING
    return f"{amount:,.0f} {cur}" if cur in ("KRW", "JPY") else f"{amount:,.2f} {cur}"


def region_text(r) -> str:
    if pd.notna(r.get("region")) and r.get("region"):
        return str(r["region"])
    if pd.notna(r.get("region_code", None)):
        return f"시도코드 {r['region_code']} (코드표 미확인)"
    return MISSING


def type_text(v) -> str:
    return {"online": "온라인", "offline": "오프라인"}.get(v, MISSING) if isinstance(v, str) else MISSING


def label_text(v) -> str:
    return MISSING if pd.isna(v) else ("이상(라벨)" if int(v) == 1 else "정상(라벨)")
