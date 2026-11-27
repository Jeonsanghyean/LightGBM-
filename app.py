"""LightGBM 기반 카드 이상거래 탐지 — 연구용 대시보드.

실행: streamlit run app.py
결과 CSV(outputs/)와 저장된 모델(models/)만 읽는다. 실시간 결제 연동은 범위 밖이다.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import ALL_FEATURES, build_features, prepare

st.set_page_config(page_title="카드 이상거래 탐지", layout="wide")
MODELS = ["LightGBM", "XGBoost", "LogisticRegression"]
RATIOS = [5, 20, 100]
FEATURE_KO = {
    "amt_ratio": "고액 결제 비율 (현재 금액 ÷ 과거 중앙값)",
    "count_5min": "직전 5분 거래 건수",
    "count_15min": "직전 15분 거래 건수",
    "count_60min": "직전 60분 거래 건수",
    "hour_rarity": "시간대 희소성 (1 − 과거 동일 시간대 비율)",
    "insufficient_history": "이력 부족 (과거 5건 미만)",
}


@st.cache_data
def load_csv(name):
    return pd.read_csv(f"outputs/{name}")


@st.cache_data
def load_scores():
    return pd.read_parquet("outputs/test_scores.parquet")


@st.cache_resource
def load_models():
    return {m: joblib.load(f"models/{m}.joblib") for m in MODELS}


@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_uploaded(raw: pd.DataFrame) -> pd.DataFrame:
    stats = load_json("models/global_stats.json")
    feat = build_features(prepare(raw), stats)
    for m, model in load_models().items():
        feat[f"score_{m}"] = model.predict_proba(feat[ALL_FEATURES])[:, 1]
    return feat


thresholds = load_json("models/thresholds.json")

st.title("LightGBM 기반 카드 이상거래 탐지")
st.caption("Sparkov 모의 카드 거래 · 비용은 가정한 상대 단위(오탐 1건 = 1) · 위험 점수는 예측값이며 사기 확정 판정이 아닙니다.")

tab_in, tab_risk, tab_cmp, tab_thr, tab_exp = st.tabs(
    ["① 거래 데이터 입력", "② 위험도 확인", "③ 모델 비교", "④ 임계값 분석", "실험 결과"]
)

# ① 거래 데이터 입력 ------------------------------------------------------
with tab_in:
    src = st.radio("데이터 출처", ["예시: 원본 Test (555,719건)", "CSV 업로드 (Sparkov 형식)"], horizontal=True)
    if src.startswith("예시"):
        data = load_scores()
    else:
        up = st.file_uploader("trans_date_trans_time, cc_num, amt, trans_num 열이 있는 CSV", type="csv")
        st.caption("업로드한 파일 안의 거래만 카드 이력으로 사용합니다. 이력이 5건 미만이면 학습 구간 전체 통계로 대체합니다. "
                   "예시 파일: data/sample_upload.csv")
        if up is None:
            st.stop()
        raw = pd.read_csv(up)
        need = {"trans_date_trans_time", "cc_num", "amt", "trans_num"}
        if not need.issubset(raw.columns):
            st.error(f"필수 열이 없습니다: {sorted(need - set(raw.columns))}")
            st.stop()
        if "is_fraud" not in raw.columns:
            raw["is_fraud"] = np.nan
        data = score_uploaded(raw.loc[:, ~raw.columns.str.startswith("Unnamed")])

    c1, c2 = st.columns([2, 1])
    query = c1.text_input("거래 ID(trans_num) 검색", placeholder="일부만 입력해도 됩니다")
    only_fraud = c2.checkbox("실제 사기 거래만 보기", value=False)
    view = data
    if query:
        view = view[view["trans_num"].astype(str).str.contains(query.strip(), regex=False)]
    if only_fraud:
        view = view[view["is_fraud"] == 1]
    st.write(f"검색 결과 {len(view):,}건 (최대 500건 표시)")
    cols = ["trans_num", "trans_date_trans_time", "cc_num", "category", "amt", "is_fraud", "score_LightGBM"]
    cols = [c for c in cols if c in view.columns]
    shown = view[cols].head(500).reset_index(drop=True)
    st.dataframe(shown, width="stretch", height=300)
    if len(shown):
        pick = st.selectbox("확인할 거래 선택", shown["trans_num"].tolist())
        st.session_state["selected"] = view[view["trans_num"] == pick].iloc[0].to_dict()
        st.success("선택 완료 → ② 위험도 확인 탭에서 결과를 보세요.")

# ② 위험도 확인 ----------------------------------------------------------
with tab_risk:
    row = st.session_state.get("selected")
    if row is None:
        st.info("① 탭에서 거래를 먼저 선택하세요.")
    else:
        c1, c2 = st.columns(2)
        model = c1.selectbox("모델", MODELS, key="risk_model")
        r = c2.selectbox("비용 비율 (미탐:오탐)", RATIOS, index=1, format_func=lambda x: f"{x}:1", key="risk_r")
        score = float(row[f"score_{model}"])
        t = thresholds[f"{model}|{r}"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("모델 점수", f"{score:.4f}")
        m2.metric("고정 임계값 (검증 선택)", f"{t:.4f}")
        m3.metric("경고 판정", "사기 의심" if score >= t else "정상")
        m4.metric("기본 0.5 기준", "사기 의심" if score >= 0.5 else "정상")
        label = row.get("is_fraud")
        if label is not None and not pd.isna(label):
            st.write(f"데이터셋 정답 라벨(is_fraud): **{'사기' if int(label) == 1 else '정상'}**")
        st.subheader("거래 특징값")
        st.table(pd.DataFrame({"특징": [FEATURE_KO[f] for f in ALL_FEATURES],
                               "값": [round(float(row[f]), 4) for f in ALL_FEATURES]}))
        st.caption(f"거래 {row['trans_num']} · 금액 {row['amt']} · 시각 {row['trans_date_trans_time']} · 과거 거래 {int(row['n_prev'])}건")

# ③ 모델 비교 ------------------------------------------------------------
with tab_cmp:
    st.subheader("원본 Test 전체 (임계값 0.5, 양성 가중치 10)")
    tm = load_csv("test_model_metrics.csv")
    st.dataframe(tm[["model", "AP_test", "Recall", "F1", "Precision", "TP", "FP", "FN"]]
                 .style.format({"AP_test": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}"}),
                 width="stretch", hide_index=True)
    st.bar_chart(tm.set_index("model")[["AP_test", "Recall", "F1", "Precision"]], stack=False)
    st.subheader("내부 성능 비교 구간 (Train 마지막 10%)")
    st.dataframe(load_csv("model_comparison.csv"), width="stretch", hide_index=True)

# ④ 임계값 분석 ----------------------------------------------------------
with tab_thr:
    r = st.radio("비용 비율 (미탐 1건 비용 : 오탐 1건 비용)", RATIOS, index=2, horizontal=True, format_func=lambda x: f"{x}:1")
    cmp = load_csv("test_threshold_comparison.csv")
    sub = cmp[cmp["r"] == r][["model", "policy", "threshold", "alerts", "FP", "FN", "Recall", "Precision", "total_cost", "cost_change_pct_vs_default"]]
    st.dataframe(sub.style.format({"threshold": "{:.4f}", "Recall": "{:.4f}", "Precision": "{:.4f}",
                                   "cost_change_pct_vs_default": "{:+.1f}%"}, na_rep="—"),
                 width="stretch", hide_index=True)
    st.bar_chart(sub.pivot(index="model", columns="policy", values="total_cost"), stack=False)
    st.caption("총비용 = FP + r × FN. 임계값은 검증 구간에서 모델·비용별로 선택해 원본 Test에 고정 적용했습니다.")

# 실험 결과 ---------------------------------------------------------------
with tab_exp:
    st.subheader("데이터 분할")
    st.dataframe(load_csv("split_summary.csv"), width="stretch", hide_index=True)
    st.subheader("LightGBM 가중치와 Platt 보정")
    st.dataframe(load_csv("weight_calibration.csv"), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.subheader("B. 특징군 단독")
    c1.dataframe(load_csv("feature_single.csv"), width="stretch", hide_index=True)
    c2.subheader("C. 특징군 제거")
    c2.dataframe(load_csv("feature_ablation.csv"), width="stretch", hide_index=True)
