"""화면 4. 이상거래 분석 — 거래별 위험 점수, 주요 위험 요인, 정상·의심 거래 비교, 시간대 분포 (설계서 표 7).

- 미국: 원본 Test 구간(555,719건)에 대한 v1 LightGBM 점수(outputs/test_scores.parquet)를 사용한다.
  점수는 학습에 쓰지 않은 구간이므로 실제 라벨과 나란히 비교할 수 있다.
- 개인: src.risk.score로 그 자리에서 계산한다(라벨 없음).
- 한국: 데이터 미확보 → 안내만.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import risk
from src.features import ALL_FEATURES as V1
from src.store import PersonalStore
from views._common import missing_notice, source_picker

st.session_state["_page"] = "analysis"
st.title("이상거래 분석")
st.caption(risk.NOTICE)
src = source_picker()
# 미국: v1 모델 임계값 / 개인: 시간대 특징을 뺀 개인 점수용 모델 임계값 (risk.py 설계 변경 참고)
t_high, t_watch = risk.thresholds() if src == "us" else risk.personal_thresholds()


@st.cache_data(show_spinner="점수 불러오는 중…")
def us_scores() -> pd.DataFrame:
    d = pd.read_parquet("outputs/test_scores.parquet")
    d["score"] = d["score_LightGBM"]
    d["risk_level"] = np.select([d["score"] >= t_high, d["score"] >= t_watch], ["높음", "주의"], "낮음")
    d["hour"] = pd.to_datetime(d["trans_date_trans_time"]).dt.hour
    return d


@st.cache_resource
def lgb_model():
    return joblib.load(risk.MODEL_PATH)


if src == "kr":
    missing_notice("kr")
    st.stop()

if src == "us":
    d = us_scores()
    st.caption("미국 원본 Test 구간 전체의 LightGBM(v1 6개 특징, 양성 가중치 10) 점수입니다. 이 구간은 학습·임계값 선택에 쓰지 않았습니다.")
    labeled = True
else:
    rows = PersonalStore().query()
    if rows.empty:
        missing_notice("personal")
        st.stop()
    res, _ = risk.score(rows)
    d = res.rename(columns={"raw_score": "score"})
    d["hour"] = d["hour_bucket"]
    d["is_fraud"] = np.nan
    labeled = False

# ---------------------------------------------------------------- 위험 수준 요약
st.subheader("위험 수준별 거래")
levels = ["높음", "주의", "낮음", "판단 보류"]
summary = d.groupby("risk_level").agg(거래=("score", "size"))
if labeled:
    summary["실제 이상 라벨"] = d.groupby("risk_level")["is_fraud"].sum()
    summary["라벨 기준 이상 비율"] = (summary["실제 이상 라벨"] / summary["거래"]).map("{:.1%}".format)
summary = summary.reindex([l for l in levels if l in summary.index])
st.dataframe(summary.reset_index().rename(columns={"risk_level": "위험 수준"}), hide_index=True, width="stretch")
st.caption(f"높음: 점수 ≥ {t_high:.3f} (5:1 비용 임계값) · 주의: 점수 ≥ {t_watch:.3f} (100:1 비용 임계값) · 낮음: 그 아래")

# ---------------------------------------------------------------- 거래별 점수와 주요 요인
st.subheader("점수가 높은 거래와 주요 위험 요인")
top = d.dropna(subset=["score"]).sort_values("score", ascending=False).head(200).reset_index(drop=True)
id_col = "trans_num" if src == "us" else "transaction_id"
show = pd.DataFrame({"거래 ID": top[id_col], "점수": top["score"].round(4), "위험 수준": top["risk_level"],
                     "금액": top["amt"] if src == "us" else top["amount"]})
if labeled:
    show["실제 라벨"] = top["is_fraud"].map({1: "이상", 0: "정상"})
st.dataframe(show, hide_index=True, width="stretch", height=260)
pick = st.selectbox("요인을 볼 거래", top[id_col].tolist()) if len(top) else None
if pick is not None:
    r = top[top[id_col] == pick].iloc[0]
    if src == "us":
        X = pd.DataFrame([r[V1].astype(float).to_numpy()], columns=V1)
        contrib = lgb_model().booster_.predict(X, pred_contrib=True)[0, :len(V1)]
        feats = V1
    else:
        feats = risk.PERSONAL_FEATURES
        X = pd.DataFrame([r[feats].astype(float).to_numpy()], columns=feats)
        contrib = joblib.load(risk.PERSONAL_MODEL_PATH).booster_.predict(X, pred_contrib=True)[0, :len(feats)]
        if r["rules"]:
            st.info("규칙 기반 확인 항목: " + " / ".join(r["rules"]))
    factors = pd.DataFrame({"특징": [risk.FEATURE_KO[f] for f in feats], "기여도(로그 오즈)": contrib,
                            "값": [round(float(r[f]), 3) for f in feats]}).sort_values("기여도(로그 오즈)", ascending=False)
    c1, c2 = st.columns([3, 2])
    c1.bar_chart(factors.set_index("특징")["기여도(로그 오즈)"], horizontal=True)
    c2.dataframe(factors, hide_index=True, width="stretch")
    st.caption("기여도는 LightGBM의 거래별 기여값(SHAP 방식)입니다. 양수는 점수를 올린 요인, 음수는 낮춘 요인입니다.")

# ---------------------------------------------------------------- 정상·의심 비교
st.subheader("정상 거래와 의심 거래 비교")
d["그룹"] = np.where(d["risk_level"].isin(["높음", "주의"]), "의심(높음·주의)",
                    np.where(d["risk_level"].eq("낮음"), "낮음", "판단 보류"))
comp = d[d["그룹"] != "판단 보류"].groupby("그룹")[V1 if src == "us" else risk.PERSONAL_FEATURES].median().T
comp.index = [risk.FEATURE_KO[f] for f in comp.index]
st.dataframe(comp.round(3), width="stretch")
st.caption("각 특징의 중앙값입니다." + (" 미국 데이터는 실제 라벨 기준 비교도 아래 시간대 그래프에 함께 표시합니다." if labeled else ""))

st.subheader("시간대별 거래 분포")
hours = d.dropna(subset=["hour"]).copy()
hours["hour"] = hours["hour"].astype(int)
dist = hours.groupby(["hour", "그룹"]).size().unstack(fill_value=0)
dist = (dist / dist.sum()).reindex(range(24), fill_value=0)
if labeled:
    lab = hours.groupby(["hour", "is_fraud"]).size().unstack(fill_value=0)
    lab = (lab / lab.sum()).reindex(range(24), fill_value=0).rename(columns={0: "정상(라벨)", 1: "이상(라벨)"})
    dist = dist.join(lab)
st.line_chart(dist)
st.caption("그룹마다 전체를 1로 둔 시간대별 비율입니다.")
