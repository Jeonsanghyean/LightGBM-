"""화면 5. 모델 성능 비교 — 국가별 성능, 위치 특징 효과, 특징군 제거, 비용 임계값 (설계서 표 7).

outputs/v2/*.csv만 읽는다. 측정하지 않은 항목(한국)은 '미측정'으로 표시하고 수치를 만들지 않는다.
"""
import os

import pandas as pd
import streamlit as st

OUT = "outputs/v2"
st.title("모델 성능 비교")


def read(name):
    p = f"{OUT}/{name}"
    return pd.read_csv(p) if os.path.exists(p) else None


status = read("experiment_status.csv")
perf = read("e1_country_performance.csv")
if perf is None:
    st.warning("국가별 실험 결과가 없습니다. `python -m src.experiments_country`를 먼저 실행하세요.")
    st.stop()

st.subheader("실험 상태")
st.dataframe(status.rename(columns={"experiment": "실험", "country": "국가", "description": "내용", "status": "상태"}),
             hide_index=True, width="stretch")
st.caption("AP(평균 정밀도)는 위험 순위의 품질입니다. 무작위로 순위를 매겼을 때의 기대 AP는 해당 구간의 이상 비율(기준선)과 같으므로 항상 함께 봅니다.")

# ---------------------------------------------------------------- 국가별 성능
st.subheader("미국 모델 성능 (E1)")
show = perf[["setting", "model", "n_features", "AP_internal_test", "baseline_internal_test", "AP_test", "baseline_test",
             "Precision_internal_test@0.5", "Recall_internal_test@0.5", "F1_internal_test@0.5"]].copy()
show.columns = ["특징 구성", "모델", "특징 수", "AP(내부 비교)", "기준선(내부)", "AP(원본 Test)", "기준선(Test)",
                "Precision@0.5", "Recall@0.5", "F1@0.5"]
st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[3:]}), hide_index=True, width="stretch")
st.bar_chart(perf.pivot(index="model", columns="setting", values="AP_test"), stack=False)
st.caption("원본 Test 구간 AP. 학습·임계값 선택에 쓰지 않은 구간입니다.")

st.subheader("한국 모델 성능 (E2)")
kr = status[status["country"] == "KR"] if status is not None else pd.DataFrame()
if len(kr) and kr["status"].str.startswith("미측정").all():
    st.info("미측정 — 한국 AI-Hub 데이터 미확보(B안). 데이터를 받으면 같은 화면에 표시됩니다.")

# ---------------------------------------------------------------- 위치 특징
st.subheader("위치 특징 추가 효과 (E4)")
loc = read("e4_location_effect.csv")
if loc is not None:
    t = loc[["setting", "with_location", "AP_internal_test", "AP_test", "AP_gain_vs_without_D"]].copy()
    t.columns = ["특징 구성", "위치 특징 포함", "AP(내부 비교)", "AP(원본 Test)", "위치 추가로 늘어난 AP(내부)"]
    st.dataframe(t.style.format({c: "{:.4f}" for c in t.columns[2:]}, na_rep="—"), hide_index=True, width="stretch")
    st.warning("Sparkov의 가맹점 좌표는 시뮬레이터가 카드 소유자 집 주변에 무작위로 만든 값입니다. "
               "이 결과는 '시뮬레이터 조건의 효과'이며 실제 위치 기반 탐지 효과로 일반화하지 않습니다.")
    dec = read("e4b_location_decomposition.csv")
    if dec is not None:
        st.markdown("**위치 특징 효과 분해 (E4b)** — 위치 특징군에는 직전 거래와의 시간 간격도 들어 있어 따로 나눠 봤습니다.")
        t = dec[["setting", "added", "AP_internal_test", "AP_gain_vs_v1_internal_test", "AP_test"]].copy()
        t.columns = ["특징 구성", "추가한 특징", "AP(내부 비교)", "v1 대비(내부)", "AP(원본 Test)"]
        st.dataframe(t.style.format({"AP(내부 비교)": "{:.4f}", "v1 대비(내부)": "{:+.4f}", "AP(원본 Test)": "{:.4f}"}),
                     hide_index=True, width="stretch")
        st.caption("거리만 추가하면 AP가 오르지 않았습니다. 늘어난 AP는 거래 사이 시간 간격(속도 계산에 포함)에서 온 것으로 보입니다.")

# ---------------------------------------------------------------- 특징군 제거
st.subheader("특징군 기여도 — 하나씩 제거 (E5)")
abl = read("e5_group_ablation.csv")
if abl is not None:
    t = abl[["setting", "AP_internal_test", "AP_drop_internal_test", "AP_test"]].copy()
    t.columns = ["설정", "AP(내부 비교)", "AP 하락(내부)", "AP(원본 Test)"]
    st.dataframe(t.style.format({c: "{:.4f}" for c in t.columns[1:]}), hide_index=True, width="stretch")
    st.bar_chart(abl[abl["removed_group"] != "-"].set_index("removed_group")["AP_drop_internal_test"], horizontal=True)

# ---------------------------------------------------------------- 비용 임계값
st.subheader("비용 임계값 최적화")
thr = read("cost_thresholds_v2.csv")
if thr is not None:
    c1, c2, c3 = st.columns(3)
    setting = c1.selectbox("특징 구성", thr["setting"].unique().tolist())
    model = c2.selectbox("모델", thr.loc[thr["setting"] == setting, "model"].unique().tolist())
    r = c3.radio("비용 비율 (미탐:오탐)", sorted(thr["r"].unique()), horizontal=True, format_func=lambda x: f"{x}:1")
    sub = thr[(thr["setting"] == setting) & (thr["model"] == model) & (thr["r"] == r)]
    t = sub[["policy", "threshold", "alerts", "FP", "FN", "Precision", "Recall", "total_cost", "cost_change_pct_vs_default"]].copy()
    t.columns = ["정책", "임계값", "경고 수", "오탐(FP)", "미탐(FN)", "Precision", "Recall", "총비용", "기본 대비 변화"]
    st.dataframe(t.style.format({"임계값": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                                 "기본 대비 변화": "{:+.1f}%"}, na_rep="—"), hide_index=True, width="stretch")
    st.caption("총비용 = FP + r × FN (오탐 1건 = 1). 임계값은 검증 구간에서 고르고 원본 Test에 그대로 적용했습니다.")
    wins = thr[thr["policy"] == "검증 임계값"]
    st.write(f"전체 {len(wins)}개 조합 중 {int((wins['cost_change_pct_vs_default'] < 0).sum())}개에서 기본 0.5보다 총비용이 낮았습니다.")
