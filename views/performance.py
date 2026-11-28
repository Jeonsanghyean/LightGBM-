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


def by_country(df, country):
    """[v2 변경 2026-11-28] 한국 결과가 생기면 e4·e5·비용 파일에 KR 행이 함께 들어가므로 국가별로 나눠 보여준다."""
    return None if df is None else df[df["country"] == country].dropna(axis=1, how="all")


NOT_MEASURED_KR = "미측정 — 한국 AI-Hub 데이터 미확보(B안). 데이터를 받으면 같은 화면에 표시됩니다."
fmt4 = lambda cols: {c: "{:.4f}" for c in cols}


def dash(col: pd.Series, fmt: str = "{:.4f}") -> pd.Series:
    """빈 값을 '—'로 보여주기 위해 미리 문자열로 바꾼다. st.dataframe은 Styler의 na_rep를 무시하고 None으로 표시한다."""
    return col.map(lambda v: "—" if pd.isna(v) else fmt.format(v))


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
e2 = read("e2_kr_performance.csv")
if e2 is None:
    st.info(NOT_MEASURED_KR)
else:
    t = e2[["setting", "model", "n_features", "AP_valid", "AP_test", "baseline_test",
            "Precision_test@0.5", "Recall_test@0.5", "F1_test@0.5"]].copy()
    t.columns = ["특징 구성", "모델", "특징 수", "AP(검증)", "AP(테스트 2024)", "기준선(테스트)",
                 "Precision@0.5", "Recall@0.5", "F1@0.5"]
    st.dataframe(t.style.format(fmt4(t.columns[3:])), hide_index=True, width="stretch")
    st.bar_chart(e2.pivot(index="model", columns="setting", values="AP_test"), stack=False)
    st.caption("한국은 분기 단위로 나눴습니다: 2021Q1~2023Q2 학습 / 2023Q3~Q4 검증 / 2024 테스트. "
               "내부 비교 구간이 없어 테스트 구간 값을 봅니다. 시각이 1시간대 단위라 5·15·60분 건수는 쓰지 않았습니다.")

st.subheader("국가 간 일반화 (E3)")
e3 = read("e3_cross_country.csv")
if e3 is None:
    st.info(NOT_MEASURED_KR)
else:
    ap = e3.drop_duplicates(["train_country", "eval_country"])[["train_country", "eval_country", "AP_test", "baseline_test"]]
    ap.columns = ["학습 국가", "평가 국가", "AP(평가 국가 테스트)", "기준선"]
    st.dataframe(ap.style.format(fmt4(ap.columns[2:])), hide_index=True, width="stretch")
    r3 = st.radio("비용 비율 (미탐:오탐) ", sorted(e3["r"].unique()), horizontal=True, format_func=lambda x: f"{x}:1", key="e3_r")
    c = e3[e3["r"] == r3][["train_country", "eval_country", "policy", "threshold", "Precision", "Recall", "total_cost"]].copy()
    c.columns = ["학습 국가", "평가 국가", "임계값 정책", "임계값", "Precision", "Recall", "총비용"]
    st.dataframe(c.style.format(fmt4(["임계값", "Precision", "Recall"])), hide_index=True, width="stretch")
    st.caption("공통 특징 5개(금액 비율, 직전 시간대·같은 날 건수, 시간대 희소성, 이력 부족)로 학습했습니다. "
               "같은 국가 조합(US→US, KR→KR)은 비교 기준입니다. 두 나라 라벨의 의미가 같은지는 확인되지 않았으므로 해석에 주의하세요.")

# ---------------------------------------------------------------- 위치 특징
st.subheader("위치 특징 추가 효과 (E4)")
loc_all = read("e4_location_effect.csv")
loc = by_country(loc_all, "US")
if loc is not None:
    st.markdown("**미국**")
    t = loc[["setting", "with_location", "AP_internal_test", "AP_test", "AP_gain_vs_without_D"]].copy()
    t.columns = ["특징 구성", "위치 특징 포함", "AP(내부 비교)", "AP(원본 Test)", "위치 추가로 늘어난 AP(내부)"]
    t["위치 추가로 늘어난 AP(내부)"] = dash(t["위치 추가로 늘어난 AP(내부)"])
    st.dataframe(t.style.format(fmt4(t.columns[2:4])), hide_index=True, width="stretch")
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
st.markdown("**한국** — 좌표가 없어 지역 단위 특징(처음 보는 지역, 지역 변경, 해외 여부)을 씁니다.")
loc_kr = by_country(loc_all, "KR")
if loc_kr is None or loc_kr.empty:
    st.info(NOT_MEASURED_KR)
else:
    t = loc_kr[["setting", "with_location", "AP_valid", "AP_test", "AP_gain_vs_without_D_test"]].copy()
    t.columns = ["특징 구성", "지역 특징 포함", "AP(검증)", "AP(테스트)", "지역 추가로 늘어난 AP(테스트)"]
    t["지역 추가로 늘어난 AP(테스트)"] = dash(t["지역 추가로 늘어난 AP(테스트)"])
    st.dataframe(t.style.format(fmt4(t.columns[2:4])), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 특징군 제거
st.subheader("특징군 기여도 — 하나씩 제거 (E5)")
abl_all = read("e5_group_ablation.csv")
abl = by_country(abl_all, "US")
if abl is not None:
    st.markdown("**미국**")
    t = abl[["setting", "AP_internal_test", "AP_drop_internal_test", "AP_test"]].copy()
    t.columns = ["설정", "AP(내부 비교)", "AP 하락(내부)", "AP(원본 Test)"]
    st.dataframe(t.style.format({c: "{:.4f}" for c in t.columns[1:]}), hide_index=True, width="stretch")
    st.bar_chart(abl[abl["removed_group"] != "-"].set_index("removed_group")["AP_drop_internal_test"], horizontal=True)
st.markdown("**한국**")
abl_kr = by_country(abl_all, "KR")
if abl_kr is None or abl_kr.empty:
    st.info(NOT_MEASURED_KR)
else:
    t = abl_kr[["setting", "AP_valid", "AP_test", "AP_drop_test"]].copy()
    t.columns = ["설정", "AP(검증)", "AP(테스트)", "AP 하락(테스트)"]
    st.dataframe(t.style.format(fmt4(t.columns[1:])), hide_index=True, width="stretch")
    st.bar_chart(abl_kr[abl_kr["removed_group"] != "-"].set_index("removed_group")["AP_drop_test"], horizontal=True)

st.subheader("한국 이상거래 유형별 Recall (E6)")
e6 = read("e6_kr_type_recall.csv")
if e6 is None:
    st.info(NOT_MEASURED_KR if e2 is None else "미측정 — 이상거래 유형 정보가 없습니다.")
else:
    ratios = [f"{r}:1" for r in sorted(e6["cost_ratio"].str.split(":").str[0].astype(int).unique())]
    t = e6.pivot(index="fraud_type", columns="cost_ratio", values="Recall")[ratios]
    t.insert(0, "테스트 이상 건수", e6.drop_duplicates("fraud_type").set_index("fraud_type")["n_test"])
    t = t.rename(columns={r: f"Recall {r}" for r in ratios}).rename_axis("이상 유형").reset_index()
    st.dataframe(t.style.format({f"Recall {r}": "{:.3f}" for r in ratios}), hide_index=True, width="stretch")
    st.caption("확장 전체 LightGBM, 비용 비율마다 검증 구간에서 고른 임계값을 테스트 구간(2024)에 적용한 Recall입니다. 건수가 적은 유형은 값이 크게 흔들립니다.")

# ---------------------------------------------------------------- 비용 임계값
st.subheader("비용 임계값 최적화")
thr = read("cost_thresholds_v2.csv")
if thr is not None:
    countries = thr["country"].unique().tolist()
    if len(countries) > 1:
        thr = thr[thr["country"] == st.radio("국가", countries, horizontal=True, key="thr_country")]
    c1, c2, c3 = st.columns(3)
    setting = c1.selectbox("특징 구성", thr["setting"].unique().tolist())
    model = c2.selectbox("모델", thr.loc[thr["setting"] == setting, "model"].unique().tolist())
    r = c3.radio("비용 비율 (미탐:오탐)", sorted(thr["r"].unique()), horizontal=True, format_func=lambda x: f"{x}:1")
    sub = thr[(thr["setting"] == setting) & (thr["model"] == model) & (thr["r"] == r)]
    t = sub[["policy", "threshold", "alerts", "FP", "FN", "Precision", "Recall", "total_cost", "cost_change_pct_vs_default"]].copy()
    t.columns = ["정책", "임계값", "경고 수", "오탐(FP)", "미탐(FN)", "Precision", "Recall", "총비용", "기본 대비 변화"]
    t["기본 대비 변화"] = dash(t["기본 대비 변화"], "{:+.1f}%")
    st.dataframe(t.style.format(fmt4(["임계값", "Precision", "Recall"])), hide_index=True, width="stretch")
    st.caption("총비용 = FP + r × FN (오탐 1건 = 1). 임계값은 검증 구간에서 고르고 테스트 구간(미국은 원본 Test)에 그대로 적용했습니다.")
    wins = thr[thr["policy"] == "검증 임계값"]
    st.write(f"전체 {len(wins)}개 조합 중 {int((wins['cost_change_pct_vs_default'] < 0).sum())}개에서 기본 0.5보다 총비용이 낮았습니다.")
