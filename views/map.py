"""화면 3. 결제 위치 지도 — 마커·팝업·기간 필터·연속 거래 거리, 위치 누락 안내 (설계서 F4).

- 미국: 좌표가 있으므로 지도에 표시. 단, 시뮬레이터가 만든 합성 좌표다.
- 한국: 시도 코드만 있으므로 지도에 점을 찍지 않고 지역 단위 막대그래프로 표시.
- 개인: 외부 지도·지오코딩을 쓰지 않으므로(설계서 6.3 ①) 주소에서 뽑은 시도 단위 표만 표시.
"""
import pandas as pd
import pydeck as pdk
import streamlit as st

from views._common import PRECISION_KO, load, missing_notice, money, show, source_picker, time_text

st.session_state["_page"] = "map"
st.title("결제 위치 지도")
src = source_picker()
df = load(src)
if df is None or df.empty:
    missing_notice(src)
    st.stop()

n_xy = int(df["latitude"].notna().sum())
prec = df["location_precision"].value_counts()
st.caption("위치 정밀도: " + " · ".join(f"{PRECISION_KO.get(k, k)} {v:,}건" for k, v in prec.items()))
if n_xy < len(df):
    st.info(f"좌표가 없는 거래 {len(df) - n_xy:,}건은 지도에 표시하지 않습니다. 가맹점 이름만으로 좌표를 추정하지 않습니다.")

# ---------------------------------------------------------------- 좌표 없는 출처: 지역 단위
if n_xy == 0:
    key = "region" if df["region"].notna().any() else ("region_code" if "region_code" in df.columns else None)
    if key is None or df[key].isna().all():
        st.warning("이 데이터에는 지도에 표시할 위치 정보(좌표·지역)가 없습니다.")
        st.stop()
    grp = df.dropna(subset=[key]).groupby(key).agg(거래=("transaction_id", "size"), 이상라벨=("is_fraud", "sum"))
    if df["is_fraud"].notna().any():
        grp["이상 비율"] = grp["이상라벨"] / grp["거래"]
    else:
        grp = grp.drop(columns="이상라벨")
    st.subheader("지역별 거래" + (" (시도 코드 — 의미 미확인)" if key == "region_code" else ""))
    st.bar_chart(grp["거래"])
    st.dataframe(grp.reset_index(), hide_index=True, width="stretch")
    if src == "personal":
        st.caption("개인 거래는 주소를 좌표로 바꾸지 않고(외부 전송 없음), 주소의 시도명만 사용합니다.")
    st.stop()

# ---------------------------------------------------------------- 좌표 있는 출처: 카드별 지도
cards = df.loc[df["latitude"].notna(), "card_id"].value_counts()
c1, c2 = st.columns([2, 3])
card = c1.selectbox("카드 (거래 많은 순)", cards.index.tolist(), format_func=lambda c: f"{c} ({cards[c]:,}건)")
d = df[(df["card_id"] == card) & df["latitude"].notna()].copy()
d["_dt"] = pd.to_datetime(d["transaction_date"])
dmin, dmax = d["_dt"].min().date(), d["_dt"].max().date()
start_default = max(dmin, (pd.Timestamp(dmax) - pd.Timedelta(days=14)).date())
period = c2.date_input("기간", value=(start_default, dmax), min_value=dmin, max_value=dmax, format="YYYY-MM-DD")
start, end = period if isinstance(period, tuple) and len(period) == 2 else (dmin, dmax)
d = d[(d["_dt"] >= pd.Timestamp(start)) & (d["_dt"] <= pd.Timestamp(end))]
d = d.sort_values(["transaction_date", "transaction_time"]).reset_index(drop=True)
if d.empty:
    st.write("선택한 기간에 거래가 없습니다.")
    st.stop()

fraud = d["is_fraud"].fillna(0).astype(int)
d["color"] = [[226, 75, 74, 220] if f else [55, 138, 221, 170] for f in fraud]
times = d.apply(time_text, axis=1)  # time_text는 열 이름으로 접근하므로 Series 행을 넘긴다
d["tip"] = [f"{r.transaction_date} {tt}<br>{money(r.amount, r.currency)}<br>{show(r.merchant_name)}"
            f"<br>{show(r.merchant_category)}<br>라벨: {'이상' if f else '정상'}"
            for r, tt, f in zip(d.itertuples(), times, fraud)]
path = [{"path": d[["longitude", "latitude"]].values.tolist()}]
deck = pdk.Deck(
    layers=[
        pdk.Layer("PathLayer", path, get_path="path", get_color=[136, 135, 128, 120], width_min_pixels=1),
        pdk.Layer("ScatterplotLayer", d, get_position=["longitude", "latitude"], get_fill_color="color",
                  get_radius=600, radius_min_pixels=4, pickable=True),
    ],
    initial_view_state=pdk.ViewState(latitude=float(d["latitude"].mean()), longitude=float(d["longitude"].mean()), zoom=7),
    tooltip={"html": "{tip}"},
)
st.pydeck_chart(deck, height=420)
st.caption(f"{len(d):,}건 · 빨강 = 실제 이상 라벨, 파랑 = 정상 라벨, 회색 선 = 시간 순서. "
           "Sparkov 좌표는 카드 소유자 집 주변에 무작위로 생성된 합성값이므로 실제 이동 경로가 아닙니다. "
           "지도 배경은 외부 타일 서버에서 불러옵니다(공개 합성 데이터에만 사용).")

# ---------------------------------------------------------------- 연속 거래 거리
st.subheader("연속 거래 간 거리")
moves = pd.DataFrame({
    "날짜": d["transaction_date"], "시각": d.apply(time_text, axis=1),
    "금액": [money(a, c) for a, c in zip(d["amount"], d["currency"])],
    "가맹점": d["merchant_name"].map(show),
    "직전 거래와 거리(km)": d["dist_prev_km"].round(1),
    "경과(분)": d["gap_prev_min"].round(0),
    "최소 평균 속도(km/h)": d["speed_kmh"].round(0),
    "단시간 원거리": d["far_fast"].map({1.0: "예", 0.0: "아니오"}),
    "실제 라벨": fraud.map({1: "이상", 0: "정상"}),
})
m1, m2, m3 = st.columns(3)
m1.metric("평균 이동 거리", f"{d['dist_prev_km'].mean():,.1f} km")
m2.metric("단시간 원거리 결제", f"{int((d['far_fast'] == 1).sum())}건")
m3.metric("이상 라벨", f"{int(fraud.sum())}건")
st.dataframe(moves, hide_index=True, width="stretch", height=300)
st.caption("거리·속도는 직전 거래(같은 카드, 더 이른 시각)와 비교합니다. 온라인 결제는 속도를 계산하지 않습니다. "
           "속도가 비정상이어도 온라인·해외·가족카드 등 다른 설명이 가능하므로 사기로 단정하지 않습니다.")
