"""화면 2. 거래 내역 조회 — 검색·필터·정렬, 거래 상세 (설계서 F3)."""
import json

import pandas as pd
import streamlit as st

from views._common import (MISSING, PRECISION_KO, label_text, load, missing_notice, money, region_text, risk_text,
                           show, source_picker, time_text, type_text)

MAX_ROWS = 5000
st.session_state["_page"] = "lookup"
st.title("거래 내역 조회")
src = source_picker()
df = load(src)
if df is None or df.empty:
    missing_notice(src)
    st.stop()

# ---------------------------------------------------------------- 필터
dates = pd.to_datetime(df["transaction_date"])
dmin, dmax = dates.min().date(), dates.max().date()
c1, c2, c3 = st.columns([2, 2, 2])
default_start = max(dmin, (pd.Timestamp(dmax) - pd.Timedelta(days=30)).date()) if src != "personal" else dmin
period = c1.date_input("기간", value=(default_start, dmax), min_value=dmin, max_value=dmax, format="YYYY-MM-DD")
cards = sorted(df["card_id"].dropna().unique())
card = c2.selectbox("카드", [None] + cards, format_func=lambda x: "전체" if x is None else x)
query = c3.text_input("가맹점·거래 ID 검색")
c4, c5, c6, c7 = st.columns(4)
cats = sorted(df["merchant_category"].dropna().unique())
cat = c4.multiselect("업종", cats)
amax = float(df["amount"].max() or 0)
amt = c5.slider("금액 범위", 0.0, amax, (0.0, amax)) if amax > 0 else (0.0, 0.0)
lab = c6.selectbox("실제 라벨", ["전체", "이상(라벨)만", "정상(라벨)만"], disabled=df["is_fraud"].isna().all())
sort = c7.selectbox("정렬", ["최신순", "오래된순", "금액 큰 순", "금액 작은 순"])

start, end = period if isinstance(period, tuple) and len(period) == 2 else (dmin, dmax)
m = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)) & df["amount"].between(*amt)
if card:
    m &= df["card_id"] == card
if cat:
    m &= df["merchant_category"].isin(cat)
if lab == "이상(라벨)만":
    m &= df["is_fraud"] == 1
elif lab == "정상(라벨)만":
    m &= df["is_fraud"] == 0
if query:
    q = query.strip().lower()
    m &= (df["merchant_name"].astype("string").str.lower().str.contains(q, regex=False).fillna(False)
          | df["transaction_id"].astype("string").str.contains(q, regex=False).fillna(False))
view = df[m]
order = {"최신순": (["transaction_date", "transaction_time"], False), "오래된순": (["transaction_date", "transaction_time"], True),
         "금액 큰 순": (["amount"], False), "금액 작은 순": (["amount"], True)}[sort]
view = view.sort_values(order[0], ascending=order[1], na_position="last")
st.caption(f"조건에 맞는 거래 {len(view):,}건" + (f" · 앞 {MAX_ROWS:,}건만 표시" if len(view) > MAX_ROWS else ""))
view = view.head(MAX_ROWS).reset_index(drop=True)

table = pd.DataFrame({
    "날짜": view["transaction_date"],
    "시각": view.apply(time_text, axis=1),
    "금액": [money(a, c) for a, c in zip(view["amount"], view["currency"])],
    "가맹점": view["merchant_name"].map(show),
    "업종": view["merchant_category"].map(show),
    "지역": view.apply(region_text, axis=1),
    "유형": view["transaction_type"].map(type_text),
    "실제 라벨": view["is_fraud"].map(label_text),
    "위험도": view["risk_level"].map(lambda v: risk_text(None if pd.isna(v) else v, src)),
})
event = st.dataframe(table, hide_index=True, width="stretch", height=360, on_select="rerun",
                     selection_mode="single-row", key=f"tbl_{src}")
st.caption("행 왼쪽 칸을 누르거나 아래에 거래 ID를 넣으면 거래 상세가 표시됩니다. "
           "‘데이터에 없음’은 원본이 제공하지 않는 정보이며 임의로 채우지 않았습니다.")
detail_id = st.text_input("거래 ID로 상세 보기", placeholder="전체 거래 ID 입력", key=f"detail_{src}")

sel = event.selection.rows if event and event.selection else []
if sel:
    r = view.iloc[sel[0]]
elif detail_id.strip():
    hit = df[df["transaction_id"] == detail_id.strip()]  # 필터와 무관하게 전체에서 찾는다
    if hit.empty:
        st.warning("해당 거래 ID를 찾을 수 없습니다.")
        st.stop()
    r = hit.iloc[0]
else:
    st.stop()

# ---------------------------------------------------------------- 상세
st.divider()
st.subheader("거래 상세")
prov = json.loads(r["provenance"]) if isinstance(r["provenance"], str) else {}
derived = lambda f: " (파생)" if prov.get(f) == "derived" else (" (가정·미확인)" if prov.get(f) == "assumed" else "")
b, l, k = st.columns(3)
with b:
    st.markdown("**기본 정보**")
    st.markdown(
        f"- 거래 ID: `{r['transaction_id']}`\n"
        f"- 날짜: {r['transaction_date']}\n"
        f"- 시각: {time_text(r)}  · *{PRECISION_KO.get(r['time_precision'], '')}*\n"
        f"- 금액: {money(r['amount'], r['currency'])}{derived('currency')}\n"
        f"- 가맹점: {show(r['merchant_name'])}{derived('merchant_name')}\n"
        f"- 업종: {show(r['merchant_category'])}\n"
        f"- 온라인·오프라인: {type_text(r['transaction_type'])}{derived('transaction_type')}\n"
        f"- 승인·취소: {show({'approved': '승인', 'cancelled': '취소'}.get(r['transaction_status']))}\n"
        f"- 카드: `{r['card_id']}`"
    )
with l:
    st.markdown("**위치 정보**")
    lat, lon = r["latitude"], r["longitude"]
    coord = MISSING if pd.isna(lat) else f"{lat:.4f}, {lon:.4f}"
    dist = r.get("dist_prev_km")
    if pd.notna(dist):
        prev_txt = f"{dist:,.1f} km · {r['gap_prev_min']:,.0f}분 전"
        if pd.notna(r.get("speed_kmh")):
            prev_txt += f" · 최소 평균 속도 {r['speed_kmh']:,.0f} km/h"
    elif pd.isna(lat):
        prev_txt = "좌표 없음 — 거리 계산 불가"
    else:
        prev_txt = "이전 거래 없음 또는 시각 정밀도 부족"
    st.markdown(
        f"- 국가: {show(r['country'])}\n"
        f"- 시도·지역: {region_text(r)}{derived('region')}\n"
        f"- 가맹점 주소: {show(r['merchant_address'])}\n"
        f"- 좌표: {coord}\n"
        f"- 위치 정밀도: {PRECISION_KO.get(r['location_precision'], MISSING)}\n"
        f"- 이전 거래와의 거리: {prev_txt}"
    )
    if r["location_precision"] == "synthetic_point":
        st.caption("Sparkov 좌표는 시뮬레이터가 카드 소유자 집 주변에 무작위로 만든 값입니다.")
    if r.get("far_fast") == 1:
        st.warning("단시간 원거리 결제 조건에 해당합니다. 온라인 결제·해외 결제·가족카드 가능성도 함께 확인하세요.")
with k:
    st.markdown("**위험 정보**")
    st.markdown(f"- 실제 라벨: {label_text(r['is_fraud'])}"
                + (f" · 유형 {r['fraud_type']}" if pd.notna(r.get('fraud_type')) else ""))
    lvl = None if pd.isna(r.get("risk_level")) else r.get("risk_level")
    score_txt = "" if pd.isna(r.get("raw_score")) else f" (점수 {float(r['raw_score']):.4f})"
    st.markdown(f"- 모델 위험도: {risk_text(lvl, src)}{score_txt}")
    if src == "us":
        st.caption("미국은 원본 Test 구간만 v1 LightGBM 점수를 표시합니다 (화면 4와 같은 값·임계값).")
    feats = {"직전 시간대 결제 건수": r.get("count_prev_hour_bucket"), "같은 날 앞선 결제 건수": r.get("count_same_day_prior"),
             "24시간 내 같은 가맹점 결제": r.get("same_merchant_24h"), "새로운 지역 정도": r.get("region_novelty")}
    st.dataframe(pd.DataFrame({"특징": list(feats), "값": [show(v if pd.isna(v) else round(float(v), 3)) for v in feats.values()]}),
                 hide_index=True, width="stretch")
