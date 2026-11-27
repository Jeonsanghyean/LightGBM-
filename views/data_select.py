"""화면 1. 데이터 선택 — 출처별 요약과 제공 필드표."""
import os

import pandas as pd
import streamlit as st

from src.store import PersonalStore

US_PATH = "data/research/us/transactions.parquet"
KR_PATH = "data/research/kr/transactions.parquet"

# 설계서 표 3·4 기준. 파생 = 다른 열에서 계산, 합성 = 시뮬레이터가 만든 값
FIELDS = pd.DataFrame(
    [
        ("거래 날짜", "있음", "있음", "입력"),
        ("거래 시각", "있음 (초)", "없음 (1시간대만)", "입력 시 (분)"),
        ("금액·통화", "있음 (USD)", "있음 (KRW, 확인 필요)", "입력"),
        ("가맹점 이름", "있음 (접두어 제거)", "없음 (가명 키만)", "입력 시"),
        ("업종", "있음 (14종)", "있음 (코드)", "입력 시"),
        ("온라인·오프라인", "파생 (14종 중 6종)", "있음", "입력 시"),
        ("승인·취소", "없음", "있음 (코드 의미 확인 필요)", "입력 / 음수 금액"),
        ("가맹점 주소", "없음", "없음", "입력 시"),
        ("시도·지역", "없음", "있음 (시도 코드)", "주소에서 파생"),
        ("좌표", "합성 (실제 매장 위치 아님)", "없음", "없음 (외부 지오코딩 미사용)"),
        ("실제 이상 라벨", "있음 (is_fraud)", "있음 (여부·유형)", "없음"),
    ],
    columns=["정보", "미국 Sparkov", "한국 AI-Hub", "개인 거래"],
)


@st.cache_data
def us_summary():
    df = pd.read_parquet(US_PATH, columns=["transaction_date", "is_fraud", "card_id"])
    return {"rows": len(df), "start": df["transaction_date"].min(), "end": df["transaction_date"].max(),
            "fraud_rate": df["is_fraud"].mean(), "cards": df["card_id"].nunique()}


st.title("데이터 선택")
st.caption("세 데이터는 서로 다른 저장소에 따로 보관됩니다. 개인 거래는 이 컴퓨터에만 저장되고 모델 학습에 쓰이지 않습니다.")

labels = {"us": "미국 Sparkov (공개 합성)", "kr": "한국 AI-Hub (합성)", "personal": "개인 카드 거래 내역"}
current = st.session_state.get("source", "us")
choice = st.radio("분석할 데이터", list(labels), index=list(labels).index(current),
                  format_func=labels.get, horizontal=True)
st.session_state["source"] = choice

c = st.columns(4)
if choice == "us":
    if os.path.exists(US_PATH):
        s = us_summary()
        c[0].metric("거래", f"{s['rows']:,}")
        c[1].metric("카드", f"{s['cards']:,}")
        c[2].metric("기간", f"{s['start'][:7]} ~ {s['end'][:7]}")
        c[3].metric("실제 사기 비율", f"{s['fraud_rate']:.2%}")
        st.info("가맹점 좌표는 시뮬레이터가 카드 소유자 집 주변에 무작위로 만든 값입니다. 실제 매장 위치가 아닙니다.")
    else:
        st.warning("변환된 미국 데이터가 없습니다. `python -m src.sources.sparkov`를 먼저 실행하세요.")
elif choice == "kr":
    if os.path.exists(KR_PATH):
        st.success("한국 데이터가 준비되어 있습니다.")
    else:
        st.warning("한국 데이터 미확보")
        st.markdown(
            "1. AI-Hub에 로그인해 「이상 판별을 위한 금융거래 정보 및 사용자 패턴 합성데이터」를 신청합니다 (내국인만 신청 가능).\n"
            "2. 카드거래 CSV 16개와 데이터 설명서를 `data/research/kr/raw/`에 넣습니다.\n"
            "3. 변환 스크립트는 2주차에 추가됩니다.\n\n"
            "AI-Hub 원본은 재배포하지 않습니다. 보고서에는 집계 결과만 싣습니다."
        )
else:
    store = PersonalStore()
    c[0].metric("저장된 거래", f"{store.count():,}")
    c[1].metric("카드 별칭", f"{len(store.cards())}")
    st.caption(f"저장 위치: 프로젝트 폴더 밖 로컬 폴더 (`{store.folder}`). OneDrive 동기화 대상이 아닙니다.")
    st.page_link("views/personal.py", label="개인 거래 입력 화면으로 이동", icon="➡️")

st.subheader("제공 필드")
st.dataframe(FIELDS, hide_index=True, width="stretch")
st.caption("‘없음’인 정보는 임의로 생성하지 않고 화면에 ‘데이터에 없음’으로 표시합니다.")
