"""미국·한국·개인 카드 거래 이상거래 탐지 — 확장 대시보드 (v2).

실행: streamlit run app_v2.py
기존 연구 대시보드는 app.py(v1)로 그대로 실행된다.
[설계 변경 2026-11-27] 화면 파일을 pages/가 아닌 views/에 둠.
  이유: pages/ 폴더는 Streamlit이 같은 폴더의 app.py(v1)에도 자동으로 붙여 v1 화면이 바뀌기 때문.
화면 2~5는 2~3주차에 추가한다.
"""
import streamlit as st

st.set_page_config(page_title="카드 이상거래 탐지 v2", layout="wide")

nav = st.navigation([
    st.Page("views/data_select.py", title="1. 데이터 선택", default=True),
    st.Page("views/lookup.py", title="2. 거래 내역 조회"),
    st.Page("views/map.py", title="3. 결제 위치 지도"),
    st.Page("views/analysis.py", title="4. 이상거래 분석"),
    st.Page("views/performance.py", title="5. 모델 성능 비교"),
    st.Page("views/personal.py", title="6. 개인 거래 입력"),
])
st.sidebar.caption("위험도는 사기 확정이 아니라 추가 확인이 필요하다는 뜻입니다.")
nav.run()
