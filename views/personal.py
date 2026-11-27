"""화면 6. 개인 거래 입력 — 직접 입력, CSV·Excel 업로드, 편집·삭제·내보내기 (설계서 F5~F7, F9).

개인정보 원칙
- 카드번호·CVC·비밀번호·인증정보는 받지 않는다. 업로드 파일에서 발견되면 해당 열을 저장 전에 제거한다.
- 업로드 파일은 메모리에서만 읽고 디스크에 따로 저장하지 않는다.
- 저장소는 프로젝트(OneDrive) 밖 로컬 폴더이고, 외부로 전송하지 않는다.
"""
import hashlib
import io
import datetime as dt

import pandas as pd
import streamlit as st

from src import privacy
from src.schema import validate
from src.sources import personal
from src.store import PersonalStore

store = PersonalStore()
VIEW_COLS = {"transaction_date": "날짜", "transaction_time": "시각", "amount": "금액", "currency": "통화",
             "merchant_name": "가맹점", "merchant_address": "주소", "region": "시도(파생)",
             "transaction_type": "유형", "transaction_status": "상태", "card_id": "카드 별칭"}
TARGET_KO = {"datetime": "날짜+시각(한 열)", "date": "날짜", "time": "시각", "amount": "금액", "currency": "통화",
             "merchant_name": "가맹점 이름", "merchant_address": "가맹점 주소", "transaction_type": "거래 유형",
             "transaction_status": "승인·취소", "card_alias": "카드 별칭"}
TYPE_KO = {None: "선택 안 함", "online": "온라인", "offline": "오프라인"}



def flash(msg: str) -> None:
    """저장·삭제 후 화면을 새로 그려도 결과 메시지가 남도록 세션에 보관하고 다시 실행한다."""
    st.session_state["flash"] = msg
    st.rerun()


st.title("개인 거래 입력")
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
st.info(
    "카드번호 전체, CVC, 비밀번호, 인증번호는 입력하지 마세요. 카드는 ‘생활카드’처럼 별칭으로 구분합니다.\n\n"
    f"입력한 거래는 이 컴퓨터의 로컬 폴더에만 저장되며 외부로 전송되지 않고 모델 학습에도 쓰이지 않습니다. "
    f"현재 {store.count():,}건 저장됨."
)

# st.tabs는 저장·삭제 후 다시 그릴 때 첫 탭으로 돌아가므로, 선택값이 세션에 유지되는 메뉴를 쓴다.
SECTIONS = ["직접 입력", "CSV·Excel 업로드", "내역 편집·내보내기", "분석 실행"]
section = st.segmented_control("메뉴", SECTIONS, default=SECTIONS[0], key="personal_section",
                               label_visibility="collapsed") or SECTIONS[0]

# ---------------------------------------------------------------- 직접 입력
if section == "직접 입력":
    with st.form("manual", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        date = c1.date_input("결제 날짜 *", value=dt.date.today(), format="YYYY-MM-DD")
        use_time = c2.checkbox("결제 시간 입력", value=True, help="시간이 없으면 시간대·반복 결제 분석을 할 수 없습니다.")
        tval = c2.time_input("결제 시간", value=dt.time(12, 0), step=60)
        amount = c3.number_input("결제 금액 *", min_value=0.0, step=100.0, format="%.2f")
        currency = c4.selectbox("통화 *", personal.CURRENCIES)
        c5, c6, c7, c8 = st.columns(4)
        merchant = c5.text_input("가맹점 이름")
        address = c6.text_input("가맹점 주소 (선택)", help="시도명만 추출해 지역 분석에 씁니다. 좌표로 변환하지 않습니다.")
        ttype = c7.selectbox("거래 유형 (선택)", list(TYPE_KO), format_func=TYPE_KO.get)
        alias = c8.text_input("카드 별칭 (선택)", value="내 카드")
        if st.form_submit_button("저장", type="primary"):
            if amount <= 0:
                st.error("결제 금액을 입력하세요.")
            else:
                row, err = personal.manual_entry(date, tval.strftime("%H:%M") if use_time else None, amount, currency,
                                                 merchant or None, address or None,
                                                 {"online": "온라인", "offline": "오프라인"}.get(ttype), alias or "내 카드")
                scan = privacy.scan(row[["merchant_name", "merchant_address", "card_id"]])
                if not scan.clean:
                    st.error("카드번호나 주민등록번호로 보이는 값이 있어 저장하지 않았습니다. 해당 칸을 지우고 다시 입력하세요.")
                elif len(err):
                    st.error(err.iloc[0]["reason"])
                else:
                    if personal.find_duplicates(row, store.query()).iloc[0]:
                        st.warning("같은 날짜·시각·금액·가맹점의 거래가 이미 있습니다. 중복이면 ‘내역 편집’에서 삭제하세요.")
                    store.insert(row)
                    flash("거래 1건을 저장했습니다.")

# ---------------------------------------------------------------- 업로드
if section == "CSV·Excel 업로드":
    up = st.file_uploader("카드 이용내역 파일 (CSV, XLSX)", type=["csv", "xlsx", "xls"])
    st.caption("카드사 홈페이지에서 내려받은 이용내역을 그대로 올려도 됩니다. 열 이름은 자동으로 추정하고, 아래에서 고칠 수 있습니다.")
    if up is not None:
        data = up.getvalue()
        fid = hashlib.sha256(data).hexdigest()
        if st.session_state.get("up_id") != fid:  # 새 파일이면 단계 상태 초기화
            st.session_state.update(up_id=fid, up_fixed=None)
        try:
            raw = personal.read_any(data, up.name)
        except Exception:
            st.error("파일을 읽을 수 없습니다. CSV(UTF-8/CP949) 또는 Excel 형식인지 확인하세요.")
            st.stop()

        # 1) 민감정보 검사
        res = privacy.scan(raw)
        if res.findings:
            for f in res.findings:
                st.warning(f)
            st.caption("제거한 열은 화면에 표시하지 않고 저장하지도 않습니다. 원본 파일도 가능하면 해당 열을 지운 뒤 보관하세요.")
        raw = privacy.drop_sensitive(raw, res)
        st.write(f"**1. 파일 확인** — {len(raw):,}행, 열 {len(raw.columns)}개")
        st.dataframe(raw.head(5), hide_index=True, width="stretch")

        # 2) 열 매핑
        st.write("**2. 열 매핑** — 자동 추정 결과를 확인하세요. 날짜와 금액은 필수입니다.")
        guess = personal.guess_mapping(raw.columns)
        opts = [None] + list(raw.columns)
        mapping = {}
        cols = st.columns(5)
        for i, t in enumerate(personal.TARGETS):
            mapping[t] = cols[i % 5].selectbox(TARGET_KO[t], opts, index=opts.index(guess[t]),
                                               format_func=lambda x: "(없음)" if x is None else str(x), key=f"map_{t}")
        d1, d2 = st.columns(2)
        default_cur = d1.selectbox("통화 열이나 기호가 없을 때 기본 통화", personal.CURRENCIES)
        default_card = d2.text_input("카드 별칭 열이 없을 때 기본 별칭", value="내 카드")
        if not (mapping["date"] or mapping["datetime"]) or not mapping["amount"]:
            st.error("날짜(또는 날짜+시각)와 금액 열을 지정하세요.")
            st.stop()

        # 3) 정규화와 오류 행 수정
        ok, err = personal.normalize(raw, mapping, default_cur, default_card or "내 카드")
        if st.session_state.get("up_fixed") is not None:
            ok = pd.concat([ok, st.session_state["up_fixed"]], ignore_index=True)
        st.write(f"**3. 형식 정규화** — 정상 {len(ok):,}행, 오류 {len(err):,}행")
        if len(err):
            st.caption("오류 행은 아래 표에서 직접 고친 뒤 ‘다시 검사’를 누르세요. 고치지 않은 행은 저장되지 않습니다.")
            edited = st.data_editor(err, hide_index=True, disabled=["row", "reason"], key=f"err_{fid}", width="stretch")
            if st.button("다시 검사"):
                fix_map = {t: (t if t in edited.columns else None) for t in personal.TARGETS}
                fixed, still = personal.normalize(edited.drop(columns=["row", "reason"]), fix_map, default_cur,
                                                  default_card or "내 카드")
                prev = st.session_state.get("up_fixed")
                st.session_state["up_fixed"] = pd.concat([prev, fixed]) if prev is not None else fixed
                st.success(f"{len(fixed)}행 수정 반영, {len(still)}행은 여전히 오류입니다.")
                st.rerun()

        # 4) 중복 확인 후 저장
        if len(ok):
            dup = personal.find_duplicates(ok, store.query())
            preview = ok[list(VIEW_COLS)].rename(columns=VIEW_COLS)
            preview.insert(0, "저장", ~dup.values)
            preview.insert(1, "중복 의심", dup.values)
            st.write(f"**4. 저장할 거래 선택** — 중복 의심 {int(dup.sum())}행은 기본으로 제외됩니다.")
            chosen = st.data_editor(preview, hide_index=True, disabled=list(preview.columns[1:]),
                                    key=f"save_{fid}", width="stretch")
            if st.button("선택한 거래 저장", type="primary"):
                sel = ok[chosen["저장"].values]
                n = store.insert(sel)
                store.log_import(data, n, len(err), int(dup.sum()))
                st.session_state["up_fixed"] = None
                flash(f"{n}건을 저장했습니다. (오류 {len(err)}행·중복 의심 {int(dup.sum())}행 제외)")

# ---------------------------------------------------------------- 편집·내보내기
if section == "내역 편집·내보내기":
    allrows = store.query()
    if allrows.empty:
        st.write("저장된 거래가 없습니다.")
    else:
        f1, f2, f3 = st.columns([2, 2, 1])
        dmin = pd.to_datetime(allrows["transaction_date"]).min().date()
        dmax = pd.to_datetime(allrows["transaction_date"]).max().date()
        period = f1.date_input("기간", value=(dmin, dmax), format="YYYY-MM-DD")
        card = f2.selectbox("카드", [None] + store.cards(), format_func=lambda x: "전체" if x is None else x)
        show_dup = f3.checkbox("중복 의심만")
        start, end = (period if isinstance(period, tuple) and len(period) == 2 else (dmin, dmax))
        rows = store.query(start, end, card).reset_index(drop=True)
        dup = personal.find_duplicates(rows) | personal.find_duplicates(rows.iloc[::-1]).iloc[::-1].values
        if show_dup:
            rows = rows[dup.values].reset_index(drop=True)
            dup = dup[dup]
        view = rows[list(VIEW_COLS)].rename(columns=VIEW_COLS)
        view.insert(0, "삭제", False)
        view.insert(1, "중복 의심", dup.values)
        st.caption(f"{len(view):,}건 · 칸을 직접 고친 뒤 ‘변경 저장’을 누르세요. 거래 추가는 ‘직접 입력’ 탭에서 합니다. 시도는 주소에서 자동 계산됩니다.")
        edited = st.data_editor(
            view, hide_index=True, width="stretch", key="editor",
            disabled=["중복 의심", "시도(파생)"],
            column_config={
                "금액": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
                "통화": st.column_config.SelectboxColumn(options=personal.CURRENCIES, required=True),
                "유형": st.column_config.SelectboxColumn(options=["online", "offline"]),
                "상태": st.column_config.SelectboxColumn(options=["approved", "cancelled"], required=True),
            },
        )
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("변경 저장", type="primary"):
            back = edited.drop(columns=["삭제", "중복 의심"]).rename(columns={v: k for k, v in VIEW_COLS.items()})
            to_delete = rows.loc[edited["삭제"].values, "transaction_id"].tolist()
            as_text = lambda d: d.astype(object).where(d.notna(), "").astype(str).values  # NULL 표기 차이 무시
            changed = (as_text(back) != as_text(rows[list(VIEW_COLS)])).any(axis=1) & ~edited["삭제"].values
            upd = rows[changed].copy()
            for col in VIEW_COLS:
                upd[col] = back.loc[changed, col].values
            upd["card_id"] = upd["card_id"].fillna("내 카드")
            upd, problems = personal.refresh_derived(upd.reset_index(drop=True))
            problems += validate(upd) if len(upd) else []
            if problems:
                st.error("저장하지 않았습니다: " + "; ".join(problems))
            else:
                store.replace(upd)
                store.delete(to_delete)
                flash(f"수정 {len(upd)}건, 삭제 {len(to_delete)}건을 반영했습니다.")
        export = rows[list(VIEW_COLS)].rename(columns=VIEW_COLS)
        b2.download_button("CSV 내보내기", export.to_csv(index=False).encode("utf-8-sig"),
                           file_name="my_card_transactions.csv", mime="text/csv")
        xbuf = io.BytesIO()
        export.to_excel(xbuf, index=False)
        b3.download_button("Excel 내보내기", xbuf.getvalue(), file_name="my_card_transactions.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with b4.popover("전체 삭제"):
            st.write("저장된 개인 거래를 모두 지웁니다. 되돌릴 수 없습니다.")
            confirm = st.text_input("확인을 위해 ‘전체삭제’를 입력하세요")
            if st.button("영구 삭제", disabled=confirm != "전체삭제"):
                store.wipe()
                flash("모든 개인 거래를 삭제했습니다.")

# ---------------------------------------------------------------- 분석
# [v2 변경 2026-11-28] 무엇을: 안내 문구 자리 → 위험도 분석 실행·결과·내보내기 / 왜: 3주차 F8 / 어떻게: src.risk.score 호출
if section == "분석 실행":
    from src import risk

    st.warning(risk.NOTICE)
    st.caption(risk.LIMITATION + " 개인 거래에는 실제 사기 여부(정답)가 없으므로 정확도는 계산하지 않습니다.")
    rows = store.query()
    if rows.empty:
        st.write("저장된 거래가 없습니다.")
    else:
        if st.button("분석 실행", type="primary"):
            res, check = risk.score(rows)
            st.session_state["risk_res"], st.session_state["risk_check"] = res, check
        res, check = st.session_state.get("risk_res"), st.session_state.get("risk_check")
        if res is not None:
            lv = res["risk_level"].value_counts()
            m = st.columns(4)
            for col, name in zip(m, ["높음", "주의", "낮음", "판단 보류"]):
                col.metric(name, f"{int(lv.get(name, 0))}건")
            if check.get("warn"):
                st.warning("입력한 거래의 특징 분포가 학습 데이터와 많이 다릅니다 ("
                           + ", ".join(check["warn_features"]) + "). 점수를 더 조심해서 해석하세요.")
            table = pd.DataFrame({
                "날짜": res["transaction_date"], "시각": res["transaction_time"].fillna("없음"),
                "금액": [f"{a:,.0f} {c}" if c in ("KRW", "JPY") else f"{a:,.2f} {c}" for a, c in zip(res["amount"], res["currency"])],
                "가맹점": res["merchant_name"].fillna("없음"), "카드": res["card_id"],
                "위험 수준": res["risk_level"],
                "점수": res["raw_score"].round(3),
                "주요 요인": res["top_factors"].map(lambda t: " / ".join(x["reason"] for x in t) if t else ""),
                "확인 권장(규칙)": res["rules"].map(lambda r: " / ".join(r)),
                "비고": res["note"].fillna(""),
            })
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption("점수는 보정 전 모델 점수이며, 위험 수준은 미국 검증 구간에서 정한 비용별 임계값으로 나눴습니다. "
                       "‘판단 보류’ 거래는 점수 대신 규칙 기반 확인 항목만 봅니다.")

            prof = risk.card_profile(rows)
            if len(prof):
                st.subheader("카드별 평소 패턴")
                st.dataframe(pd.DataFrame({
                    "카드": prof["card_id"], "통화": prof["currency"], "거래 수": prof["n"],
                    "금액 중앙값": prof["median_amount"].round(0), "금액 25~75%": [f"{a:,.0f} ~ {b:,.0f}" for a, b in zip(prof["q1"], prof["q3"])],
                    "자주 쓰는 시간대": prof["top_hours"].map(lambda h: ", ".join(f"{x:02d}시" for x in h) or "시각 없음"),
                    "일평균 건수": prof["daily_avg"].round(1)}), hide_index=True, width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.download_button("결과 CSV 내보내기", table.to_csv(index=False).encode("utf-8-sig"),
                               file_name="my_card_risk.csv", mime="text/csv")
            xb = io.BytesIO()
            table.to_excel(xb, index=False)
            c2.download_button("결과 Excel 내보내기", xb.getvalue(), file_name="my_card_risk.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if c3.button("결과를 내 거래 내역에 저장"):
                store.replace(risk.to_store_rows(rows, res))
                st.session_state.pop("risk_res", None)
                flash("위험도 결과를 로컬 저장소에 저장했습니다.")
