"""1주차 단위 테스트: 공통 스키마, 민감정보 탐지, 개인 거래 정규화·중복, 로컬 저장소.

실행: python -m pytest -q
"""
import os

import pandas as pd
import pytest

from src import privacy
from src.schema import COLUMNS, conform, validate
from src.sources import personal
from src.store import PersonalStore

TEST_CARD = "4111 1111 1111 1111"  # 결제 업계의 공개 테스트 카드번호 (Luhn 유효)


# ---------------------------------------------------------------- 스키마
def _one(**kw):
    base = dict(transaction_id="t1", source="personal", card_id="c", transaction_date="2026-11-20",
                time_precision="day", amount=1000.0, currency="KRW", location_precision="none", provenance="{}")
    base.update(kw)
    return conform(pd.DataFrame([base]))


def test_schema_valid_row():
    assert validate(_one()) == []


@pytest.mark.parametrize("kw, word", [
    (dict(source="bank"), "source"),
    (dict(amount=-5.0), "음수"),
    (dict(time_precision="minute"), "시각이 없는데"),
    (dict(latitude=37.5, longitude=127.0, location_precision="province"), "좌표"),
    (dict(hour_bucket=24, transaction_time="10:00", time_precision="minute"), "hour_bucket"),
])
def test_schema_violations(kw, word):
    assert any(word in p for p in validate(_one(**kw)))


def test_us_research_parquet_conforms():
    path = "data/research/us/transactions.parquet"
    if not os.path.exists(path):
        pytest.skip("python -m src.sources.sparkov 먼저 실행")
    df = pd.read_parquet(path)
    assert len(df) == 1_852_394
    assert validate(df) == []
    assert df["merchant_address"].isna().all() and df["region"].isna().all()
    assert (df["location_precision"] == "synthetic_point").all()
    assert not df["merchant_name"].str.startswith("fraud_").any()
    for col in ["first", "last", "street", "dob", "job", "cc_num"]:
        assert col not in df.columns
    assert df.groupby("segment")["is_fraud"].sum().to_dict() == {
        "train": 5968, "valid": 706, "internal_test": 832, "test": 2145}


# ---------------------------------------------------------------- 민감정보
def test_luhn():
    assert privacy.luhn_ok("4111111111111111")
    assert not privacy.luhn_ok("4111111111111112")
    assert not privacy.luhn_ok("12345")


def test_scan_blocks_card_number_and_names():
    df = pd.DataFrame({"이용일자": ["2026-11-20"], "메모": [f"카드 {TEST_CARD}"], "CVC": ["123"],
                       "금액": ["10,000"], "승인번호": ["12345678"], "shopping": ["x"]})
    res = privacy.scan(df)
    assert set(res.drop_columns) == {"메모", "CVC"}
    assert "4111" not in " ".join(res.findings)  # 안내문에 값이 들어가지 않음
    assert list(privacy.drop_sensitive(df, res).columns) == ["이용일자", "금액", "승인번호", "shopping"]


def test_scan_rrn_and_masked():
    df = pd.DataFrame({"비고": ["900101-1234567"], "카드": ["1234-****-****-5678"]})
    res = privacy.scan(df)
    assert res.drop_columns == ["비고"] and res.masked_card_columns == ["카드"]


# ---------------------------------------------------------------- 파싱
@pytest.mark.parametrize("s", ["2026.11.20", "2026-11-20", "20261120", "11/20/2026", "2026/11/20", "2026년 11월 20일"])
def test_parse_date(s):
    assert personal.parse_date(s) == "2026-11-20"


def test_parse_date_invalid():
    assert personal.parse_date("2026-13-40") is None and personal.parse_date("어제") is None


@pytest.mark.parametrize("s, out", [("14:05", ("14:05", "minute")), ("1405", ("14:05", "minute")),
                                    ("오후 2:05", ("14:05", "minute")), ("14:05:30", ("14:05:30", "second")),
                                    ("오전 12:10", ("00:10", "minute")), ("25:00", (None, None))])
def test_parse_time(s, out):
    assert personal.parse_time(s) == out


@pytest.mark.parametrize("s, out", [("1,234원", (1234.0, "KRW", False)), ("₩1,234", (1234.0, "KRW", False)),
                                    ("$12.50", (12.5, "USD", False)), ("(1,234)", (1234.0, None, True)),
                                    ("-5,000", (5000.0, None, True)), ("12.5 EUR", (12.5, "EUR", False))])
def test_parse_amount(s, out):
    assert personal.parse_amount(s) == out


@pytest.mark.parametrize("addr, region", [("서울특별시 중구 세종대로 110", "서울"), ("경기도 광주시 경안동", "경기"),
                                          ("광주광역시 북구", "광주"), ("전북특별자치도 전주시", "전북"),
                                          ("충청남도 천안시", "충남"), ("Seoul", None)])
def test_extract_region(addr, region):
    assert personal.extract_region(addr) == region


def test_guess_mapping_korean_statement():
    cols = ["이용일자", "이용시간", "이용하신곳", "이용금액", "승인구분", "가맹점주소"]
    m = personal.guess_mapping(cols)
    assert m["date"] == "이용일자" and m["time"] == "이용시간" and m["merchant_name"] == "이용하신곳"
    assert m["amount"] == "이용금액" and m["transaction_status"] == "승인구분" and m["merchant_address"] == "가맹점주소"


def test_guess_mapping_datetime_column():
    m = personal.guess_mapping(["거래일시", "가맹점명", "결제금액"])
    assert m["datetime"] == "거래일시" and m["date"] is None


def test_normalize_end_to_end():
    raw = pd.DataFrame({
        "거래일시": ["2026.11.20 14:05", "2026-11-21", "잘못된날짜", "2026-11-22 09:00"],
        "가맹점명": ["스타벅스", "GS25", "x", "환불가게"],
        "결제금액": ["5,500원", "$3.00", "100", "-12,000"],
        "가맹점주소": ["서울 강남구", None, None, "부산 해운대구"],
    })
    ok, err = personal.normalize(raw, personal.guess_mapping(raw.columns))
    assert len(ok) == 3 and len(err) == 1 and "날짜" in err.iloc[0]["reason"]
    assert validate(ok) == []
    a, b, c = ok.iloc[0], ok.iloc[1], ok.iloc[2]
    assert (a.transaction_time, a.time_precision, a.hour_bucket, a.region) == ("14:05", "minute", 14, "서울")
    assert pd.isna(b.transaction_time) and b.time_precision == "day" and b.currency == "USD"
    assert b.location_precision == "none" and pd.isna(b.latitude)  # 주소 없으면 위치 없음, 좌표 추정 안 함
    assert c.transaction_status == "cancelled" and c.amount == 12000
    assert ok["is_fraud"].isna().all()


def test_manual_entry_and_duplicates():
    a, _ = personal.manual_entry("2026-11-20", "12:30", 9000, "KRW", "김밥집", card_alias="생활카드")
    b, _ = personal.manual_entry("2026-11-20", "12:30", 9000, "KRW", "김밥 집", card_alias="생활카드")
    c, _ = personal.manual_entry("2026-11-20", "12:31", 9000, "KRW", "김밥집", card_alias="생활카드")
    both = pd.concat([a, b, c], ignore_index=True)
    assert personal.find_duplicates(both).tolist() == [False, True, False]
    assert personal.find_duplicates(c, existing=a).tolist() == [False]


def test_read_any_cp949_and_excel(tmp_path):
    df = pd.DataFrame({"이용일자": ["2026-11-20"], "이용금액": ["1,000"]})
    assert personal.read_any(df.to_csv(index=False).encode("cp949"), "a.csv").equals(df)
    xlsx = tmp_path / "a.xlsx"
    df.to_excel(xlsx, index=False)
    assert personal.read_any(xlsx.read_bytes(), "a.xlsx").equals(df)


# ---------------------------------------------------------------- 저장소
def test_store_roundtrip(tmp_path):
    st = PersonalStore(tmp_path / "p")
    ok, _ = personal.manual_entry("2026-11-20", "12:30", 9000, "KRW", "김밥집", card_alias="생활카드")
    ok2, _ = personal.manual_entry("2026-11-25", None, 30, "USD", "Amazon", card_alias="해외카드")
    assert st.insert(pd.concat([ok, ok2])) == 2
    assert st.count() == 2 and st.cards() == ["생활카드", "해외카드"]
    got = st.query(start="2026-11-21")
    assert len(got) == 1 and got.iloc[0]["currency"] == "USD" and pd.isna(got.iloc[0]["transaction_time"])
    assert validate(st.query()) == []
    edited = st.query(card="생활카드").assign(amount=9500.0)
    st.replace(edited)
    assert st.query(card="생활카드").iloc[0]["amount"] == 9500.0
    assert st.delete([ok2.iloc[0]["transaction_id"]]) == 1
    st.log_import(b"file", 1, 0, 0)
    st.wipe()
    assert st.count() == 0
    raw = (tmp_path / "p" / "personal.db").read_bytes()
    assert "김밥집".encode() not in raw and b"Amazon" not in raw  # VACUUM 후 파일에 흔적 없음


def test_store_rejects_invalid(tmp_path):
    st = PersonalStore(tmp_path / "p")
    bad = _one(amount=-1.0)
    with pytest.raises(ValueError):
        st.insert(bad)


def test_store_has_no_sensitive_columns():
    assert not any(k in c for c in COLUMNS for k in ("cvc", "card_number", "password", "cc_num"))


def test_training_code_never_reads_personal_store():
    import pathlib
    for f in ["src/build_data.py", "src/experiments.py", "src/features.py"]:
        text = pathlib.Path(f).read_text(encoding="utf-8")
        for word in ("PersonalStore", "personal.db", "CardFDS", "src.store", "from .store", "sources.personal"):
            assert word not in text, (f, word)


def test_refresh_derived_after_edit():
    ok, _ = personal.manual_entry("2026-11-20", "12:30", 9000, "KRW", "김밥집", "서울 중구")
    edited = ok.copy()
    edited.loc[0, ["transaction_time", "merchant_address"]] = ["오후 11:05", "제주특별자치도 제주시"]
    out, problems = personal.refresh_derived(edited)
    assert problems == [] and validate(out) == []
    r = out.iloc[0]
    assert (r.transaction_time, r.hour_bucket, r.region, r.location_precision) == ("23:05", 23, "제주", "province")
    edited.loc[0, ["transaction_time", "merchant_address"]] = [None, None]
    out, _ = personal.refresh_derived(edited)
    assert out.iloc[0].time_precision == "day" and out.iloc[0].location_precision == "none"
    edited.loc[0, "transaction_date"] = "2026-02-30"
    assert personal.refresh_derived(edited)[1]  # 잘못된 날짜는 저장 거부
