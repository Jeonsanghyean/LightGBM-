"""2주차 테스트: 한국 변환기(B안: 공식 열 정의로 만든 가짜 행), 신규 특징 경계 조건.

가짜 행은 이 테스트 안에서만 쓰며 분석·결과에는 사용하지 않는다 (설계서 9.1 B2).
"""
import numpy as np
import pandas as pd
import pytest

from src.features_ext import add_extended
from src.schema import conform, validate
from src.sources import aihub_card as kr


def fake_kr(rows):
    base = {c: "0" for c in kr.OFFICIAL_COLUMNS}
    base.update({"기준년월": "202101", "카드이용한도금액": "3000000", "가맹점KEY": "M1", "이상거래유형": "",
                 "이상거래설명": "", "연령": "3", "남녀구분코드": "1", "국내해외여부": "국내", "가맹점광역시도코드": "8"})
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        out.append(d)
    df = pd.DataFrame(out).astype(str)
    return df.rename(columns={"이상거래여부": "이상거래 여부", "이상거래유형": "이상거래 유형"})  # 공식 표기의 공백


ROWS = [
    {"승인일자": "20210105", "승인SEQ": "1", "승인시간대": "9", "카드KEY": "A", "통합승인금액": "17000"},
    {"승인일자": "20210105", "승인SEQ": "2", "승인시간대": "9", "카드KEY": "A", "통합승인금액": "5000", "가맹점KEY": "M2"},
    {"승인일자": "20210105", "승인SEQ": "3", "승인시간대": "10", "카드KEY": "A", "통합승인금액": "990000",
     "이상거래여부": "1", "이상거래유형": "T1", "가맹점광역시도코드": "1"},
    {"승인일자": "20210106", "승인SEQ": "1", "승인시간대": "2", "카드KEY": "A", "통합승인금액": "12000",
     "국내해외여부": "해외", "가맹점광역시도코드": ""},
    {"승인일자": "20210106", "승인SEQ": "1", "승인시간대": "30", "카드KEY": "B", "통합승인금액": "1,000"},
]


def test_kr_columns_and_sensitive_dropped():
    raw = fake_kr(ROWS)
    assert kr.check_columns(kr.normalize_columns(raw))["missing_required"] == []
    common, extra = kr.to_common(raw)
    assert "연령" not in extra.columns and "남녀구분코드" not in extra.columns
    assert not any("연령" in c or "남녀" in c for c in common.columns)


def test_kr_common_schema_rules():
    common, extra = kr.to_common(fake_kr(ROWS))
    assert validate(common) == []
    assert common["transaction_time"].isna().all() and (common["time_precision"] == "hour").all()
    assert common["latitude"].isna().all() and common["merchant_name"].isna().all()
    assert common["merchant_address"].isna().all()
    # 코드표 미확인 → 시도명·온라인 여부·승인 상태는 NULL, 원본 코드는 extra에 보관
    assert common["region"].isna().all() and common["transaction_type"].isna().all()
    assert common["transaction_status"].isna().all()
    assert extra["region_code"].iloc[0] == "8"
    assert common["is_fraud"].tolist() == [0, 0, 1, 0, 0]
    assert common["fraud_type"].iloc[2] == "T1" and pd.isna(common["fraud_type"].iloc[0])
    assert common["country"].tolist()[:3] == ["KR", "KR", "KR"] and pd.isna(common["country"].iloc[3])
    assert common["location_precision"].tolist()[:4] == ["province"] * 3 + ["none"]
    assert extra["is_overseas"].tolist()[:4] == [0, 0, 0, 1]
    assert pd.isna(common["hour_bucket"].iloc[4])  # 30은 0~23 밖 → NULL
    assert common["amount"].iloc[4] == 1000.0


def test_kr_codebook_applied_only_when_given():
    cb = kr.load_codebook()
    cb = {**cb, "region": {"8": "경기"}, "internet_sale": {"0": "offline", "1": "online"}}
    common, _ = kr.to_common(fake_kr(ROWS), cb)
    assert common["region"].iloc[0] == "경기" and common["transaction_type"].iloc[0] == "offline"
    assert pd.isna(common["region"].iloc[3])  # 해외 거래는 시도 없음


def test_kr_missing_required_raises():
    with pytest.raises(ValueError):
        kr.to_common(fake_kr(ROWS).drop(columns=["카드KEY"]))


def test_kr_quality_and_segments():
    raw = fake_kr(ROWS)
    common, extra = kr.to_common(raw)
    q, labels = kr.quality_report(raw, common, extra)
    qd = dict(zip(q["check"], q["value"]))
    assert qd["rows"] == 5 and qd["fraud_rows"] == 1 and qd["hour_out_of_0_23"] == 1
    assert qd["seq_increases_within_card_date"].startswith("1.0000")
    seg = kr.assign_segments(pd.DataFrame({"transaction_date": ["2021-01-01", "2023-06-30", "2023-07-01", "2024-01-01"]}))
    assert seg.tolist() == ["train", "train", "valid", "test"]


def test_extended_on_hourly_kr():
    common, extra = kr.to_common(fake_kr(ROWS))
    f = add_extended(common, extra).set_index(common["transaction_id"])
    ids = common["transaction_id"]
    # 같은 9시 시간대의 두 거래는 서로를 이력으로 세지 않는다
    assert f.loc[ids[0], "count_prev_hour_bucket"] == 0 and f.loc[ids[1], "count_prev_hour_bucket"] == 0
    # 10시 거래: 직전 시간대(9시) 2건, 같은 날 앞선 2건
    assert f.loc[ids[2], "count_prev_hour_bucket"] == 2 and f.loc[ids[2], "count_same_day_prior"] == 2
    # 좌표 없음 → 거리·속도 NaN
    assert f[["dist_prev_km", "speed_kmh", "far_fast"]].isna().all().all()
    # 지역: 10시 거래는 이전 2건이 모두 코드 8, 현재 코드 1 → 처음 보는 지역
    assert f.loc[ids[2], "region_novelty"] == 1.0 and f.loc[ids[2], "region_changed"] == 1.0
    assert np.isnan(f.loc[ids[0], "region_novelty"])  # 이전 거래 없음
    assert f.loc[ids[3], "is_overseas"] == 1
    assert np.isnan(f.loc[ids[4], "count_prev_hour_bucket"])  # 시간대 NULL


def _personal(rows):
    base = dict(source="personal", card_id="c", currency="KRW", location_precision="none", provenance="{}",
                time_precision="minute")
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["transaction_id"] = [f"t{i}" for i in range(len(df))]
    return conform(df)


def test_extended_location_and_online_exclusion():
    df = _personal([
        dict(transaction_date="2026-11-01", transaction_time="10:00", amount=1, latitude=37.5665, longitude=126.9780,
             location_precision="exact", transaction_type="offline"),
        dict(transaction_date="2026-11-01", transaction_time="11:00", amount=1, latitude=35.1796, longitude=129.0756,
             location_precision="exact", transaction_type="offline"),
        dict(transaction_date="2026-11-01", transaction_time="11:10", amount=1, latitude=37.5665, longitude=126.9780,
             location_precision="exact", transaction_type="online"),
        dict(transaction_date="2026-11-01", transaction_time=None, time_precision="day", amount=1),
    ])
    f = add_extended(df).set_index("transaction_id")
    assert 320 < f.loc["t1", "dist_prev_km"] < 330  # 서울-부산 약 325km
    assert f.loc["t1", "gap_prev_min"] == 60 and 320 < f.loc["t1", "speed_kmh"] < 330
    assert f.loc["t1", "far_fast"] == 0  # 500km 미만, 900km/h 미만
    assert np.isnan(f.loc["t2", "speed_kmh"])  # 온라인 거래는 속도 계산 제외
    assert f.loc["t2", "dist_prev_km"] > 300  # 거리 자체는 기록
    assert f.loc[["t3"]].drop(columns=["is_overseas"]).isna().all().all()  # 시각 없음 → 시간 특징 전부 NaN
    # 11:10 거래: 같은 날 앞선 '시간대'는 10시뿐. 11:00 거래는 같은 11시 시간대라 세지 않는다 (국가 공통 정의)
    assert f.loc["t2", "count_same_day_prior"] == 1 and f.loc["t2", "count_prev_hour_bucket"] == 1


def test_extended_far_fast():
    df = _personal([
        dict(transaction_date="2026-11-01", transaction_time="10:00", amount=1, latitude=37.5665, longitude=126.9780,
             location_precision="exact"),
        dict(transaction_date="2026-11-01", transaction_time="11:00", amount=1, latitude=35.6762, longitude=139.6503,
             location_precision="exact"),  # 도쿄, 약 1,150km를 1시간에
    ])
    f = add_extended(df).set_index("transaction_id")
    assert f.loc["t1", "speed_kmh"] > 900 and f.loc["t1", "far_fast"] == 1
