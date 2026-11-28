"""3주차 테스트: 개인 거래 위험도 규칙 (설계서 4.1, 6.3 ③④, 12장)."""
import pathlib

import numpy as np
import pandas as pd
import pytest

from src import risk
from src.schema import validate
from src.sources import personal
from src.store import PersonalStore


def make(rows, card="생활카드"):
    frames = []
    for r in rows:
        ok, err = personal.manual_entry(r[0], r[1], r[2], r[3], r[4] if len(r) > 4 else "가게", card_alias=card)
        assert err.empty
        frames.append(ok)
    return pd.concat(frames, ignore_index=True)


KRW_DAYS = [(f"2026-11-{d:02d}", "12:30", 9000 + 100 * d, "KRW", "김밥집") for d in range(1, 9)]


def test_risk_level_boundaries():
    hi, watch = risk.thresholds()
    assert risk.risk_level(hi, hi, watch) == "높음"
    assert risk.risk_level(hi - 1e-9, hi, watch) == "주의"
    assert risk.risk_level(watch - 1e-9, hi, watch) == "낮음"
    assert risk.risk_level(np.nan, hi, watch) == "판단 보류"
    assert 0.69 < hi < 0.70 and 0.06 < watch < 0.08  # 기존 측정 임계값


def test_personal_model_excludes_hour_and_routine_is_low():
    assert "hour_rarity" not in risk.PERSONAL_FEATURES
    hi, watch = risk.personal_thresholds()
    assert 0 < watch < hi < 1
    # 매일 같은 시간대의 비슷한 금액 결제는 '낮음'이어야 한다 (v1 모델에서는 외삽으로 '주의'가 나왔던 사례)
    rows = [(f"2026-10-{d:02d}", "12:%02d" % d, 8000 + 300 * d, "KRW", "김밥집") for d in range(1, 21)]
    res, check = risk.score(make(rows))
    assert (res["risk_level"].iloc[5:] == "낮음").all() and not check["warn"]


def test_new_hour_rule_and_large_night_purchase():
    rows = [(f"2026-10-{d:02d}", "12:%02d" % d, 8000 + 300 * d, "KRW", "김밥집") for d in range(1, 21)]
    rows += [("2026-10-21", "03:12", 450000, "KRW", "해외직구몰")]
    res, _ = risk.score(make(rows))
    r = res.iloc[-1]
    assert r["risk_level"] in ("주의", "높음")
    assert any("03시대" in x for x in r["rules"]) and any("배" in x for x in r["rules"])


def test_low_history_non_usd_is_held_then_scored():
    res, _ = risk.score(make(KRW_DAYS))
    first5, rest = res.iloc[:5], res.iloc[5:]
    assert (first5["risk_level"] == "판단 보류").all()
    assert first5["note"].str.contains("USD 외 통화").all()
    assert rest["raw_score"].notna().all() and rest["risk_level"].isin(["높음", "주의", "낮음"]).all()
    assert rest["top_factors"].map(len).eq(3).all()
    assert (rest["model_version"] == risk.MODEL_VERSION).all()


def test_usd_low_history_scored_with_note():
    res, _ = risk.score(make([("2026-11-01", "10:00", 30, "USD", "Shop")]))
    r = res.iloc[0]
    assert pd.notna(r["raw_score"]) and "신뢰도 낮음" in r["note"]


def test_day_precision_and_cancelled_are_held():
    df = make(KRW_DAYS + [("2026-11-09", None, 9000, "KRW", "김밥집")])
    canc, _ = personal.manual_entry("2026-11-09", "13:00", 9000, "KRW", "김밥집", card_alias="생활카드")
    canc["transaction_status"] = "cancelled"
    res, _ = risk.score(pd.concat([df, canc], ignore_index=True))
    day = res[res["time_precision"] == "day"].iloc[0]
    assert day["risk_level"] == "판단 보류" and "결제 시각" in day["note"]
    c = res[res["transaction_status"] == "cancelled"].iloc[0]
    assert c["risk_level"] == "판단 보류" and "취소" in c["note"]


def test_currency_histories_are_separate():
    # 원화 이력 8건이 있어도 첫 달러 거래는 '이력 부족'으로 본다
    df = make(KRW_DAYS + [("2026-11-10", "09:00", 25, "USD", "Amazon")])
    res, _ = risk.score(df)
    usd = res[res["currency"] == "USD"].iloc[0]
    assert usd["insufficient_history"] == 1


def test_rule_flags():
    rows = KRW_DAYS + [("2026-11-09", "10:00", 200000, "KRW", "명품관")]
    rows += [("2026-11-10", f"1{i}:00", 3000, "KRW", "편의점") for i in range(3)]
    res, _ = risk.score(make(rows))
    big = res[res["amount"] == 200000].iloc[0]
    assert any("배" in f for f in big["rules"])
    conv = res[res["merchant_name"] == "편의점"]
    assert conv["rules"].map(lambda r: any("같은 가맹점" in x for x in r)).all()


def test_distribution_check_shape():
    res, check = risk.score(make(KRW_DAYS))
    assert set(check) >= {"checked", "out_of_range", "warn"}
    assert check["checked"] == int(res["raw_score"].notna().sum())


def test_no_labels_used_or_created():
    res, _ = risk.score(make(KRW_DAYS))
    assert "is_fraud" not in res.columns or res["is_fraud"].isna().all()


def test_results_roundtrip_store(tmp_path):
    df = make(KRW_DAYS)
    st = PersonalStore(tmp_path / "p")
    st.insert(df)
    rows = st.query()
    res, _ = risk.score(rows)
    out = risk.to_store_rows(rows, res)
    assert validate(out) == []
    st.replace(out)
    back = st.query()
    assert back["risk_level"].notna().all() and back["is_fraud"].isna().all()


def test_country_experiments_never_touch_personal_or_mix_labels():
    text = pathlib.Path("src/experiments_country.py").read_text(encoding="utf-8")
    for word in ("PersonalStore", "personal.db", "CardFDS", "src.store"):
        assert word not in text
    # 미국·한국 학습 데이터를 합치는 경로가 없음. concat은 결과 표를 합치는 한 줄에만 허용
    # (학습 행 출처는 tests/test_kr_experiments.py의 test_cross_country_trains_each_country_separately에서 확인)
    lines = [l for l in text.splitlines() if "concat" in l]
    assert len(lines) == 1 and "결과 표만 합침" in lines[0] and "kres[k]" in lines[0], lines
