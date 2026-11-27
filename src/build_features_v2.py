"""2주차: v2 특징 데이터 생성과 재계산 검증.

실행: python -m src.build_features_v2
산출: data/research/us/features_v2.parquet  (기존 v1 특징 6개 + 신규 특징, segment 포함)
      data/research/kr/features_v2.parquet  (한국 원본이 있을 때만)
      outputs/v2/feature_checks_v2.csv      (무작위 2,000건 직접 재계산 일치 여부)

- 미국 v1 특징은 data/features.parquet 값을 그대로 붙인다 (기존 실험과 동일한 값 보장).
- 한국 v1 특징은 기존 build_features를 시간대 시각으로 호출해 계산한다.
  5·15·60분 건수는 정확한 시각이 없어 의미가 없으므로 NaN으로 지운다.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .features import build_features, global_stats
from .features_ext import NEW_FEATURES, add_extended, event_seconds, haversine_km

US = "data/research/us"
KR = "data/research/kr"
V1 = ["amt_ratio", "count_5min", "count_15min", "count_60min", "hour_rarity", "insufficient_history"]


def build_us() -> pd.DataFrame:
    common = pd.read_parquet(f"{US}/transactions.parquet")
    ext = add_extended(common)
    v1 = pd.read_parquet("data/features.parquet", columns=["trans_num"] + V1).rename(columns={"trans_num": "transaction_id"})
    out = common[["transaction_id", "card_id", "transaction_date", "transaction_time", "time_precision", "amount",
                  "merchant_id", "transaction_type", "latitude", "longitude", "is_fraud", "segment"]] \
        .merge(v1, on="transaction_id", how="left").merge(ext, on="transaction_id", how="left")
    assert out[V1].notna().all().all(), "v1 특징 병합 누락"
    out.to_parquet(f"{US}/features_v2.parquet", index=False)
    return out


def v1_for_hourly(common: pd.DataFrame) -> pd.DataFrame:
    """시간대 정밀도 데이터에 기존 v1 특징(금액 비율·시간대 희소성·이력 부족)을 계산한다."""
    d = pd.DataFrame({
        "trans_num": common["transaction_id"].values, "cc_num": common["card_id"].values,
        "amt": common["amount"].values, "is_fraud": common["is_fraud"].values,
        "ts": pd.to_datetime(common["transaction_date"]) + pd.to_timedelta(common["hour_bucket"].astype(float), unit="h"),
        "segment": common["segment"].values,
    }).dropna(subset=["ts", "amt"])
    d = d.sort_values("ts", kind="mergesort").reset_index(drop=True)
    d["_order"] = np.arange(len(d))
    stats = global_stats(d[d["segment"] == "train"])
    f = build_features(d, stats)
    f[["count_5min", "count_15min", "count_60min"]] = np.nan  # 분 단위 시각 없음
    return f[["trans_num"] + V1].rename(columns={"trans_num": "transaction_id"})


def build_kr() -> pd.DataFrame | None:
    if not os.path.exists(f"{KR}/transactions.parquet"):
        print("[B안] 한국 변환 데이터 없음 → 한국 특징 생성 건너뜀")
        return None
    common = pd.read_parquet(f"{KR}/transactions.parquet")
    extra = pd.read_parquet(f"{KR}/extra.parquet")
    out = common[["transaction_id", "card_id", "transaction_date", "hour_bucket", "amount", "is_fraud", "fraud_type",
                  "segment"]].merge(v1_for_hourly(common), on="transaction_id", how="left") \
        .merge(add_extended(common, extra), on="transaction_id", how="left")
    out.to_parquet(f"{KR}/features_v2.parquet", index=False)
    return out


# ---------------------------------------------------------------- 재계산 검증
def brute_force(common: pd.DataFrame, i: int, sec: pd.Series) -> dict:
    """한 거래의 신규 특징을 원시 이력에서 직접 계산한다 (벡터화 구현과 독립적인 단순 구현)."""
    r = common.iloc[i]
    t = sec.iloc[i]
    hist = common[(common["card_id"] == r.card_id)]
    hsec = sec[hist.index]
    res = {}
    if np.isnan(t):
        return res
    b = np.floor(t / 3600) * 3600
    hb = np.floor(hsec / 3600) * 3600
    res["count_prev_hour_bucket"] = int(((hb >= b - 3600) & (hb < b)).sum())
    day0 = np.floor(t / 86400) * 86400
    res["count_same_day_prior"] = int(((hb >= day0) & (hb < b)).sum())
    if pd.notna(r.merchant_id):
        m = hist["merchant_id"] == r.merchant_id
        res["same_merchant_24h"] = int((m & (hsec >= t - 86400) & (hsec < t)).sum())
    earlier = hist[(hsec < t).values]
    if len(earlier) and pd.notna(r.latitude):
        p = earlier.loc[sec[earlier.index].idxmax()]
        res["dist_prev_km"] = float(haversine_km(p.latitude, p.longitude, r.latitude, r.longitude))
        res["gap_prev_min"] = float((t - sec[p.name]) / 60)
    return res


def verify(common: pd.DataFrame, feats: pd.DataFrame, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    sec = event_seconds(common)
    by_id = feats.set_index("transaction_id")
    rng = np.random.default_rng(seed)
    rows = []
    # 카드별 이력만 다루도록 미리 나눠 속도를 확보
    for i in rng.choice(len(common), n, replace=False):
        expect = brute_force(common, i, sec)
        got = by_id.loc[common.iloc[i].transaction_id]
        for k, v in expect.items():
            rows.append({"feature": k, "match": bool(np.isclose(got[k], v, equal_nan=True))})
    res = pd.DataFrame(rows).groupby("feature")["match"].agg(["sum", "size"]).reset_index()
    res.columns = ["feature", "matched", "checked"]
    return res


def main():
    os.makedirs("outputs/v2", exist_ok=True)
    us = build_us()
    print(f"US features_v2: {len(us):,} rows")
    print(us.groupby("is_fraud")[NEW_FEATURES[:7]].mean().T.round(3))
    common = pd.read_parquet(f"{US}/transactions.parquet")
    checks = verify(common, us)
    checks.insert(0, "source", "us_sparkov")
    checks.to_csv("outputs/v2/feature_checks_v2.csv", index=False, encoding="utf-8-sig")
    print(checks.to_string(index=False))
    build_kr()


if __name__ == "__main__":
    main()
