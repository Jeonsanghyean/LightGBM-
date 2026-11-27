"""1~2주차: 원본 점검, 시간순 분할, 특징 추출과 경계 검증.

실행: python -m src.build_data
산출: data/features.parquet, outputs/data_check.csv, outputs/split_summary.csv,
      outputs/feature_checks.csv, models/global_stats.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .features import ALL_FEATURES, build_features, check_raw, global_stats, load_raw, prepare, time_split

RAW = "data/raw"
OUT = "outputs"
MODELS = "models"


def boundary_checks(d: pd.DataFrame) -> pd.DataFrame:
    """동일 시각·시간 창 경계를 무작위 거래에 대해 직접 계산해 대조한다."""
    rng = np.random.default_rng(0)
    rows = []
    by_card = {k: v for k, v in d[["cc_num", "ts", "amt"]].groupby("cc_num")}
    for i in rng.choice(len(d), 2000, replace=False):
        r = d.iloc[i]
        hist = by_card[r.cc_num]
        past = hist[hist["ts"] < r.ts]
        ok = True
        for col, m in (("count_5min", 5), ("count_15min", 15), ("count_60min", 60)):
            ok &= int(((past["ts"] >= r.ts - pd.Timedelta(minutes=m))).sum()) == r[col]
        ok &= len(past) == r.n_prev
        if len(past) >= 5:
            ok &= np.isclose(r.amt / max(past["amt"].median(), 0.01), r.amt_ratio)
            share = (past["ts"].dt.hour == r.ts.hour).mean()
            ok &= np.isclose(1 - share, r.hour_rarity)
        rows.append(ok)
    # 같은 카드·같은 시각 거래가 서로를 이력으로 세지 않는지
    dup = d[d.duplicated(["cc_num", "ts"], keep=False)]
    same_ok = bool(dup.groupby(["cc_num", "ts"])["n_prev"].nunique().eq(1).all()) if len(dup) else True
    return pd.DataFrame(
        [
            {"check": "random_2000_recompute_match", "value": int(np.sum(rows)), "total": len(rows)},
            {"check": "same_card_same_time_groups", "value": int(dup.groupby(["cc_num", "ts"]).ngroups) if len(dup) else 0, "total": None},
            {"check": "same_time_share_history", "value": int(same_ok), "total": 1},
        ]
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    tr = load_raw(f"{RAW}/fraudTrain.csv")
    te = load_raw(f"{RAW}/fraudTest.csv")
    pd.DataFrame([check_raw(tr, "fraudTrain.csv"), check_raw(te, "fraudTest.csv")]).to_csv(
        f"{OUT}/data_check.csv", index=False, encoding="utf-8-sig"
    )

    tr = prepare(tr)
    tr["segment"] = time_split(tr)
    te = prepare(te)
    te["segment"] = "test"
    # 원본 Test는 Train 이후 기간이므로 이어 붙이면 카드 이력이 시간순으로 이어진다
    print("train end:", tr["ts"].max(), "/ test start:", te["ts"].min())

    stats = global_stats(tr[tr["segment"] == "train"])
    with open(f"{MODELS}/global_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    both = pd.concat([tr, te], ignore_index=True)
    both["_order"] = np.arange(len(both))  # 이미 Train→Test 시간순
    feat = build_features(both, stats)

    order = ["train", "valid", "internal_test", "test"]
    summ = (
        feat.groupby("segment")
        .agg(rows=("is_fraud", "size"), fraud=("is_fraud", "sum"), start=("ts", "min"), end=("ts", "max"))
        .reindex(order)
        .reset_index()
    )
    summ["normal"] = summ["rows"] - summ["fraud"]
    summ.to_csv(f"{OUT}/split_summary.csv", index=False, encoding="utf-8-sig")
    print(summ)

    checks = boundary_checks(feat)
    checks.to_csv(f"{OUT}/feature_checks.csv", index=False, encoding="utf-8-sig")
    print(checks)

    keep = ["trans_num", "trans_date_trans_time", "ts", "cc_num", "amt", "merchant", "category",
            "is_fraud", "segment", "n_prev"] + ALL_FEATURES
    feat[keep].to_parquet("data/features.parquet", index=False)
    print("saved data/features.parquet", feat.shape)


if __name__ == "__main__":
    main()
