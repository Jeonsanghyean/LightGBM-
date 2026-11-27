"""미국 Sparkov → 공통 스키마 변환 (설계서 F1).

실행: python -m src.sources.sparkov
산출: data/research/us/transactions.parquet

변환 규칙 (설계서 표 3·4)
- 카드 소유자 이름·주소·생년월일·직업·성별·소유자 좌표는 버린다 (저장하지 않음).
- street/city/zip은 소유자 주소이므로 merchant_address로 쓰지 않는다 → NULL.
- merch_lat/long은 시뮬레이터가 소유자 집 주변에 무작위로 만든 좌표 → location_precision='synthetic_point'.
- merchant 이름의 'fraud_' 접두어는 모든 행에 붙어 있으므로 제거해 표시용으로만 쓴다.
- 온라인 여부는 category 접미어(_net/_pos)에서 파생하고, 판별 불가 업종은 NULL.
- 기존 분할(segment)은 data/features.parquet에서 가져와 함께 저장한다 (기존 실험과 같은 구간).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..schema import conform, pseudonymize, validate

RAW = "data/raw"
OUT = "data/research/us/transactions.parquet"
DROP_PERSONAL_LIKE = ["first", "last", "gender", "street", "city", "state", "zip", "lat", "long",
                      "city_pop", "job", "dob", "unix_time"]


def to_common(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.drop(columns=[c for c in DROP_PERSONAL_LIKE if c in raw.columns])
    ts = pd.to_datetime(raw["trans_date_trans_time"])
    cat = raw["category"].astype(str)
    ttype = np.select([cat.str.endswith("_net"), cat.str.endswith("_pos")], ["online", "offline"], default=None)

    prov_base = {"card_id": "derived", "merchant_id": "derived", "merchant_name": "derived",
                 "hour_bucket": "derived", "latitude": "observed_synthetic", "longitude": "observed_synthetic"}
    prov_typed = json.dumps({**prov_base, "transaction_type": "derived"}, sort_keys=True)
    prov_plain = json.dumps(prov_base, sort_keys=True)

    df = pd.DataFrame({
        "transaction_id": raw["trans_num"].astype(str),
        "source": "us_sparkov",
        "country": "US",
        "card_id": pseudonymize(raw["cc_num"], "us_"),
        "transaction_date": ts.dt.strftime("%Y-%m-%d"),
        "transaction_time": ts.dt.strftime("%H:%M:%S"),
        "hour_bucket": ts.dt.hour,
        "time_precision": "second",
        "amount": raw["amt"].astype(float),
        "currency": "USD",
        "merchant_id": pseudonymize(raw["merchant"], "m_"),
        "merchant_name": raw["merchant"].str.removeprefix("fraud_"),
        "merchant_category": cat,
        "merchant_address": None,
        "region": None,
        "latitude": raw["merch_lat"].astype(float),
        "longitude": raw["merch_long"].astype(float),
        "location_precision": "synthetic_point",
        "transaction_type": ttype,
        "transaction_status": None,
        "is_fraud": raw["is_fraud"].astype(int),
        "fraud_type": None,
        "provenance": np.where(pd.notna(ttype), prov_typed, prov_plain),
    })
    return conform(df)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    parts = [pd.read_csv(f"{RAW}/{f}", index_col=0) for f in ("fraudTrain.csv", "fraudTest.csv")]
    df = to_common(pd.concat(parts, ignore_index=True))
    seg = pd.read_parquet("data/features.parquet", columns=["trans_num", "segment"])
    df = df.merge(seg.rename(columns={"trans_num": "transaction_id"}), on="transaction_id", how="left")
    problems = validate(df)
    if problems:
        raise SystemExit("스키마 검증 실패:\n" + "\n".join(problems))
    df.to_parquet(OUT, index=False)
    print(f"saved {OUT}: {len(df):,} rows")
    print(df.groupby("segment", dropna=False)["is_fraud"].agg(["size", "sum"]))
    print("transaction_type:", df["transaction_type"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
