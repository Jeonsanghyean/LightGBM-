"""공통 거래 스키마 (설계서 표 4).

미국 Sparkov, 한국 AI-Hub, 개인 거래가 모두 이 열 구조로 변환된다.
원본에 없는 값은 NULL(None/NaN)로 두고, 다른 열에서 계산한 값은 provenance에 'derived'로 남긴다.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

# 열 이름 → (pandas dtype, NULL 허용 여부)
COLUMNS: dict[str, tuple[str, bool]] = {
    "transaction_id": ("string", False),
    "source": ("string", False),
    "country": ("string", True),
    "card_id": ("string", False),
    "transaction_date": ("string", False),  # YYYY-MM-DD
    "transaction_time": ("string", True),  # HH:MM 또는 HH:MM:SS
    "hour_bucket": ("Int64", True),
    "time_precision": ("string", False),
    "amount": ("float64", False),
    "currency": ("string", False),
    "merchant_id": ("string", True),
    "merchant_name": ("string", True),
    "merchant_category": ("string", True),
    "merchant_address": ("string", True),
    "region": ("string", True),
    "latitude": ("float64", True),
    "longitude": ("float64", True),
    "location_precision": ("string", False),
    "transaction_type": ("string", True),
    "transaction_status": ("string", True),
    "is_fraud": ("Int64", True),
    "fraud_type": ("string", True),
    "fraud_probability": ("float64", True),
    "raw_score": ("float64", True),
    "risk_level": ("string", True),
    "model_version": ("string", True),
    "provenance": ("string", False),  # JSON: {필드: observed|derived|user_input}
}

ALLOWED = {
    "source": {"us_sparkov", "kr_aihub", "personal"},
    "time_precision": {"second", "minute", "hour", "day"},
    "location_precision": {"exact", "synthetic_point", "province", "country", "none"},
    "transaction_type": {"online", "offline"},
    "transaction_status": {"approved", "cancelled"},
    "risk_level": {"높음", "주의", "낮음", "판단 보류"},
}

# 고정 salt: 공개 합성 데이터의 키를 화면·결과에 원본 그대로 노출하지 않기 위한 가명화 (설계서 6.3 ②)
_SALT = "card-fds-v2"


def pseudonymize(values: pd.Series, prefix: str = "") -> pd.Series:
    """키 → SHA-256 앞 16자리. 같은 키는 항상 같은 값이 된다."""
    uniq = pd.Series(values.astype(str).unique())
    mapped = {
        v: prefix + hashlib.sha256(f"{_SALT}|{v}".encode()).hexdigest()[:16] for v in uniq
    }
    return values.astype(str).map(mapped)


def provenance(**fields: str) -> str:
    return json.dumps(fields, ensure_ascii=False, sort_keys=True)


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, (t, _) in COLUMNS.items()})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """빠진 열은 NULL로 추가하고 열 순서·타입을 맞춘다. 스키마 밖 열은 뒤에 그대로 둔다."""
    out = df.copy()
    for c, (t, _) in COLUMNS.items():
        if c not in out.columns:
            out[c] = pd.Series([None] * len(out), dtype=t, index=out.index)
        else:
            out[c] = out[c].astype(t)
    extra = [c for c in out.columns if c not in COLUMNS]
    return out[list(COLUMNS) + extra]


def validate(df: pd.DataFrame) -> list[str]:
    """위반 내용을 문장 목록으로 돌려준다. 빈 목록이면 통과."""
    problems = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"필수 열 없음: {missing}"]
    for c, (_, nullable) in COLUMNS.items():
        n_null = int(df[c].isna().sum())
        if not nullable and n_null:
            problems.append(f"{c}: NULL {n_null}건 (NULL 불가)")
    for c, allowed in ALLOWED.items():
        bad = df[c].dropna()
        bad = bad[~bad.isin(allowed)]
        if len(bad):
            problems.append(f"{c}: 허용되지 않은 값 {sorted(bad.unique())[:5]}")
    if df["transaction_id"].duplicated().any():
        problems.append(f"transaction_id 중복 {int(df['transaction_id'].duplicated().sum())}건")
    if (df["amount"] < 0).any():
        problems.append("amount 음수 존재 (취소는 transaction_status로 표시)")
    hb = df["hour_bucket"].dropna()
    if len(hb) and ((hb < 0) | (hb > 23)).any():
        problems.append("hour_bucket 범위(0~23) 위반")
    # 시각이 없는데 정밀도가 초·분이면 모순
    no_time = df["transaction_time"].isna() & df["time_precision"].isin(["second", "minute"])
    if no_time.any():
        problems.append(f"시각이 없는데 time_precision이 second/minute인 행 {int(no_time.sum())}건")
    has_xy = df["latitude"].notna() & df["longitude"].notna()
    bad_loc = has_xy & ~df["location_precision"].isin(["exact", "synthetic_point"])
    if bad_loc.any():
        problems.append(f"좌표가 있는데 location_precision이 좌표형이 아닌 행 {int(bad_loc.sum())}건")
    return problems
