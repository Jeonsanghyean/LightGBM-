"""한국 AI-Hub 카드거래 합성데이터 → 공통 스키마 (설계서 F2, 9.1절 B안).

실행: python -m src.sources.aihub_card
입력: data/research/kr/raw/*카드거래*.csv  (AI-Hub에서 사용자가 직접 신청·다운로드)
산출: data/research/kr/transactions.parquet   공통 스키마
      data/research/kr/extra.parquet          공통 스키마에 없는 원본 열 (한도, 원본 코드 등)
      outputs/v2/kr_quality.csv, kr_label_types.csv

B안 상태(2026-11-27): 원본 미확보. 이 변환기는 공식 페이지의 32개 열 정의로 작성했고,
tests/test_week2.py의 가짜 행으로만 검증했다. 실제 파일을 받으면 품질 보고서부터 확인한다.

원칙
- 정확한 시각이 없으므로 transaction_time = NULL, time_precision = 'hour'.
- 좌표·주소·가맹점 이름은 원본에 없으므로 NULL.
- 코드의 의미(시도·온라인 여부·승인 상태)는 config/aihub_codebook.yaml에 확인된 값이 있을 때만 변환한다.
- 연령·성별은 읽은 직후 버린다.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..schema import conform, pseudonymize, validate

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/research/kr/raw"
OUT_DIR = ROOT / "data/research/kr"
CODEBOOK = ROOT / "config/aihub_codebook.yaml"

# 공식 페이지 '라벨링데이터 포맷 * 카드거래 데이터' 32개 열 (2026-11-25 확인)
OFFICIAL_COLUMNS = [
    "기준년월", "승인일자", "승인SEQ", "승인시간대", "카드KEY", "통합승인금액", "개인법인구분코드_회원", "국내해외여부",
    "카드이용한도금액", "가맹점광역시도코드", "이상거래여부", "이상거래유형", "이상거래설명", "연령", "남녀구분코드",
    "승인거래코드", "승인발생경로코드", "가맹점승인업종코드", "가맹점누적매출금액_구간화", "개인법인구분코드_가맹점",
    "가맹점여부_신규", "인터넷판매여부", "경과일수_최종이용일자", "가맹점상태코드", "가맹점형태구분코드", "로고구분코드",
    "일시불할부구분코드", "전월_매출건수", "카드구분코드", "할부가능개월수", "가맹점KEY", "전월_매출금액",
]
REQUIRED = ["승인일자", "승인SEQ", "승인시간대", "카드KEY", "통합승인금액", "국내해외여부", "가맹점광역시도코드",
            "이상거래여부", "이상거래유형", "가맹점승인업종코드", "인터넷판매여부", "승인거래코드", "가맹점KEY"]
DROP_SENSITIVE = ["연령", "남녀구분코드"]
# 미래 정보가 섞였을 수 있어 모델에 쓰지 않는 집계 열 (설계서 4.1 F2 제약)
NOT_FOR_MODEL = ["경과일수_최종이용일자", "전월_매출건수", "전월_매출금액", "가맹점누적매출금액_구간화"]


def load_codebook(path: Path = CODEBOOK) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """'이상거래 여부'처럼 공백이 들어간 열 이름을 공식 표기로 맞춘다."""
    return df.rename(columns=lambda c: re.sub(r"\s+", "", str(c)).replace("﻿", ""))


def read_quarters(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw_dir / "*카드*.csv")))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        for enc in ("utf-8-sig", "cp949"):
            try:
                part = pd.read_csv(f, dtype=str, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        part = normalize_columns(part).drop(columns=DROP_SENSITIVE, errors="ignore")
        part["source_file"] = os.path.basename(f)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def check_columns(df: pd.DataFrame) -> dict:
    present = set(df.columns)
    return {"missing_required": [c for c in REQUIRED if c not in present],
            "missing_official": [c for c in OFFICIAL_COLUMNS if c not in present and c not in DROP_SENSITIVE],
            "unexpected": sorted(present - set(OFFICIAL_COLUMNS) - {"source_file"})}


def _map(series: pd.Series, mapping: dict) -> pd.Series:
    if not mapping:
        return pd.Series([None] * len(series), index=series.index, dtype="object")
    return series.astype(str).str.strip().map({str(k): v for k, v in mapping.items()})


def to_common(raw: pd.DataFrame, codebook: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """→ (공통 스키마, extra). 필수 열이 없으면 ValueError."""
    cb = codebook or load_codebook()
    raw = normalize_columns(raw).drop(columns=DROP_SENSITIVE, errors="ignore")
    miss = check_columns(raw)["missing_required"]
    if miss:
        raise ValueError(f"필수 열 없음: {miss}")

    date = pd.to_datetime(raw["승인일자"].str.strip(), format="%Y%m%d", errors="coerce")
    hour = pd.to_numeric(raw["승인시간대"], errors="coerce")
    hour = hour.where(hour.between(0, 23))  # 범위 밖(구간 코드일 가능성)은 NULL로 두고 품질 보고서에 기록
    seq = pd.to_numeric(raw["승인SEQ"], errors="coerce")
    amount = pd.to_numeric(raw["통합승인금액"].astype(str).str.replace(",", ""), errors="coerce")

    dom = raw["국내해외여부"].astype(str).str.strip()
    is_dom = dom.isin(cb["domestic_values"])
    is_ovs = dom.isin(cb["overseas_values"])
    region = _map(raw["가맹점광역시도코드"], cb["region"]).where(is_dom)
    fraud_raw = raw["이상거래여부"].astype(str).str.strip()
    is_fraud = np.select([fraud_raw.isin(cb["fraud_positive_values"]), fraud_raw.isin(cb["fraud_negative_values"])],
                         [1, 0], default=-1)

    key = raw["승인일자"].astype(str) + "|" + raw["승인SEQ"].astype(str) + "|" + raw["카드KEY"].astype(str)
    tid = key.map(lambda k: "kr_" + hashlib.sha256(k.encode()).hexdigest()[:20])

    prov = {"card_id": "derived", "merchant_id": "derived", "transaction_id": "derived",
            "hour_bucket": "observed", "currency": "assumed", "location_precision": "observed_code"}
    if cb["region"]:
        prov["region"] = "codebook"
    common = pd.DataFrame({
        "transaction_id": tid,
        "source": "kr_aihub",
        "country": np.where(is_dom, "KR", None),
        "card_id": pseudonymize(raw["카드KEY"], "kr_"),
        "transaction_date": date.dt.strftime("%Y-%m-%d"),
        "transaction_time": None,
        "hour_bucket": hour.astype("Int64"),
        "time_precision": "hour",
        "amount": amount,
        "currency": cb["currency"],
        "merchant_id": pseudonymize(raw["가맹점KEY"], "km_"),
        "merchant_name": None,
        "merchant_category": raw["가맹점승인업종코드"].astype(str).str.strip(),
        "merchant_address": None,
        "region": region,
        "latitude": np.nan,
        "longitude": np.nan,
        # 국내 거래는 시도 코드가 있으므로 province(코드 의미 미확인이어도 '시도 단위' 정밀도는 맞음), 해외는 국가 미제공
        "location_precision": np.where(is_dom & raw["가맹점광역시도코드"].notna(), "province", "none"),
        "transaction_type": _map(raw["인터넷판매여부"], cb["internet_sale"]),
        "transaction_status": _map(raw["승인거래코드"], cb["approval_code"]),
        "is_fraud": pd.Series(is_fraud).where(is_fraud >= 0).astype("Int64").values,
        "fraud_type": raw["이상거래유형"].where(raw["이상거래유형"].astype(str).str.strip().ne("")),
        "provenance": json.dumps(prov, sort_keys=True),
    })
    common = conform(common)
    extra = pd.DataFrame({
        "transaction_id": tid,
        "seq": seq,
        "region_code": raw["가맹점광역시도코드"].astype(str).str.strip(),
        "is_overseas": np.where(is_ovs, 1, np.where(is_dom, 0, np.nan)),
        "card_limit": pd.to_numeric(raw.get("카드이용한도금액"), errors="coerce"),
        "internet_sale_code": raw["인터넷판매여부"],
        "approval_code": raw["승인거래코드"],
        "fraud_desc": raw.get("이상거래설명"),
        "source_file": raw.get("source_file"),
        **{f"raw_{c}": raw[c] for c in NOT_FOR_MODEL if c in raw.columns},
    })
    return common, extra


def quality_report(raw: pd.DataFrame, common: pd.DataFrame, extra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = normalize_columns(raw)
    cols = check_columns(raw)
    hour_raw = pd.to_numeric(raw["승인시간대"], errors="coerce")
    rows = [
        ("rows", len(raw)),
        ("fraud_rows", int((common["is_fraud"] == 1).sum())),
        ("fraud_label_unparsed", int(common["is_fraud"].isna().sum())),
        ("missing_required_columns", ";".join(cols["missing_required"]) or "-"),
        ("missing_official_columns", ";".join(cols["missing_official"]) or "-"),
        ("unexpected_columns", ";".join(cols["unexpected"]) or "-"),
        ("bad_date", int(common["transaction_date"].isna().sum())),
        ("bad_amount", int(common["amount"].isna().sum())),
        ("negative_amount", int((common["amount"] < 0).sum())),
        ("hour_values", ",".join(map(str, sorted(hour_raw.dropna().unique().astype(int))[:40]))),
        ("hour_out_of_0_23", int(hour_raw.notna().sum() - common["hour_bucket"].notna().sum())),
        ("duplicate_key(date,seq,card)", int(common["transaction_id"].duplicated().sum())),
        ("region_codes", ",".join(sorted(extra["region_code"].dropna().unique())[:40])),
        ("domestic_values", ",".join(sorted(raw["국내해외여부"].dropna().astype(str).unique()))),
        ("internet_sale_values", ",".join(sorted(raw["인터넷판매여부"].dropna().astype(str).unique()))),
        ("approval_code_values", ",".join(sorted(raw["승인거래코드"].dropna().astype(str).unique())[:40])),
        ("seq_increases_within_card_date", seq_order_check(common, extra)),
        ("codebook_verified", str(load_codebook().get("verified"))),
    ]
    quality = pd.DataFrame(rows, columns=["check", "value"])
    labels = (common.assign(fraud_type=common["fraud_type"].fillna("(없음)"))
              .groupby(["is_fraud", "fraud_type"], dropna=False).size().rename("rows").reset_index())
    return quality, labels


def seq_order_check(common: pd.DataFrame, extra: pd.DataFrame) -> str:
    """같은 카드·같은 날에서 승인SEQ가 시간대 순서와 함께 증가하는 비율. 1.0에 가까워야 정렬 키로 쓸 수 있다."""
    d = pd.DataFrame({"card": common["card_id"], "date": common["transaction_date"],
                      "hour": common["hour_bucket"], "seq": extra["seq"].values}).dropna()
    if d.empty:
        return "n/a"
    d = d.sort_values(["card", "date", "hour", "seq"])
    same = (d["card"] == d["card"].shift()) & (d["date"] == d["date"].shift())
    ok = (d["seq"] > d["seq"].shift())[same]
    return f"{ok.mean():.4f} ({int(same.sum())} pairs)" if same.any() else "n/a"


def assign_segments(common: pd.DataFrame) -> pd.Series:
    """분기 단위 시간순 분할: 2021Q1~2023Q2 학습 / 2023Q3~Q4 검증 / 2024 테스트 (설계서 F2)."""
    d = pd.to_datetime(common["transaction_date"])
    return pd.Series(np.select([d < "2023-07-01", d < "2024-01-01"], ["train", "valid"], default="test"),
                     index=common.index, name="segment")


def main():
    raw = read_quarters()
    if raw.empty:
        print(f"[B안] 원본 없음: {RAW_DIR} 에 AI-Hub 카드거래 CSV를 넣으면 변환합니다. 지금은 건너뜁니다.")
        return
    common, extra = to_common(raw)
    common["segment"] = assign_segments(common).values
    os.makedirs(ROOT / "outputs/v2", exist_ok=True)
    quality, labels = quality_report(raw, common, extra)
    quality.to_csv(ROOT / "outputs/v2/kr_quality.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(ROOT / "outputs/v2/kr_label_types.csv", index=False, encoding="utf-8-sig")
    print(quality.to_string(index=False))
    problems = validate(common.dropna(subset=["transaction_date", "amount"]))
    if problems:
        print("스키마 경고:", problems)
    common.to_parquet(OUT_DIR / "transactions.parquet", index=False)
    extra.to_parquet(OUT_DIR / "extra.parquet", index=False)
    print(f"saved {len(common):,} rows")


if __name__ == "__main__":
    main()
