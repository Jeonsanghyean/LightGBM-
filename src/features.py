"""데이터 적재·점검·시간순 분할과 세 거래 패턴 특징 추출.

모든 개인 특징은 '거래 직전까지'의 같은 카드 이력만 사용한다.
같은 카드에서 같은 시각에 발생한 거래끼리는 서로의 이력으로 보지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS = ["trans_date_trans_time", "cc_num", "amt", "trans_num", "is_fraud"]
FEATURE_GROUPS = {
    "amount": ["amt_ratio"],
    "repeat": ["count_5min", "count_15min", "count_60min"],
    "hour": ["hour_rarity"],
}
COMMON = ["insufficient_history"]
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g] + COMMON
MIN_HISTORY = 5
WINDOWS = {"count_5min": 5, "count_15min": 15, "count_60min": 60}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c != "Unnamed: 0")
    return df


def check_raw(df: pd.DataFrame, name: str) -> dict:
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: 필수 열 누락 {missing_cols}")
    return {
        "file": name,
        "rows": len(df),
        "fraud": int(df["is_fraud"].sum()),
        "missing_required": int(df[REQUIRED_COLS].isna().sum().sum()),
        "duplicate_trans_num": int(df["trans_num"].duplicated().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "start": str(df["trans_date_trans_time"].min()),
        "end": str(df["trans_date_trans_time"].max()),
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    # 원래 순서를 보조키로 두어 정렬을 결정적으로 만든다
    df["_order"] = np.arange(len(df))
    return df.sort_values(["ts", "_order"], kind="mergesort").reset_index(drop=True)


def time_split(train: pd.DataFrame, fracs=(0.8, 0.1)) -> pd.Series:
    """시간순 약 80/10/10 분할. 같은 시각의 거래는 경계를 넘지 않도록 뒤로 민다."""
    n = len(train)
    ts = train["ts"].to_numpy()
    bounds = []
    for cum in (fracs[0], fracs[0] + fracs[1]):
        b = int(n * cum)
        while 0 < b < n and ts[b] == ts[b - 1]:
            b += 1
        bounds.append(b)
    seg = np.empty(n, dtype=object)
    seg[: bounds[0]] = "train"
    seg[bounds[0] : bounds[1]] = "valid"
    seg[bounds[1] :] = "internal_test"
    return pd.Series(seg, index=train.index, name="segment")


def global_stats(train_part: pd.DataFrame) -> dict:
    """학습 구간 종료 시점에 고정하는 전체 카드 기본 통계."""
    hours = train_part["ts"].dt.hour
    share = hours.value_counts(normalize=True).reindex(range(24), fill_value=0.0)
    return {"median_amt": float(train_part["amt"].median()), "hour_share": share.tolist()}


def build_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """df: prepare()를 거친 시간순 데이터(여러 원본을 이어 붙여도 됨)."""
    d = df.sort_values(["cc_num", "ts", "_order"], kind="mergesort").copy()
    d["hour"] = d["ts"].dt.hour
    tie_key = [d["cc_num"], d["ts"]]

    def first_in_tie(s: pd.Series) -> pd.Series:
        # 같은 카드·같은 시각 묶음에서는 첫 거래 기준의 '직전' 값을 공유한다
        return s.groupby(tie_key).transform("first")

    g = d.groupby("cc_num", sort=False)
    n_prev = first_in_tie(g.cumcount())
    past_median = first_in_tie(
        g["amt"].transform(lambda s: s.expanding().median().shift(1))
    )
    hour_prev = first_in_tie(d.groupby(["cc_num", "hour"], sort=False).cumcount())

    insufficient = n_prev < MIN_HISTORY
    base_median = np.where(insufficient, stats["median_amt"], past_median)
    d["amt_ratio"] = d["amt"] / np.maximum(base_median, 0.01)

    hour_share_global = np.asarray(stats["hour_share"])[d["hour"].to_numpy()]
    personal_share = hour_prev / n_prev.where(n_prev > 0, 1)
    d["hour_rarity"] = 1.0 - np.where(insufficient, hour_share_global, personal_share)

    # 직전 w분 [t-w, t) 동일 카드 거래 건수 (현재·동시각 거래 제외)
    t_s = d["ts"].to_numpy().astype("datetime64[s]").astype("int64")  # 초 단위로 명시
    cards = d["cc_num"].to_numpy()
    starts = np.flatnonzero(np.r_[True, cards[1:] != cards[:-1]])
    ends = np.r_[starts[1:], len(d)]
    for col, minutes in WINDOWS.items():
        out = np.empty(len(d), dtype=np.int32)
        w = np.int64(minutes * 60)
        for s, e in zip(starts, ends):
            tt = t_s[s:e]
            first_same = np.searchsorted(tt, tt, side="left")
            lo = np.searchsorted(tt, tt - w, side="left")
            out[s:e] = first_same - lo
        d[col] = out

    d["insufficient_history"] = insufficient.astype(np.int8)
    d["n_prev"] = n_prev.astype(np.int32)
    return d.sort_values(["ts", "_order"], kind="mergesort").reset_index(drop=True)
