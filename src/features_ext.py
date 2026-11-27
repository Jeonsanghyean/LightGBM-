"""v2 신규 특징 (설계서 6.1 표 8). 입력은 공통 스키마(출처 하나), 출력은 transaction_id별 특징표.

모든 특징은 '같은 카드에서 해당 거래보다 엄격히 이전'의 거래만 본다.
- 시각 정밀도가 hour인 거래(한국)는 시간대 시작 시각을 거래 시각으로 쓰고, 같은 시간대 거래끼리는 동시로 본다.
- 시각이 없는(day) 거래는 시간 기반 특징이 NaN이고, 다른 거래의 이력으로도 세지 않는다.
- 계산할 수 없는 값은 0이 아니라 NaN.

[설계 변경 2026-11-27] 표 8의 count_same_hour → count_prev_hour_bucket, count_same_day → count_same_day_prior
  이유: 한국 데이터는 같은 시간대 안의 순서를 알 수 없어, '같은 시간대 건수'를 세면 그 시간대의 이후 거래(미래)가 섞인다.
  미국·개인 데이터도 국가 간 비교(E3)를 위해 같은 정의(시간대 단위)로 계산한다.

구현: (그룹 코드 × BIG + 시각) 값을 한 번 정렬한 뒤 searchsorted로 세므로 카드별 반복문이 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIG = 10**10  # 그룹 사이 간격(초). 데이터 기간(수 년 ≈ 1e8초)보다 충분히 커야 한다.
ALL_PAST = -1e9  # '이전 전체' 창의 하한. -inf를 쓰면 앞 그룹까지 세어지므로 BIG보다 작은 유한값을 쓴다.
EARTH_KM = 6371.0088
FAR_KM, FAR_HOURS, MAX_KMH = 500.0, 6.0, 900.0
NEW_FEATURES = ["count_prev_hour_bucket", "count_same_day_prior", "same_merchant_24h",
                "dist_prev_km", "gap_prev_min", "speed_kmh", "far_fast",
                "region_novelty", "region_changed", "is_overseas"]


def event_seconds(df: pd.DataFrame) -> pd.Series:
    """거래 시각(초). hour 정밀도는 시간대 시작, day 정밀도는 NaN."""
    date = pd.to_datetime(df["transaction_date"], errors="coerce")
    sec = pd.Series(np.nan, index=df.index)
    exact = df["time_precision"].isin(["second", "minute"]) & df["transaction_time"].notna()
    if exact.any():
        stamp = pd.to_datetime(df.loc[exact, "transaction_date"].astype(str) + " "
                               + df.loc[exact, "transaction_time"].astype(str), format="mixed", errors="coerce")
        sec[exact] = stamp.astype("datetime64[s]").astype("int64").where(stamp.notna())
    hourly = df["time_precision"].eq("hour") & df["hour_bucket"].notna()
    if hourly.any():
        sec[hourly] = (date[hourly] + pd.to_timedelta(df.loc[hourly, "hour_bucket"].astype(int), unit="h")
                       ).astype("datetime64[s]").astype("int64")
    return sec


def _window_count(group: np.ndarray, t: np.ndarray, lo: np.ndarray | float, hi: np.ndarray | float = 0.0) -> np.ndarray:
    """같은 group에서 시각이 [t+lo, t+hi) 안인 다른 거래 수. group/t가 NaN인 행은 NaN."""
    out = np.full(len(t), np.nan)
    ok = ~np.isnan(t) & (group >= 0)
    if not ok.any():
        return out
    v = group[ok] * BIG + t[ok]
    sv = np.sort(v)
    lo_ = lo[ok] if isinstance(lo, np.ndarray) else lo
    hi_ = hi[ok] if isinstance(hi, np.ndarray) else hi
    out[ok] = np.searchsorted(sv, v + hi_, "left") - np.searchsorted(sv, v + lo_, "left")
    return out


def _prev_index(group: np.ndarray, t: np.ndarray) -> np.ndarray:
    """같은 group에서 엄격히 이전 시각의 직전 거래 행 번호 (없으면 -1). 동시 거래는 건너뛴다."""
    prev = np.full(len(t), -1)
    ok = np.flatnonzero(~np.isnan(t) & (group >= 0))
    if not len(ok):
        return prev
    v = group[ok] * BIG + t[ok]
    order = np.argsort(v, kind="mergesort")
    sv, rows = v[order], ok[order]
    p = np.searchsorted(sv, sv, "left") - 1
    same_group = (p >= 0) & (group[rows[np.maximum(p, 0)]] == group[rows])
    res = np.where(same_group, rows[np.maximum(p, 0)], -1)
    prev[rows] = res
    return prev


def _codes(*cols: pd.Series) -> np.ndarray:
    """여러 열의 조합 → 정수 그룹 코드. 하나라도 NULL이면 -1."""
    frame = pd.concat([c.rename(f"c{i}").reset_index(drop=True) for i, c in enumerate(cols)], axis=1)
    codes = frame.groupby(list(frame.columns), dropna=False, sort=False).ngroup().to_numpy()
    return np.where(frame.isna().any(axis=1).to_numpy(), -1, codes)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def add_extended(df: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """df: 공통 스키마. extra: (한국) transaction_id, region_code, is_overseas. → transaction_id + NEW_FEATURES."""
    df = df.reset_index(drop=True)
    t = event_seconds(df).to_numpy(dtype=float)
    # 기준점을 자정에 맞춰야 floor(t/3600)이 실제 시간대 경계와 일치한다
    base = np.floor(np.nanmin(t) / 86400) * 86400 if np.isfinite(t).any() else 0.0
    t = t - base
    bucket = np.floor(t / 3600) * 3600
    day_start = np.floor((t + base) / 86400) * 86400 - base  # 날짜 경계는 절대 시각 기준(UTC 무관, 원본 날짜 그대로)
    card = _codes(df["card_id"])
    out = pd.DataFrame({"transaction_id": df["transaction_id"]})

    # B 반복: 시간대 단위 (국가 공통 정의)
    out["count_prev_hour_bucket"] = _window_count(card, bucket, -3600.0, 0.0)
    out["count_same_day_prior"] = _window_count(card, bucket, day_start - bucket, 0.0)
    # 같은 가맹점 반복: 직전 24시간 (hour 정밀도는 직전 24개 시간대)
    merchant = df["merchant_id"].where(df["merchant_id"].notna(),
                                       df["merchant_name"].astype("string").str.replace(r"\s", "", regex=True).str.lower())
    out["same_merchant_24h"] = _window_count(_codes(df["card_id"], merchant), t, -86400.0, 0.0)

    # D 위치: 좌표가 있고 시각이 분 단위 이상인 거래끼리만
    has_xy = (df["latitude"].notna() & df["longitude"].notna()
              & df["location_precision"].isin(["exact", "synthetic_point"])
              & df["time_precision"].isin(["second", "minute"])).to_numpy()
    tx = np.where(has_xy, t, np.nan)
    prev = _prev_index(card, tx)
    hp = prev >= 0
    lat, lon = df["latitude"].to_numpy(float), df["longitude"].to_numpy(float)
    dist = np.full(len(df), np.nan)
    gap = np.full(len(df), np.nan)
    dist[hp] = haversine_km(lat[prev[hp]], lon[prev[hp]], lat[hp], lon[hp])
    gap[hp] = (tx[hp] - tx[prev[hp]]) / 60.0
    online = df["transaction_type"].eq("online").fillna(False).to_numpy()
    either_online = np.zeros(len(df), bool)
    either_online[hp] = online[hp] | online[prev[hp]]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(hp & (gap >= 1.0) & ~either_online, dist / (gap / 60.0), np.nan)
    far = np.where(np.isnan(dist), np.nan,
                   ((dist >= FAR_KM) & (gap <= FAR_HOURS * 60)) | (np.nan_to_num(speed) > MAX_KMH))
    out["dist_prev_km"], out["gap_prev_min"], out["speed_kmh"], out["far_fast"] = dist, gap, speed, far

    # D 지역: 한국은 원본 시도 코드(의미 미확인이어도 '같은 코드인지'는 비교 가능), 개인은 주소에서 뽑은 시도
    region = df["region"].copy()
    overseas = pd.Series(np.nan, index=df.index)
    if extra is not None and len(extra):
        ex = df[["transaction_id"]].merge(extra, on="transaction_id", how="left")
        region = region.where(region.notna(), ex["region_code"].where(ex["is_overseas"].eq(0)))
        overseas = ex["is_overseas"].astype(float)
    tb = np.where(region.notna().to_numpy(), t, np.nan)  # hour 정밀도는 t가 이미 시간대 시작
    past_all = _window_count(card, tb, ALL_PAST, 0.0)
    past_same = _window_count(_codes(df["card_id"], region), tb, ALL_PAST, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["region_novelty"] = np.where(past_all > 0, 1.0 - past_same / past_all, np.nan)
    prev_r = _prev_index(card, tb)
    reg = region.astype("string").fillna("").to_numpy(dtype=object)  # pd.NA 비교 오류 방지 (NULL 행은 prev_r=-1)
    out["region_changed"] = np.where(prev_r >= 0, (reg != reg[np.maximum(prev_r, 0)]).astype(float), np.nan)
    out["is_overseas"] = overseas.to_numpy()
    return out
