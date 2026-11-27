"""개인 거래 위험도 분석 (설계서 4.1절 F8, 6.3절 ③④, 12장).

점수 산출 규칙
- 시각이 분 단위 이상인 '승인' 거래만 개인 점수용 LightGBM(models/v2/lightgbm_personal.joblib)으로 점수를 낸다.
  [설계 변경 2026-11-28] v1 모델 대신 hour_rarity를 뺀 모델을 쓴다. Sparkov 학습 구간의 hour_rarity는 0.83~1.0에만
  있어, 같은 시간대에 반복 결제하는 실제 개인(hour_rarity≈0)은 모델이 본 적 없는 값이 되고 평범한 결제가 '주의'로
  나왔다(브라우저 점검). 시간대 정보는 규칙 기반 확인 항목으로만 보여준다.
- 날짜·시간대만 있는 거래, 취소 거래는 점수를 보류하고 규칙 기반 확인 항목만 보여준다.
- 카드별 과거 거래가 5건 미만이면 금액 기준이 학습 구간 전체 중앙값(USD)으로 대체되므로,
  통화가 USD가 아니면 점수를 보류한다 (12장 기록).
- 이력은 (카드 별칭, 통화)별로 계산한다. 원화 결제와 달러 결제의 금액을 서로 비교하지 않기 위해서다.
- 위험 수준: 개인 점수용 모델의 5:1 임계값 이상 '높음', 100:1 임계값 이상 '주의', 그 아래 '낮음'
  (models/v2/personal_model.json, 미국 검증 구간에서 선택).
- 개인 거래에는 정답 라벨이 없으므로 정확도를 계산하지 않는다. 점수는 '추가 확인 필요' 정도를 뜻한다.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from .features import ALL_FEATURES as V1, build_features

MODEL_PATH = "models/LightGBM.joblib"  # v1 (미국 데이터 분석 화면용)
PERSONAL_MODEL_PATH = "models/v2/lightgbm_personal.joblib"
PERSONAL_META = "models/v2/personal_model.json"
PERSONAL_FEATURES = [c for c in V1 if c != "hour_rarity"]
STATS_PATH = "models/global_stats.json"
THRESH_PATH = "models/thresholds.json"
PLATT_PATH = "models/v2/platt_lightgbm_v1.json"
RANGES_PATH = "models/v2/train_ranges_v1.json"
MODEL_VERSION = "us_sparkov_lightgbm_personal_nohour_w10"
NOTICE = "위험도가 높다는 것은 사기 확정이 아니라 추가 확인이 필요하다는 뜻입니다."
LIMITATION = ("모델은 미국 모의 거래(Sparkov)로 학습했습니다. 실제 국내 카드 사용 패턴에 맞는다는 검증은 없으므로 "
              "참고용으로만 보세요.")
FEATURE_KO = {
    "amt_ratio": "평소 대비 결제 금액",
    "count_5min": "직전 5분 결제 건수",
    "count_15min": "직전 15분 결제 건수",
    "count_60min": "직전 60분 결제 건수",
    "hour_rarity": "평소 쓰지 않는 시간대",
    "insufficient_history": "카드 이력 부족",
}
OUT_OF_RANGE_WARN = 0.30


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def thresholds() -> tuple[float, float]:
    t = _load_json(THRESH_PATH)
    return float(t["LightGBM|5"]), float(t["LightGBM|100"])


def personal_thresholds() -> tuple[float, float]:
    t = _load_json(PERSONAL_META)["thresholds"]
    return float(t["5"]), float(t["100"])


def risk_level(score: float, t_high: float, t_watch: float) -> str:
    if score is None or np.isnan(score):
        return "판단 보류"
    return "높음" if score >= t_high else ("주의" if score >= t_watch else "낮음")


def calibrate(s: np.ndarray) -> np.ndarray:
    """검증 구간 앞 절반으로 학습한 Platt 계수로 표시용 확률을 만든다. 계수 파일이 없으면 NaN."""
    if not os.path.exists(PERSONAL_META):
        return np.full(len(s), np.nan)
    c = _load_json(PERSONAL_META)["platt"]
    s = np.clip(s, 1e-7, 1 - 1e-7)
    return 1 / (1 + np.exp(-(c["a"] * np.log(s / (1 - s)) + c["b"])))


# ---------------------------------------------------------------- 특징
def _key(df: pd.DataFrame) -> pd.Series:
    return df["card_id"].astype(str) + "|" + df["currency"].astype(str)


def personal_features(df: pd.DataFrame) -> pd.DataFrame:
    """분 단위 시각이 있는 승인 거래로 기존 v1 특징을 계산한다 (이력도 그런 거래만)."""
    ok = df["time_precision"].isin(["second", "minute"]) & df["transaction_time"].notna() \
        & df["transaction_status"].fillna("approved").eq("approved")
    d = df[ok]
    if d.empty:
        return pd.DataFrame(columns=["transaction_id"] + V1)
    ts = pd.to_datetime(d["transaction_date"].astype(str) + " " + d["transaction_time"].astype(str), format="mixed")
    raw = pd.DataFrame({"trans_num": d["transaction_id"].values, "cc_num": _key(d).values,
                        "amt": d["amount"].astype(float).values, "ts": ts.values})
    raw = raw.sort_values("ts", kind="mergesort").reset_index(drop=True)
    raw["_order"] = np.arange(len(raw))
    f = build_features(raw, _load_json(STATS_PATH))
    return f[["trans_num"] + V1].rename(columns={"trans_num": "transaction_id"})


def card_profile(df: pd.DataFrame) -> pd.DataFrame:
    """(카드, 통화)별 평소 패턴: 금액 중앙값·사분위, 자주 쓰는 시간대 3개, 일평균 건수."""
    ap = df[df["transaction_status"].fillna("approved").eq("approved")].copy()
    ap["key"] = _key(ap)
    rows = []
    for k, g in ap.groupby("key"):
        hours = g["hour_bucket"].dropna().astype(int)
        top = hours.value_counts().head(3).index.tolist()
        days = g["transaction_date"].nunique()
        rows.append({"key": k, "card_id": g["card_id"].iloc[0], "currency": g["currency"].iloc[0], "n": len(g),
                     "median_amount": g["amount"].median(), "q1": g["amount"].quantile(0.25),
                     "q3": g["amount"].quantile(0.75), "top_hours": top, "daily_avg": len(g) / max(days, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 규칙 기반
def rule_flags(df: pd.DataFrame) -> pd.Series:
    """모델과 별개인 확인 권장 항목 (설계서 4.1). 과거 거래만 기준으로 삼는다(같은 날 건수는 '같은 날' 전체)."""
    d = df.copy()
    d["key"] = _key(d)
    d["_date"] = pd.to_datetime(d["transaction_date"])
    order = d.sort_values(["key", "_date", "transaction_time"], na_position="first").index
    flags = {i: [] for i in d.index}
    for _, g in d.loc[order].groupby("key", sort=False):
        approved = g["transaction_status"].fillna("approved").eq("approved")
        for pos, (i, r) in enumerate(g.iterrows()):
            prev = g.iloc[:pos]
            prev = prev[prev["_date"] < r["_date"]] if pd.isna(r["transaction_time"]) else prev
            prev = prev[approved.loc[prev.index]]
            if len(prev) >= 5 and r["amount"] >= 5 * prev["amount"].median():
                flags[i].append(f"평소 결제액 중앙값({prev['amount'].median():,.0f} {r['currency']})의 "
                                f"{r['amount'] / prev['amount'].median():.1f}배")
            if pd.notna(r["merchant_name"]):
                same = g[(g["_date"] == r["_date"]) & (g["merchant_name"] == r["merchant_name"])]
                if len(same) >= 3:
                    flags[i].append(f"같은 날 같은 가맹점 결제 {len(same)}회")
            past_days = prev[prev["_date"] < r["_date"]]
            if past_days["_date"].nunique() >= 5:
                avg = len(past_days) / past_days["_date"].nunique()
                today = int((g["_date"] == r["_date"]).sum())
                if today >= 3 * avg and today >= 3:
                    flags[i].append(f"하루 결제 {today}건 (평소 일평균 {avg:.1f}건의 {today / avg:.1f}배)")
    return pd.Series(flags).reindex(df.index).map(lambda x: x or [])


# ---------------------------------------------------------------- 점수·설명
def _reason(feat: str, row: pd.Series) -> str:
    v = row[feat]
    if feat == "amt_ratio":
        return f"평소 결제액 중앙값의 {v:.1f}배" if v >= 1 else f"평소 결제액 중앙값의 {v:.2f}배(평소보다 적음)"
    if feat.startswith("count_"):
        return f"{FEATURE_KO[feat].replace(' 결제 건수', '')} 안에 다른 결제 {int(v)}건"
    if feat == "hour_rarity":
        return f"이 카드의 과거 결제 중 {int(row['hour_bucket']):02d}시대 비율 {100 * (1 - v):.0f}%"
    if feat == "insufficient_history":
        return "이 카드의 과거 거래가 5건 미만이라 평소 패턴 비교가 약함" if v else "이력 충분"
    return f"{FEATURE_KO.get(feat, feat)} {v}"


def score(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """df: 개인 거래(공통 스키마). → (거래별 결과, 데이터 전체 점검 결과)."""
    df = df.reset_index(drop=True)
    t_high, t_watch = personal_thresholds()
    feats = personal_features(df)
    res = df[["transaction_id", "card_id", "transaction_date", "transaction_time", "hour_bucket", "amount",
              "currency", "merchant_name", "transaction_status", "time_precision"]].merge(feats, on="transaction_id", how="left")
    res["rules"] = rule_flags(df).values

    hold = pd.Series("", index=res.index)
    hold[~res["time_precision"].isin(["second", "minute"])] = "결제 시각(분 단위)이 없어 시간대·반복 결제 특징을 계산할 수 없음"
    hold[res["transaction_status"].eq("cancelled")] = "취소 거래는 점수를 내지 않음"
    low_hist_non_usd = res["insufficient_history"].eq(1) & res["currency"].ne("USD") & hold.eq("")
    hold[low_hist_non_usd] = "카드 이력 5건 미만 + USD 외 통화: 금액 기준(학습 데이터 중앙값, USD)을 쓸 수 없음"
    scorable = hold.eq("") & res[PERSONAL_FEATURES].notna().all(axis=1)
    F = PERSONAL_FEATURES

    res["raw_score"] = np.nan
    res["fraud_probability"] = np.nan
    res["top_factors"] = [[] for _ in range(len(res))]
    if scorable.any():
        model = joblib.load(PERSONAL_MODEL_PATH)
        X = res.loc[scorable, F].astype(float)
        s = model.predict_proba(X)[:, 1]
        contrib = model.booster_.predict(X, pred_contrib=True)[:, :len(F)]  # 마지막 열은 기준값(bias)
        res.loc[scorable, "raw_score"] = s
        res.loc[scorable, "fraud_probability"] = calibrate(s)
        tops = []
        for row_i, c in zip(res.index[scorable], contrib):
            order = np.argsort(-c)[:3]
            tops.append([{"feature": FEATURE_KO[F[j]], "contribution": float(c[j]),
                          "reason": _reason(F[j], res.loc[row_i])} for j in order])
        res.loc[scorable, "top_factors"] = pd.Series(tops, index=res.index[scorable])
    res["risk_level"] = [risk_level(v, t_high, t_watch) for v in res["raw_score"]]
    res["note"] = hold.where(hold.ne(""), None)
    res.loc[scorable & res["insufficient_history"].eq(1), "note"] = "점수 산출(이력 5건 미만 — 신뢰도 낮음)"
    res["model_version"] = np.where(scorable, MODEL_VERSION, None)
    # 시간대 정보는 모델 대신 규칙으로: 이력 5건 이상인데 이 시간대 결제가 처음이면 확인 항목에 추가
    new_hour = res["hour_rarity"].eq(1.0) & res["insufficient_history"].eq(0) & res["hour_bucket"].notna()
    for i in res.index[new_hour]:
        res.at[i, "rules"] = list(res.at[i, "rules"]) + [f"이 카드가 지금까지 {int(res.at[i, 'hour_bucket']):02d}시대에 결제한 적 없음"]

    # 직전 거래와의 차이 (같은 카드·통화, 더 이른 분 단위 거래)
    ts = pd.to_datetime(res["transaction_date"].astype(str) + " " + res["transaction_time"].fillna("").astype(str),
                        format="mixed", errors="coerce")
    tmp = pd.DataFrame({"key": _key(res), "ts": ts, "amount": res["amount"]}).dropna(subset=["ts"])
    tmp = tmp.sort_values(["key", "ts"], kind="mergesort")
    same = tmp["key"].eq(tmp["key"].shift()) & tmp["ts"].ne(tmp["ts"].shift())
    res["prev_gap_min"] = ((tmp["ts"] - tmp["ts"].shift()).dt.total_seconds() / 60).where(same)
    res["prev_amount_ratio"] = (tmp["amount"] / tmp["amount"].shift()).where(same)

    check = distribution_check(res[scorable])
    return res, check


def distribution_check(scored: pd.DataFrame) -> dict:
    """특징별로 학습 구간 1~99 백분위를 벗어난 비율. 30%를 넘는 특징이 있으면 경고 (설계서 6.3 ④)."""
    if scored.empty or not os.path.exists(PERSONAL_META):
        return {"checked": 0, "out_of_range": {}, "warn": False}
    ranges = _load_json(PERSONAL_META)["train_ranges_1_99"]
    share = {c: float(((scored[c] < lo) | (scored[c] > hi)).mean()) for c, (lo, hi) in ranges.items() if c in scored}
    return {"checked": int(len(scored)), "out_of_range": share,
            "warn": any(v > OUT_OF_RANGE_WARN for v in share.values()),
            "warn_features": [FEATURE_KO.get(c, c) for c, v in share.items() if v > OUT_OF_RANGE_WARN]}


def to_store_rows(df: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """개인 저장소에 다시 쓸 행: 원래 거래 + fraud_probability/raw_score/risk_level/model_version."""
    out = df.drop(columns=["fraud_probability", "raw_score", "risk_level", "model_version"], errors="ignore") \
        .merge(res[["transaction_id", "fraud_probability", "raw_score", "risk_level", "model_version"]],
               on="transaction_id", how="left")
    return out
