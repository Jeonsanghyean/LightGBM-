"""3주차: 국가별 실험 E1~E6 (설계서 6.2 표 9).

실행: python -m src.experiments_country   (먼저 build_data → experiments → build_features_v2)
산출: outputs/v2/*.csv, models/v2/platt_lightgbm_v1.json, models/v2/train_ranges_v1.json
      한국 특징 파일(data/research/kr/features_v2.parquet)이 있으면 E2·E3·한국 E4·E5·E6도 실행한다.
      → e2_kr_performance.csv, e3_cross_country.csv, e6_kr_type_recall.csv, 기존 e4·e5·비용·보정 파일에 KR 행 추가
      한국 데이터가 없으면 미국 결과 파일은 이전과 같은 내용으로 만들어진다.

기존 설정 유지 (src/experiments.py의 함수를 그대로 가져다 씀, v1 파일은 수정하지 않음)
- 하이퍼파라미터·양성 가중치 10·비용 비율 5:1, 20:1, 100:1
- 시간순 분할(train / valid / internal_test / test)
- 검증 구간 앞쪽 절반으로 Platt 보정, 임계값은 검증 구간 전체에서 선택해 테스트 구간에 고정 적용
- 모든 AP 옆에 기준선(해당 구간 이상 비율)을 기록

국가별로 학습·평가를 분리한다. 미국·한국 라벨을 한 학습 데이터로 합치는 코드 경로는 없다.
Sparkov 좌표는 무작위 합성값이므로 D군(위치) 효과는 '시뮬레이터 조건의 결과'로만 해석한다.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline

from .experiments import COST_RATIOS, DEFAULT_T, POS_WEIGHT, at_threshold, best_threshold, make_model, platt
from .features import ALL_FEATURES as V1

OUT = "outputs/v2"
MODELS_V2 = "models/v2"
US_FEATURES = "data/research/us/features_v2.parquet"
KR_FEATURES = "data/research/kr/features_v2.parquet"
SEGMENTS = ("train", "valid", "internal_test", "test")
MODEL_KINDS = ("LightGBM", "XGBoost", "LogisticRegression")

# 특징군 (설계서 표 8). gap_prev_min은 좌표가 있는 거래끼리만 계산되므로 D군에 둔다.
GROUPS_US = {
    "A": ["amt_ratio"],
    "B": ["count_5min", "count_15min", "count_60min", "count_prev_hour_bucket", "count_same_day_prior",
          "same_merchant_24h"],
    "C": ["hour_rarity"],
    "D": ["dist_prev_km", "gap_prev_min", "speed_kmh", "far_fast"],
}
GROUP_KO = {"A": "A 고액", "B": "B 반복", "C": "C 시간대", "D": "D 위치"}
COMMON = ["insufficient_history"]
EXT_US = [f for g in GROUPS_US.values() for f in g] + COMMON
NO_D_US = [f for f in EXT_US if f not in GROUPS_US["D"]]
V1_PLUS_D = V1 + GROUPS_US["D"]
CROSS_FEATURES = ["amt_ratio", "count_prev_hour_bucket", "count_same_day_prior", "hour_rarity", "insufficient_history"]
NOT_MEASURED = "미측정(한국 데이터 미확보)"

# [v2 추가 2026-11-28] 한국 특징군 (설계서 표 8). 5·15·60분 건수는 시각이 없어 NaN이므로 제외.
# D군은 좌표가 없어 지역 단위 특징으로 대신한다.
GROUPS_KR = {
    "A": ["amt_ratio"],
    "B": ["count_prev_hour_bucket", "count_same_day_prior", "same_merchant_24h"],
    "C": ["hour_rarity"],
    "D": ["region_novelty", "region_changed", "is_overseas"],
}
EXT_KR = [f for g in GROUPS_KR.values() for f in g] + COMMON
NO_D_KR = [f for f in EXT_KR if f not in GROUPS_KR["D"]]
V1_KR = ["amt_ratio", "hour_rarity", "insufficient_history"]  # v1 6개 중 한국에서 계산 가능한 것
V1_KR_PLUS_D = V1_KR + GROUPS_KR["D"]


# ---------------------------------------------------------------- 데이터
def load_us() -> dict[str, pd.DataFrame]:
    """features_v2를 v1 data/features.parquet의 행 순서로 맞춘다.

    LightGBM의 행 부분추출(subsample)은 행 순서에 따라 달라지므로, v1 6개 특징 설정이
    기존 결과(내부 비교 AP 0.4455)를 그대로 재현하려면 학습 행 순서가 같아야 한다.
    """
    order = pd.read_parquet("data/features.parquet", columns=["trans_num", "segment", "is_fraud"] + V1)
    f = pd.read_parquet(US_FEATURES).set_index("transaction_id")
    f = f.reindex(order["trans_num"].to_numpy())
    assert len(f) == len(order) and f.index.notna().all(), "features_v2와 features.parquet 행 불일치"
    assert (f["segment"].to_numpy() == order["segment"].to_numpy()).all(), "구간 불일치"
    # v1 특징은 기존 파일 값과 완전히 같아야 한다
    for c in V1:
        assert np.array_equal(f[c].to_numpy(float), order[c].to_numpy(float)), f"v1 특징 불일치: {c}"
    f = f.reset_index().rename(columns={"index": "transaction_id"})
    f["is_fraud"] = f["is_fraud"].astype(int)
    return {s: f[f["segment"] == s].reset_index(drop=True) for s in SEGMENTS}


def model_for(kind: str, cols: list[str], data: pd.DataFrame):
    """LR은 NaN을 다룰 수 없으므로, NaN이 있는 특징 조합에서만 학습 구간 중앙값 대체 + 결측 표시 열을 붙인다."""
    m = make_model(kind, POS_WEIGHT)
    if kind == "LogisticRegression" and data[cols].isna().any().any():
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), m)
    return m


def eval_segments(seg: dict[str, pd.DataFrame]) -> list[str]:
    """평가 구간. 미국은 valid·internal_test·test, 한국은 내부 비교 구간이 없어 valid·test."""
    return [s for s in ("valid", "internal_test", "test") if s in seg]


def fit_scores(kind: str, cols: list[str], seg: dict[str, pd.DataFrame]):
    tr = seg["train"]
    m = model_for(kind, cols, tr).fit(tr[cols], tr["is_fraud"])
    return m, {s: m.predict_proba(seg[s][cols])[:, 1] for s in eval_segments(seg)}


def event_time(d: pd.DataFrame) -> pd.Series:
    """거래 시각. 미국은 초 단위 시각, 한국은 날짜 + 1시간대 (같은 시간대 안의 순서는 알 수 없음)."""
    if "transaction_time" in d.columns:
        return pd.to_datetime(d["transaction_date"] + " " + d["transaction_time"].fillna("00:00:00"))
    return pd.to_datetime(d["transaction_date"]) + pd.to_timedelta(d["hour_bucket"].astype(float).fillna(0), unit="h")


def valid_halves(va: pd.DataFrame) -> int:
    """검증 구간 시간순 앞 절반 경계 (같은 시각은 뒤로 밀어 경계를 넘지 않게). experiments.py와 같은 규칙."""
    ts = event_time(va)
    half = len(va) // 2
    while half < len(va) and ts.iloc[half] == ts.iloc[half - 1]:
        half += 1
    return half


# ---------------------------------------------------------------- 지표
def ap_row(seg, scores) -> dict:
    row = {}
    for s in eval_segments(seg):
        y = seg[s]["is_fraud"]
        row[f"AP_{s}"] = average_precision_score(y, scores[s])
        row[f"baseline_{s}"] = float(y.mean())  # 무작위 순위의 기대 AP = 이상 비율
    return row


def threshold_rows(country, setting, kind, seg, scores) -> list[dict]:
    """검증 구간에서 비용 최소 임계값 선택 → 테스트 구간에 고정 적용. 기본 0.5와 비교."""
    rows = []
    yv, yt = seg["valid"]["is_fraud"], seg["test"]["is_fraud"]
    for r in COST_RATIOS:
        t, vcost = best_threshold(yv, scores["valid"], r)
        base = at_threshold(yt, scores["test"], DEFAULT_T, r)
        sel = at_threshold(yt, scores["test"], t, r)
        for policy, res in (("기본 0.5", base), ("검증 임계값", sel)):
            rows.append({"country": country, "setting": setting, "model": kind, "cost_ratio": f"{r}:1", "r": r,
                         "policy": policy, "valid_cost_selected": vcost if policy != "기본 0.5" else np.nan, **res,
                         "cost_change_pct_vs_default": (sel["total_cost"] - base["total_cost"]) / base["total_cost"] * 100
                         if policy != "기본 0.5" and base["total_cost"] else np.nan})  # 기본 비용 0이면 변화율 정의 안 됨
    return rows


def calibration_row(seg, s_valid, half) -> tuple[dict, tuple[float, float]]:
    """검증 구간 앞 절반으로 Platt 보정 → 뒤 절반에서 Brier·LogLoss 비교. (계수 a, b) 반환: p = σ(a·logit(s)+b)."""
    va = seg["valid"]
    ya, yb = va["is_fraud"].iloc[:half], va["is_fraud"].iloc[half:]
    s_a, s_b = s_valid[:half], s_valid[half:]
    cal_b = platt(s_a, ya, s_b)
    from sklearn.linear_model import LogisticRegression
    z = lambda s: np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z(s_a).reshape(-1, 1), ya)
    coef = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
    return {
        "Brier_before": brier_score_loss(yb, s_b), "Brier_after_Platt": brier_score_loss(yb, cal_b),
        "LogLoss_before": log_loss(yb, s_b, labels=[0, 1]), "LogLoss_after_Platt": log_loss(yb, cal_b, labels=[0, 1]),
    }, coef


def not_measured_rows(experiment: str, desc: str) -> dict:
    return {"experiment": experiment, "country": "KR", "description": desc, "status": NOT_MEASURED}


# ---------------------------------------------------------------- 한국 (데이터 확보 후)
def load_kr(path: str = KR_FEATURES) -> dict[str, pd.DataFrame]:
    """한국 특징 파일 → 구간별 표. 날짜·시간대 순으로 안정 정렬한다 (같은 시간대 안은 원본 순서 유지).
    라벨을 해석하지 못한 행(is_fraud 결측)은 학습·평가에서 빼고 건수를 남긴다."""
    f = pd.read_parquet(path)
    n_unlabeled = int(f["is_fraud"].isna().sum())
    if n_unlabeled:
        print(f"[한국] 라벨 해석 불가 {n_unlabeled:,}건 제외")
    f = f[f["is_fraud"].notna()].copy()
    f["is_fraud"] = f["is_fraud"].astype(int)
    f = f.assign(_t=event_time(f)).sort_values("_t", kind="mergesort").drop(columns="_t")
    missing = {"train", "valid", "test"} - set(f["segment"])
    assert not missing, f"한국 구간 없음: {sorted(missing)}"
    return {s: f[f["segment"] == s].reset_index(drop=True) for s in ("train", "valid", "test")}


def run_kr(seg: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """E2 한국 단독(3종) · 한국 E4(지역 특징 추가 전후) · 한국 E5(특징군 제거) · E6(이상 유형별 Recall).
    미국 실험과 같은 설정(하이퍼파라미터·양성 가중치·비용 비율·Platt·검증 임계값 고정 적용)."""
    half = valid_halves(seg["valid"])
    perf, thr, calib, lgb = [], [], [], {}
    for setting, cols in (("v1 사용 가능 3개", V1_KR), ("확장 전체", EXT_KR)):
        for kind in MODEL_KINDS:
            _, sc = fit_scores(kind, cols, seg)
            yt = seg["test"]["is_fraud"]
            perf.append({"experiment": "E2", "country": "KR", "setting": setting, "model": kind,
                         "n_features": len(cols), "features": ", ".join(cols), **ap_row(seg, sc),
                         **{f"{k}_test@0.5": v for k, v in at_threshold(yt, sc["test"], DEFAULT_T).items()
                            if k in ("Precision", "Recall", "F1", "FP", "FN")}})
            thr += threshold_rows("KR", setting, kind, seg, sc)
            print(f"E2 {setting} {kind}: AP_test={perf[-1]['AP_test']:.4f} (기준선 {perf[-1]['baseline_test']:.4f})")
            if kind == "LightGBM":
                lgb[setting] = sc
                c, coef = calibration_row(seg, sc["valid"], half)
                calib.append({"country": "KR", "setting": setting, "model": kind, **c, "platt_a": coef[0], "platt_b": coef[1]})

    loc = []
    cached = {"v1 사용 가능 3개": lgb["v1 사용 가능 3개"], "확장 전체 (D 포함)": lgb["확장 전체"]}  # E2에서 학습한 점수 재사용
    for setting, cols, with_d in (("v1 사용 가능 3개", V1_KR, False), ("v1 3개 + D 지역", V1_KR_PLUS_D, True),
                                  ("확장 (D 제외)", NO_D_KR, False), ("확장 전체 (D 포함)", EXT_KR, True)):
        sc = cached[setting] if setting in cached else fit_scores("LightGBM", cols, seg)[1]
        loc.append({"experiment": "E4", "country": "KR", "setting": setting, "model": "LightGBM",
                    "with_location": with_d, "n_features": len(cols), **ap_row(seg, sc)})
    loc = pd.DataFrame(loc)
    for base, plus in (("v1 사용 가능 3개", "v1 3개 + D 지역"), ("확장 (D 제외)", "확장 전체 (D 포함)")):
        b = loc.loc[loc.setting == base, "AP_test"].iloc[0]
        loc.loc[loc.setting == plus, "AP_gain_vs_without_D_test"] = loc.loc[loc.setting == plus, "AP_test"] - b

    full = ap_row(seg, lgb["확장 전체"])
    abl = [{"experiment": "E5", "country": "KR", "setting": "확장 전체", "removed_group": "-", **full, "AP_drop_test": 0.0}]
    for g, cols in GROUPS_KR.items():
        r = ap_row(seg, fit_scores("LightGBM", [c for c in EXT_KR if c not in cols], seg)[1])
        abl.append({"experiment": "E5", "country": "KR", "setting": f"{GROUP_KO[g]} 제거", "removed_group": GROUP_KO[g],
                    **r, "AP_drop_test": full["AP_test"] - r["AP_test"]})

    return {"perf": pd.DataFrame(perf), "thr": pd.DataFrame(thr), "calib": pd.DataFrame(calib), "loc": loc,
            "abl": pd.DataFrame(abl), "types": type_recall(seg, lgb["확장 전체"])}


def type_recall(seg: dict[str, pd.DataFrame], sc: dict[str, np.ndarray]) -> pd.DataFrame:
    """E6: 테스트 구간 이상거래 유형별 Recall. 임계값은 검증 구간에서 비용별로 고른 값을 그대로 쓴다."""
    te = seg["test"]
    if "fraud_type" not in te.columns or te.loc[te["is_fraud"] == 1, "fraud_type"].isna().all():
        return pd.DataFrame()
    rows = []
    for r in COST_RATIOS:
        t, _ = best_threshold(seg["valid"]["is_fraud"], sc["valid"], r)
        hit = sc["test"] >= t
        pos = te["is_fraud"].to_numpy() == 1
        types = te["fraud_type"].fillna("(유형 없음)").to_numpy()
        for ft in sorted(set(types[pos])):
            m = pos & (types == ft)
            rows.append({"experiment": "E6", "country": "KR", "setting": "확장 전체", "model": "LightGBM",
                         "cost_ratio": f"{r}:1", "threshold": t, "fraud_type": ft, "n_test": int(m.sum()),
                         "detected": int((hit & m).sum()), "Recall": float((hit & m).sum() / m.sum())})
    return pd.DataFrame(rows)


def run_cross(us: dict[str, pd.DataFrame], kr: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """E3 국가 간 일반화 (공통 특징 5개, LightGBM). 학습 국가 train으로 학습 → 평가 국가 test에서 평가.
    같은 국가 조합(US→US, KR→KR)도 비교 기준으로 함께 기록한다. 두 나라 라벨을 한 학습 데이터로 합치지 않는다.
    임계값: ① 평가 국가 검증 구간에서 다시 고름 ② 학습 국가 검증 구간 임계값 그대로 ③ 기본 0.5."""
    data = {"US": us, "KR": kr}
    models = {c: model_for("LightGBM", CROSS_FEATURES, d["train"]).fit(d["train"][CROSS_FEATURES], d["train"]["is_fraud"])
              for c, d in data.items()}
    rows = []
    for tr_c, m in models.items():
        own_valid = m.predict_proba(data[tr_c]["valid"][CROSS_FEATURES])[:, 1]
        for ev_c, d in data.items():
            sv, st_ = (m.predict_proba(d[s][CROSS_FEATURES])[:, 1] for s in ("valid", "test"))
            yv, yt = d["valid"]["is_fraud"], d["test"]["is_fraud"]
            base = {"experiment": "E3", "train_country": tr_c, "eval_country": ev_c, "model": "LightGBM",
                    "features": ", ".join(CROSS_FEATURES), "AP_test": average_precision_score(yt, st_),
                    "baseline_test": float(yt.mean())}
            for r in COST_RATIOS:
                for policy, t in (("평가 국가 검증 임계값", best_threshold(yv, sv, r)[0]),
                                  ("학습 국가 검증 임계값", best_threshold(data[tr_c]["valid"]["is_fraud"], own_valid, r)[0]),
                                  ("기본 0.5", DEFAULT_T)):
                    rows.append({**base, "cost_ratio": f"{r}:1", "r": r, "policy": policy, **at_threshold(yt, st_, t, r)})
            print(f"E3 {tr_c}→{ev_c}: AP_test={base['AP_test']:.4f} (기준선 {base['baseline_test']:.4f})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 실행
def run_us() -> dict[str, pd.DataFrame]:
    seg = load_us()
    half = valid_halves(seg["valid"])
    perf, thr, calib = [], [], []
    lgb_scores = {}

    # E1: 미국 단독 — v1 6개 특징 / 확장 전체, 모델 3종
    for setting, cols in (("v1 6개 특징", V1), ("확장 전체", EXT_US)):
        for kind in MODEL_KINDS:
            m, sc = fit_scores(kind, cols, seg)
            yi = seg["internal_test"]["is_fraud"]
            perf.append({"experiment": "E1", "country": "US", "setting": setting, "model": kind,
                         "n_features": len(cols), "features": ", ".join(cols), **ap_row(seg, sc),
                         **{f"{k}_internal_test@0.5": v for k, v in at_threshold(yi, sc["internal_test"], DEFAULT_T).items()
                            if k in ("Precision", "Recall", "F1", "FP", "FN")}})
            thr += threshold_rows("US", setting, kind, seg, sc)
            print(f"E1 {setting} {kind}: AP_internal_test={perf[-1]['AP_internal_test']:.4f}")
            if kind == "LightGBM":
                lgb_scores[setting] = sc
                c, coef = calibration_row(seg, sc["valid"], half)
                calib.append({"country": "US", "setting": setting, "model": kind, **c,
                              "platt_a": coef[0], "platt_b": coef[1]})
                if setting == "v1 6개 특징":
                    os.makedirs(MODELS_V2, exist_ok=True)
                    with open(f"{MODELS_V2}/platt_lightgbm_v1.json", "w", encoding="utf-8") as fh:
                        json.dump({"a": coef[0], "b": coef[1], "fit": "valid 앞 절반", "model": "models/LightGBM.joblib",
                                   "formula": "p = 1/(1+exp(-(a*logit(s)+b)))"}, fh, ensure_ascii=False, indent=2)

    # E4: 위치 특징(D군) 추가 전후 — LightGBM
    loc = []
    pairs = [("v1 6개 특징", V1, None), ("v1 + D 위치", V1_PLUS_D, None),
             ("확장 (D 제외)", NO_D_US, None), ("확장 전체 (D 포함)", EXT_US, lgb_scores["확장 전체"])]
    for setting, cols, cached in pairs:
        sc = cached if cached is not None else (lgb_scores["v1 6개 특징"] if cols is V1 else fit_scores("LightGBM", cols, seg)[1])
        loc.append({"experiment": "E4", "country": "US", "setting": setting, "model": "LightGBM",
                    "with_location": setting in ("v1 + D 위치", "확장 전체 (D 포함)"), "n_features": len(cols), **ap_row(seg, sc)})
        if setting in ("v1 + D 위치", "확장 (D 제외)"):
            thr += threshold_rows("US", setting, "LightGBM", seg, sc)
        print(f"E4 {setting}: AP_internal_test={loc[-1]['AP_internal_test']:.4f}")
    loc = pd.DataFrame(loc)
    for base, plus in (("v1 6개 특징", "v1 + D 위치"), ("확장 (D 제외)", "확장 전체 (D 포함)")):
        b = loc.loc[loc.setting == base, "AP_internal_test"].iloc[0]
        loc.loc[loc.setting == plus, "AP_gain_vs_without_D"] = loc.loc[loc.setting == plus, "AP_internal_test"] - b

    # E5: 특징군 제거 — 확장 전체 기준, LightGBM
    full = ap_row(seg, lgb_scores["확장 전체"])
    abl = [{"experiment": "E5", "country": "US", "setting": "확장 전체", "removed_group": "-", **full, "AP_drop_internal_test": 0.0}]
    for g, cols in GROUPS_US.items():
        keep = [c for c in EXT_US if c not in cols]
        _, sc = fit_scores("LightGBM", keep, seg)
        r = ap_row(seg, sc)
        abl.append({"experiment": "E5", "country": "US", "setting": f"{GROUP_KO[g]} 제거", "removed_group": GROUP_KO[g],
                    **r, "AP_drop_internal_test": full["AP_internal_test"] - r["AP_internal_test"]})
        print(f"E5 {GROUP_KO[g]} 제거: AP_internal_test={r['AP_internal_test']:.4f}")

    # 개인 위험도 분포 점검용: 학습 구간 1~99 백분위 (설계서 6.3 ④)
    tr = seg["train"]
    ranges = {c: [float(tr[c].quantile(0.01)), float(tr[c].quantile(0.99))] for c in V1 if c != "insufficient_history"}
    with open(f"{MODELS_V2}/train_ranges_v1.json", "w", encoding="utf-8") as fh:
        json.dump({"segment": "train", "percentiles": [1, 99], "ranges": ranges}, fh, ensure_ascii=False, indent=2)

    return {"perf": pd.DataFrame(perf), "thr": pd.DataFrame(thr), "calib": pd.DataFrame(calib),
            "loc": loc, "abl": pd.DataFrame(abl)}


def run_e4_decomposition() -> pd.DataFrame:
    """[v2 추가 2026-11-28] E4 분해: D군에는 시간 간격(gap_prev_min)이 들어 있어, D군 효과가 '위치'에서 온 것인지
    '직전 거래와의 시간 간격'에서 온 것인지 나눠 본다. Sparkov 좌표는 무작위이므로 해석에 필요하다.
    실행: python -m src.experiments_country --e4b
    """
    seg = load_us()
    rows = []
    for setting, extra in (("v1 6개 특징", []), ("v1 + 시간 간격만", ["gap_prev_min"]),
                           ("v1 + 거리만", ["dist_prev_km"]), ("v1 + 거리·속도·원거리 (시간 간격 제외)",
                                                               ["dist_prev_km", "speed_kmh", "far_fast"]),
                           ("v1 + D 위치 전체", GROUPS_US["D"])):
        _, sc = fit_scores("LightGBM", V1 + extra, seg)
        rows.append({"experiment": "E4b", "country": "US", "setting": setting, "added": ", ".join(extra) or "-",
                     **ap_row(seg, sc)})
        print(f"E4b {setting}: AP_internal_test={rows[-1]['AP_internal_test']:.4f}")
    out = pd.DataFrame(rows)
    out["AP_gain_vs_v1_internal_test"] = out["AP_internal_test"] - out.loc[0, "AP_internal_test"]
    out.to_csv(f"{OUT}/e4b_location_decomposition.csv", index=False, encoding="utf-8-sig")
    return out


PERSONAL_FEATURES = [c for c in V1 if c != "hour_rarity"]


def train_personal_model() -> pd.DataFrame:
    """[v2 추가 2026-11-28] 개인 거래 점수용 LightGBM (v1에서 hour_rarity 제외).
    이유: Sparkov 학습 구간의 hour_rarity는 0.83~1.0에만 분포한다(가상 사용자가 24시간에 고르게 결제).
    실제 개인은 같은 시간대에 반복 결제해 hour_rarity≈0이 흔하고, 이는 모델이 본 적 없는 값이라
    평범한 점심 결제가 '주의'로 나오는 외삽 문제가 생긴다(브라우저 점검에서 확인). 시간대 희소성은 규칙 기반 정보로만 보여준다.
    실행: python -m src.experiments_country --personal
    산출: models/v2/lightgbm_personal.joblib, personal_model.json(임계값·Platt·학습 범위), outputs/v2/personal_model.csv
    """
    import joblib

    seg = load_us()
    m, sc = fit_scores("LightGBM", PERSONAL_FEATURES, seg)
    half = valid_halves(seg["valid"])
    cal, coef = calibration_row(seg, sc["valid"], half)
    th = {f"{r}": best_threshold(seg["valid"]["is_fraud"], sc["valid"], r)[0] for r in COST_RATIOS}
    tr = seg["train"]
    ranges = {c: [float(tr[c].quantile(0.01)), float(tr[c].quantile(0.99))] for c in PERSONAL_FEATURES
              if c != "insufficient_history"}
    os.makedirs(MODELS_V2, exist_ok=True)
    joblib.dump(m, f"{MODELS_V2}/lightgbm_personal.joblib")
    with open(f"{MODELS_V2}/personal_model.json", "w", encoding="utf-8") as fh:
        json.dump({"features": PERSONAL_FEATURES, "thresholds": th, "platt": {"a": coef[0], "b": coef[1]},
                   "train_ranges_1_99": ranges, "trained_on": "US Sparkov train segment", "pos_weight": POS_WEIGHT},
                  fh, ensure_ascii=False, indent=2)
    row = {"model": "LightGBM (hour_rarity 제외, 개인 점수용)", "features": ", ".join(PERSONAL_FEATURES),
           **ap_row(seg, sc), **cal, **{f"threshold_{r}:1": v for r, v in th.items()}}
    out = pd.DataFrame([row])
    out.to_csv(f"{OUT}/personal_model.csv", index=False, encoding="utf-8-sig")
    print(out.T)
    return out


def status_table(kr_available: bool, kr_types: bool = False) -> pd.DataFrame:
    rows = [
        {"experiment": "E1", "country": "US", "description": "미국 단독 (v1 6개 / 확장 전체), 3종", "status": "측정"},
        {"experiment": "E4", "country": "US", "description": "위치 특징 추가 전후 (시뮬레이터 합성 좌표 조건)", "status": "측정"},
        {"experiment": "E5", "country": "US", "description": "특징군 A·B·C·D 제거", "status": "측정"},
        {"experiment": "E4b", "country": "US", "description": "D군 분해: 시간 간격 vs 거리 (--e4b로 실행)", "status": "측정"},
    ]
    if kr_available:
        rows += [{"experiment": e, "country": "KR", "description": d, "status": "측정"} for e, d in (
            ("E2", "한국 단독 (v1 사용 가능 3개 / 확장 전체), 3종"),
            ("E3", "국가 간 일반화 US→KR, KR→US (공통 특징 " + ", ".join(CROSS_FEATURES) + ")"),
            ("E4", "한국 지역 특징(D) 추가 전후"),
            ("E5", "한국 특징군 A·B·C·D 제거"))]
        rows.append({"experiment": "E6", "country": "KR", "description": "한국 이상거래 유형별 Recall",
                     "status": "측정" if kr_types else "미측정(이상거래 유형 열 없음)"})
        return pd.DataFrame(rows)
    rows += [not_measured_rows("E2", "한국 단독, 3종"),
             not_measured_rows("E3", "국가 간 일반화 US→KR, KR→US (공통 특징 " + ", ".join(CROSS_FEATURES) + ")"),
             not_measured_rows("E4", "한국 위치(지역) 특징 추가 전후"),
             not_measured_rows("E5", "한국 특징군 제거"),
             not_measured_rows("E6", "한국 이상거래 유형별 Recall (선택)")]
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    kr_available = os.path.exists(KR_FEATURES)
    res = run_us()
    kr_types = False
    if kr_available:
        # [v2 변경 2026-11-28] 한국 특징 파일이 있으면 한국 실험을 실행한다 (이전에는 일부러 멈췄음)
        kr = load_kr()
        kres = run_kr(kr)
        for k in ("thr", "calib", "loc", "abl"):
            res[k] = pd.concat([res[k], kres[k]], ignore_index=True)  # 결과 표만 합침. 학습 데이터는 국가별로 따로
        kres["perf"].to_csv(f"{OUT}/e2_kr_performance.csv", index=False, encoding="utf-8-sig")
        run_cross(load_us(), kr).to_csv(f"{OUT}/e3_cross_country.csv", index=False, encoding="utf-8-sig")
        kr_types = not kres["types"].empty
        if kr_types:
            kres["types"].to_csv(f"{OUT}/e6_kr_type_recall.csv", index=False, encoding="utf-8-sig")
    save = lambda df, name: df.to_csv(f"{OUT}/{name}", index=False, encoding="utf-8-sig")
    save(res["perf"], "e1_country_performance.csv")
    save(res["thr"], "cost_thresholds_v2.csv")
    save(res["calib"], "e1_calibration.csv")
    save(res["loc"], "e4_location_effect.csv")
    save(res["abl"], "e5_group_ablation.csv")
    save(status_table(kr_available, kr_types), "experiment_status.csv")
    print(res["perf"][["setting", "model", "AP_internal_test", "baseline_internal_test", "AP_test", "baseline_test"]])
    print("done")


if __name__ == "__main__":
    import sys

    if "--e4b" in sys.argv:
        run_e4_decomposition()
    elif "--personal" in sys.argv:
        train_personal_model()
    else:
        main()
