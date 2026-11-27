"""3~6주차: 가중치·보정 실험, 실험 A~D, 비용별 임계값 선택과 원본 Test 평가.

실행: python -m src.experiments   (먼저 python -m src.build_data)
"""
from __future__ import annotations

import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_FEATURES, COMMON, FEATURE_GROUPS

OUT, MODELS = "outputs", "models"
SEED = 42
POS_WEIGHT = 10  # 실험 A~D 공통 양성 가중치
COST_RATIOS = (5, 20, 100)
DEFAULT_T = 0.5
GROUP_KO = {"amount": "고액 결제", "repeat": "반복 결제", "hour": "이상 시간대"}


# ---------------------------------------------------------------- 모델
def make_model(kind: str, spw: float):
    if kind == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, scale_pos_weight=spw,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
    if kind == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
            scale_pos_weight=spw, tree_method="hist", random_state=SEED, n_jobs=-1,
            eval_metric="aucpr",
        )
    if kind == "LogisticRegression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight={0: 1, 1: spw}, max_iter=2000, random_state=SEED),
        )
    raise ValueError(kind)


def fit_score(kind, spw, cols, tr, *evals):
    m = make_model(kind, spw).fit(tr[cols], tr["is_fraud"])
    return m, [m.predict_proba(e[cols])[:, 1] for e in evals]


# ---------------------------------------------------------------- 지표
def at_threshold(y, s, t, r=None):
    y = np.asarray(y)
    pred = np.asarray(s) >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    res = {"threshold": t, "alerts": tp + fp, "TP": tp, "FP": fp, "FN": fn,
           "Precision": prec, "Recall": rec, "F1": f1}
    if r is not None:
        res["total_cost"] = fp + r * fn
    return res


def best_threshold(y, s, r):
    """총비용 FP + r·FN 최소 임계값. 비용이 같으면 경고가 적은(높은) 임계값."""
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    last = np.r_[ss[1:] != ss[:-1], True]  # 같은 점수 묶음의 끝
    thr, tp, fp = ss[last], tp[last], fp[last]
    cost = fp + r * (yy.sum() - tp)
    alerts = tp + fp
    # 경고 0건(모든 점수 위) 후보 추가
    thr = np.r_[np.nextafter(ss[0], np.inf), thr]
    cost = np.r_[r * yy.sum(), cost]
    alerts = np.r_[0, alerts]
    i = np.lexsort((alerts, cost))[0]
    return float(thr[i]), float(cost[i])


def platt(s_fit, y_fit, s_eval):
    z = lambda s: np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z(s_fit).reshape(-1, 1), y_fit)
    return lr.predict_proba(z(s_eval).reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------- 실험
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    d = pd.read_parquet("data/features.parquet")
    tr, va, it, te = (d[d.segment == s].reset_index(drop=True) for s in ("train", "valid", "internal_test", "test"))
    ratio = (tr.is_fraud == 0).sum() / tr.is_fraud.sum()
    print(f"neg/pos ratio in train = {ratio:.3f}")

    # 검증 구간을 시간순 앞·뒤 절반으로 나눠 Platt 보정 학습/평가
    half = len(va) // 2
    while half < len(va) and va.ts.iloc[half] == va.ts.iloc[half - 1]:
        half += 1
    va_a, va_b = va.iloc[:half], va.iloc[half:]

    # 4) 가중치 4종 + Platt 보정
    rows = []
    for name, w in (("1", 1), ("5", 5), ("10", 10), (f"neg/pos({ratio:.3f})", ratio)):
        _, (s_va, s_it) = fit_score("LightGBM", w, ALL_FEATURES, tr, va, it)
        s_a, s_b = s_va[:half], s_va[half:]
        cal_b = platt(s_a, va_a.is_fraud, s_b)
        rows.append({
            "scale_pos_weight": name,
            "AP_valid": average_precision_score(va.is_fraud, s_va),
            "AP_internal_test": average_precision_score(it.is_fraud, s_it),
            "Brier_before": brier_score_loss(va_b.is_fraud, s_b),
            "Brier_after_Platt": brier_score_loss(va_b.is_fraud, cal_b),
            "LogLoss_before": log_loss(va_b.is_fraud, s_b, labels=[0, 1]),
            "LogLoss_after_Platt": log_loss(va_b.is_fraud, cal_b, labels=[0, 1]),
            "mean_score_before": float(s_b.mean()),
            "mean_score_after": float(cal_b.mean()),
            "actual_fraud_rate": float(va_b.is_fraud.mean()),
        })
        print(rows[-1])
    pd.DataFrame(rows).to_csv(f"{OUT}/weight_calibration.csv", index=False, encoding="utf-8-sig")

    # A) 모델 비교 (내부 성능 비교 구간)
    models, scores = {}, {}
    rows = []
    for kind in ("LightGBM", "XGBoost", "LogisticRegression"):
        m, (s_va, s_it, s_te) = fit_score(kind, POS_WEIGHT, ALL_FEATURES, tr, va, it, te)
        models[kind], scores[kind] = m, {"valid": s_va, "internal_test": s_it, "test": s_te}
        rows.append({"model": kind, "AP_internal_test": average_precision_score(it.is_fraud, s_it),
                     **{k: v for k, v in at_threshold(it.is_fraud, s_it, DEFAULT_T).items() if k != "threshold"}})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(f"{OUT}/model_comparison.csv", index=False, encoding="utf-8-sig")

    # B) 특징군 단독 / C) 특징군 제거 (LightGBM, 가중치 10)
    full_ap = average_precision_score(it.is_fraud, scores["LightGBM"]["internal_test"])
    single, ablation = [], [{"setting": "전체 특징", "removed_group": "-", "AP_internal_test": full_ap, "AP_drop": 0.0}]
    for g, cols in FEATURE_GROUPS.items():
        _, (s,) = fit_score("LightGBM", POS_WEIGHT, cols, tr, it)
        single.append({"group": GROUP_KO[g], "features": ", ".join(cols),
                       "AP_internal_test": average_precision_score(it.is_fraud, s)})
        keep = [c for c in ALL_FEATURES if c not in cols]
        _, (s,) = fit_score("LightGBM", POS_WEIGHT, keep, tr, it)
        ap = average_precision_score(it.is_fraud, s)
        ablation.append({"setting": f"{GROUP_KO[g]} 제거", "removed_group": GROUP_KO[g],
                         "AP_internal_test": ap, "AP_drop": full_ap - ap})
    single.append({"group": "(기준) 사기 비율", "features": "-", "AP_internal_test": float(it.is_fraud.mean())})
    pd.DataFrame(single).to_csv(f"{OUT}/feature_single.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ablation).to_csv(f"{OUT}/feature_ablation.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(single), pd.DataFrame(ablation), sep="\n")

    # D) 비용별 임계값: 검증 구간에서 선택 → 원본 Test에 고정 적용
    thr_rows, cmp_rows, test_rows = [], [], []
    for kind in models:
        s_va, s_te = scores[kind]["valid"], scores[kind]["test"]
        test_rows.append({"model": kind, "AP_test": average_precision_score(te.is_fraud, s_te),
                          **at_threshold(te.is_fraud, s_te, DEFAULT_T)})
        for r in COST_RATIOS:
            t, vcost = best_threshold(va.is_fraud, s_va, r)
            vdef = at_threshold(va.is_fraud, s_va, DEFAULT_T, r)["total_cost"]
            thr_rows.append({"model": kind, "cost_ratio": f"{r}:1", "r": r, "threshold": t,
                             "valid_cost_selected": vcost, "valid_cost_default": vdef})
            base = at_threshold(te.is_fraud, s_te, DEFAULT_T, r)
            sel = at_threshold(te.is_fraud, s_te, t, r)
            for label, res in (("기본 0.5", base), ("검증 임계값", sel)):
                cmp_rows.append({"model": kind, "cost_ratio": f"{r}:1", "r": r, "policy": label, **res})
            cmp_rows[-1]["cost_change_pct_vs_default"] = (sel["total_cost"] - base["total_cost"]) / base["total_cost"] * 100
    pd.DataFrame(thr_rows).to_csv(f"{OUT}/thresholds.csv", index=False, encoding="utf-8-sig")
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(f"{OUT}/test_threshold_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(f"{OUT}/test_model_metrics.csv", index=False, encoding="utf-8-sig")
    print(cmp[["model", "cost_ratio", "policy", "threshold", "FP", "FN", "total_cost", "cost_change_pct_vs_default"]])

    # 대시보드용 산출물
    for kind, m in models.items():
        joblib.dump(m, f"{MODELS}/{kind}.joblib")
    with open(f"{MODELS}/thresholds.json", "w", encoding="utf-8") as f:
        json.dump({f"{x['model']}|{x['r']}": x["threshold"] for x in thr_rows}, f, indent=2)
    scored = te[["trans_num", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "is_fraud", "n_prev"] + ALL_FEATURES].copy()
    for kind in models:
        scored[f"score_{kind}"] = scores[kind]["test"]
    scored.to_parquet(f"{OUT}/test_scores.parquet", index=False)
    print("done")


if __name__ == "__main__":
    main()
