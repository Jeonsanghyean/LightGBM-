"""한국 실험 코드 테스트 (E2·E3·한국 E4·E5·E6). 이슈 3-3을 데이터 확보 전에 준비한다.

가짜 행은 이 테스트 안에서만 쓰며, 수치는 결과로 보고하지 않는다 (설계서 9.1 B2).
가짜 한국 원본 → aihub_card.to_common → assign_segments → build_features_v2.build_kr → load_kr → run_kr
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from src import build_features_v2 as bf
from src import experiments_country as ec
from src.sources import aihub_card as kr
from tests.test_week2 import fake_kr


def fake_kr_raw(n=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2021-01-01", "2024-06-30", freq="D").strftime("%Y%m%d").tolist()
    fraud = rng.random(n) < 0.08
    rows = []
    for i in range(n):
        f = bool(fraud[i])
        rows.append({
            "승인일자": days[rng.integers(0, len(days))], "승인SEQ": str(i), "승인시간대": str(rng.integers(0, 24)),
            "카드KEY": f"C{rng.integers(0, 40)}", "가맹점KEY": f"M{rng.integers(0, 15)}",
            # 이상 거래가 조금 더 큰 편이지만 정상 거래와 겹치게 (완전히 구분되면 비용 0이 되어 현실과 다름)
            "통합승인금액": str(int(rng.lognormal(11.0 if f else 10.0, 1.0))),
            "이상거래여부": "1" if f else "0", "이상거래유형": rng.choice(["T1", "T2"]) if f else "",
            "가맹점광역시도코드": str(rng.integers(1, 18)), "국내해외여부": "해외" if f and rng.random() < 0.3 else "국내",
        })
    return fake_kr(rows)


@pytest.fixture
def kr_seg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    common, extra = kr.to_common(fake_kr_raw())
    common["segment"] = kr.assign_segments(common).values
    out = pathlib.Path(bf.KR)
    out.mkdir(parents=True)
    common.to_parquet(out / "transactions.parquet", index=False)
    extra.to_parquet(out / "extra.parquet", index=False)
    feats = bf.build_kr()
    assert feats is not None and len(feats) == len(common)
    return ec.load_kr(f"{bf.KR}/features_v2.parquet")


def fake_us_seg(seed=1) -> dict[str, pd.DataFrame]:
    """E3용 미국 쪽 가짜 구간 (공통 특징 5개만). 크기를 한국과 다르게 두어 학습 행 출처를 구분한다."""
    rng = np.random.default_rng(seed)
    seg = {}
    for s, n in (("train", 1234), ("valid", 400), ("internal_test", 400), ("test", 400)):
        y = (rng.random(n) < 0.05).astype(int)
        seg[s] = pd.DataFrame({
            "amt_ratio": rng.lognormal(0, 0.5, n) + 4 * y, "count_prev_hour_bucket": rng.poisson(0.3, n),
            "count_same_day_prior": rng.poisson(1, n), "hour_rarity": rng.uniform(0.8, 1, n),
            "insufficient_history": (rng.random(n) < 0.05).astype(int), "is_fraud": y})
    return seg


def test_load_kr_segments_sorted_and_no_internal_test(kr_seg):
    assert set(kr_seg) == {"train", "valid", "test"}
    for s, d in kr_seg.items():
        t = ec.event_time(d)
        assert t.is_monotonic_increasing, s
        assert d["is_fraud"].isin([0, 1]).all()
    assert ec.event_time(kr_seg["train"]).max() < ec.event_time(kr_seg["valid"]).min()
    assert ec.event_time(kr_seg["valid"]).max() < ec.event_time(kr_seg["test"]).min()


def test_load_kr_drops_unlabeled(tmp_path):
    f = pd.DataFrame({"transaction_date": ["2021-01-01", "2023-08-01", "2024-02-01", "2024-02-02"],
                      "hour_bucket": [1, 2, 3, 4], "is_fraud": [0, 1, 0, None],
                      "segment": ["train", "valid", "test", "test"]})
    f.to_parquet(tmp_path / "f.parquet", index=False)
    seg = ec.load_kr(str(tmp_path / "f.parquet"))
    assert sum(len(d) for d in seg.values()) == 3


def test_run_kr_outputs(kr_seg):
    res = ec.run_kr(kr_seg)
    perf = res["perf"]
    assert len(perf) == 6 and set(perf["setting"]) == {"v1 사용 가능 3개", "확장 전체"}
    assert (perf["country"] == "KR").all() and not any("internal_test" in c for c in perf.columns)
    assert perf["AP_test"].between(0, 1).all()
    assert np.isclose(perf["baseline_test"].iloc[0], kr_seg["test"]["is_fraud"].mean())
    # 5·15·60분 건수는 한국 특징에 쓰지 않음
    assert not perf["features"].str.contains("count_5min|count_15min|count_60min").any()
    assert len(res["thr"]) == 2 * 3 * 3 * 2 and (res["thr"]["country"] == "KR").all()
    assert len(res["calib"]) == 2
    assert res["loc"]["with_location"].tolist() == [False, True, False, True]
    assert res["loc"]["AP_gain_vs_without_D_test"].notna().sum() == 2
    assert len(res["abl"]) == 5 and res["abl"]["AP_drop_test"].iloc[0] == 0
    types = res["types"]
    assert set(types["fraud_type"]) <= {"T1", "T2"} and types["Recall"].between(0, 1).all()
    n_pos = int(kr_seg["test"]["is_fraud"].sum())
    assert types.groupby("cost_ratio")["n_test"].sum().eq(n_pos).all()


def test_type_recall_empty_without_types(kr_seg):
    seg = {k: v.assign(fraud_type=None) for k, v in kr_seg.items()}
    sc = {"valid": np.zeros(len(seg["valid"])), "test": np.zeros(len(seg["test"]))}
    assert ec.type_recall(seg, sc).empty


def test_cross_country_trains_each_country_separately(kr_seg, monkeypatch):
    us = fake_us_seg()
    sizes = []
    real = ec.model_for

    def spy(kind, cols, data):
        m = real(kind, cols, data)
        fit = m.fit
        m.fit = lambda X, y: (sizes.append(len(X)), fit(X, y))[1]
        return m

    monkeypatch.setattr(ec, "model_for", spy)
    out = ec.run_cross(us, kr_seg)
    # 학습은 나라별 train 구간 한 번씩: 두 나라 행을 합쳐 학습하지 않음
    assert sorted(sizes) == sorted([len(us["train"]), len(kr_seg["train"])])
    pairs = set(zip(out["train_country"], out["eval_country"]))
    assert pairs == {("US", "US"), ("US", "KR"), ("KR", "US"), ("KR", "KR")}
    assert len(out) == 4 * 3 * 3
    assert set(out["policy"]) == {"평가 국가 검증 임계값", "학습 국가 검증 임계값", "기본 0.5"}
    # 같은 국가 조합에서는 두 임계값 정책이 같은 값
    same = out[(out.train_country == out.eval_country) & (out.policy != "기본 0.5")]
    assert same.groupby(["train_country", "r"])["threshold"].nunique().eq(1).all()
    assert out["features"].iloc[0] == ", ".join(ec.CROSS_FEATURES)


def test_status_table_kr():
    st = ec.status_table(True, kr_types=True)
    assert (st.loc[st.country == "KR", "status"] == "측정").all() and len(st[st.country == "KR"]) == 5
    st = ec.status_table(True, kr_types=False)
    assert st.loc[(st.country == "KR") & (st.experiment == "E6"), "status"].iloc[0].startswith("미측정")
    st = ec.status_table(False)
    assert st.loc[st.country == "KR", "status"].str.startswith("미측정").all()


def test_threshold_rows_zero_default_cost_is_nan():
    """기본 0.5의 총비용이 0이면 변화율을 0으로 나누지 않고 NaN으로 둔다."""
    y = pd.Series([0, 0, 1, 1])
    seg = {"valid": pd.DataFrame({"is_fraud": y}), "test": pd.DataFrame({"is_fraud": y})}
    sc = {"valid": np.array([0.1, 0.2, 0.8, 0.9]), "test": np.array([0.1, 0.2, 0.8, 0.9])}
    rows = pd.DataFrame(ec.threshold_rows("KR", "x", "LightGBM", seg, sc))
    assert rows.loc[rows.policy == "검증 임계값", "cost_change_pct_vs_default"].isna().all()
