"""화면 테스트: 대시보드 v2의 화면 1~6을 streamlit AppTest로 실행한다 (이슈 3-4).

실제 Sparkov·AI-Hub 데이터 없이 돌도록, 임시 폴더에 가짜 프로젝트를 만든다.
- 코드·모델·설정·v2 결과 CSV: 저장소 것을 링크
- 미국 연구 데이터: 가짜 Sparkov 원본 행 → src.sources.sparkov.to_common → add_extended
- outputs/test_scores.parquet: 가짜 원본 Test 구간 점수
- 개인 저장소: FDS_PERSONAL_DIR로 임시 폴더 (실제 개인 저장소는 건드리지 않음)
"""
import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import risk
from src.features import ALL_FEATURES as V1
from src.features_ext import add_extended
from src.sources import personal
from src.sources.sparkov import to_common
from src.store import PersonalStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = ["data_select", "lookup", "map", "analysis", "performance", "personal"]
N, N_TEST = 60, 20  # 마지막 20건을 원본 Test 구간으로 둔다


def fake_raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.Timestamp("2020-06-01") + pd.to_timedelta(np.sort(rng.integers(0, 30 * 24 * 3600, N)), unit="s")
    return pd.DataFrame({
        "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": rng.choice([4000000000000001, 4000000000000002, 4000000000000003], N),
        "merchant": [f"fraud_가게{i % 7}" for i in range(N)],
        "category": rng.choice(["grocery_pos", "shopping_net", "travel"], N),
        "amt": rng.uniform(1, 500, N).round(2),
        "merch_lat": rng.uniform(40, 41, N), "merch_long": rng.uniform(-74, -73, N),
        "is_fraud": (np.arange(N) % 10 == 0).astype(int),
        "trans_num": [f"tx{i:03d}" for i in range(N)],
    })


@pytest.fixture
def project(tmp_path, monkeypatch):
    for name in ["src", "views", "models", "config", ".streamlit", "app_v2.py"]:
        (tmp_path / name).symlink_to(ROOT / name)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "v2").symlink_to(ROOT / "outputs" / "v2")

    raw = fake_raw()
    us = to_common(raw)
    us["segment"] = ["train"] * (N - N_TEST) + ["test"] * N_TEST
    research = tmp_path / "data" / "research" / "us"
    research.mkdir(parents=True)
    us.to_parquet(research / "transactions.parquet", index=False)
    add_extended(us).to_parquet(research / "features_v2.parquet", index=False)

    rng = np.random.default_rng(1)
    test = raw.tail(N_TEST).reset_index(drop=True)
    t_high, t_watch = risk.thresholds()
    scores = pd.DataFrame({"trans_num": test["trans_num"], "trans_date_trans_time": test["trans_date_trans_time"],
                           "amt": test["amt"], "is_fraud": test["is_fraud"],
                           **{f: rng.uniform(0, 1, N_TEST) for f in V1}})
    scores["insufficient_history"] = (scores["insufficient_history"] > 0.5).astype(int)
    # 세 위험 수준이 모두 나오도록 점수를 고정
    scores["score_LightGBM"] = np.r_[[t_high + 0.01] * 3, [t_watch + 0.001] * 3, rng.uniform(0, t_watch * 0.9, N_TEST - 6)]
    scores["score_XGBoost"] = scores["score_LogisticRegression"] = scores["score_LightGBM"]
    scores.to_parquet(tmp_path / "outputs" / "test_scores.parquet", index=False)

    monkeypatch.setenv("FDS_PERSONAL_DIR", str(tmp_path / "personal"))
    monkeypatch.chdir(tmp_path)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield tmp_path
    st.cache_data.clear()
    st.cache_resource.clear()


def run(page: str, source: str, root: pathlib.Path) -> AppTest:
    """진입점 app_v2.py(st.navigation)로 띄운 뒤 화면을 바꾼다. 화면 파일만 따로 띄우면 st.page_link가 실패한다."""
    at = AppTest.from_file(str(root / "app_v2.py"), default_timeout=60)
    at.session_state["source"] = source
    at.run()
    return at if page == "data_select" else at.switch_page(f"views/{page}.py").run()


def add_personal(n_days=8):
    rows = [personal.manual_entry(f"2026-11-{d:02d}", "12:30", 9000 + 100 * d, "KRW", "김밥집", card_alias="생활카드")[0]
            for d in range(1, n_days + 1)]
    df = pd.concat(rows, ignore_index=True)
    PersonalStore().insert(df)
    return df


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("source", ["us", "kr", "personal"])
def test_every_page_opens_without_error(project, page, source):
    """설계서 10장: 세 출처 모두 화면 1~6이 오류 없이 열림 (개인 저장소가 빈 경우)."""
    at = run(page, source, project)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["lookup", "map", "analysis"])
def test_personal_pages_with_data(project, page):
    add_personal()
    at = run(page, "personal", project)
    assert not at.exception, [e.value for e in at.exception]


def test_kr_pages_say_missing(project):
    for page in ["lookup", "map", "analysis"]:
        at = run(page, "kr", project)
        assert any("미확보" in w.value for w in at.warning), page


def test_performance_marks_unmeasured(project):
    at = run("performance", "us", project)
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption) \
        + " ".join(df.value.to_string() for df in at.dataframe)
    assert "미측정" in text


def lookup_table(at) -> pd.DataFrame:
    return at.dataframe[0].value


def test_lookup_shows_us_risk_only_for_test_segment(project):
    """화면 2: 원본 Test 구간은 v1 점수의 위험 수준, 학습·검증 구간은 '미산출'."""
    at = run("lookup", "us", project)
    assert not at.exception
    risk_col = lookup_table(at)["위험도"]
    assert len(risk_col) == N
    counts = risk_col.value_counts().to_dict()
    assert counts.get("높음") == 3 and counts.get("주의") == 3
    assert counts.get("미산출 (학습·검증 구간)") == N - N_TEST
    assert not risk_col.str.contains("주차").any()


def test_lookup_detail_shows_score(project):
    at = run("lookup", "us", project)
    at.text_input(key="detail_us").input("tx059").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "모델 위험도:" in text and "점수 " in text


def test_lookup_personal_risk_before_and_after_scoring(project):
    df = add_personal()
    at = run("lookup", "personal", project)
    assert set(lookup_table(at)["위험도"]) == {"미산출 (화면 6에서 분석 실행)"}

    res, _ = risk.score(PersonalStore().query())
    PersonalStore().replace(risk.to_store_rows(PersonalStore().query(), res))
    at = run("lookup", "personal", project)
    assert set(lookup_table(at)["위험도"]) <= {"높음", "주의", "낮음", "판단 보류"}
    assert len(lookup_table(at)) == len(df)


def test_view_tests_never_touch_real_personal_store(project):
    assert PersonalStore().folder.resolve().is_relative_to(project.resolve())


@pytest.mark.parametrize("with_feedback", [False, True])
def test_personal_analysis_shows_saved_feedback(project, with_feedback):
    """화면 6 분석 결과: 저장된 확인 결과가 '내 확인' 칸과 건수 문구에 나온다. 기록이 없어도 열린다."""
    df = add_personal()
    if with_feedback:
        PersonalStore().set_feedback({df["transaction_id"].iloc[0]: "사기였음"})
    at = AppTest.from_file(str(project / "app_v2.py"), default_timeout=60)
    at.session_state["personal_section"] = "분석 실행"
    at.run()
    at.switch_page("views/personal.py").run()
    next(b for b in at.button if b.label == "분석 실행").click().run()
    assert not at.exception, [e.value for e in at.exception]
    expect = ("사기였음 1건", "미확인 7건") if with_feedback else ("사기였음 0건", "미확인 8건")
    assert any(all(e in c.value for e in expect) for c in at.caption)


def test_performance_with_kr_results(project):
    """화면 5: 한국 결과 파일이 있으면 E2·E3·E6 표와 국가별 E4·E5를 보여주고, 미국 표는 미국 행만 쓴다.
    결과는 가짜 한국 행으로 만든 것이며 화면 동작 확인용이다."""
    import shutil

    from src import build_features_v2 as bf
    from src import experiments_country as ec
    from src.sources import aihub_card as kr
    from tests.test_kr_experiments import fake_kr_raw, fake_us_seg

    v2 = project / "outputs" / "v2"
    v2.unlink()
    shutil.copytree(ROOT / "outputs" / "v2", v2)
    common, extra = kr.to_common(fake_kr_raw())
    common["segment"] = kr.assign_segments(common).values
    (project / bf.KR).mkdir(parents=True)
    common.to_parquet(project / bf.KR / "transactions.parquet", index=False)
    extra.to_parquet(project / bf.KR / "extra.parquet", index=False)
    bf.build_kr()
    seg = ec.load_kr(f"{bf.KR}/features_v2.parquet")
    res = ec.run_kr(seg)
    for name, k in (("e4_location_effect.csv", "loc"), ("e5_group_ablation.csv", "abl"), ("cost_thresholds_v2.csv", "thr")):
        pd.concat([pd.read_csv(v2 / name), res[k]], ignore_index=True).to_csv(v2 / name, index=False)
    res["perf"].to_csv(v2 / "e2_kr_performance.csv", index=False)
    res["types"].to_csv(v2 / "e6_kr_type_recall.csv", index=False)
    ec.run_cross(fake_us_seg(), seg).to_csv(v2 / "e3_cross_country.csv", index=False)

    at = run("performance", "us", project)
    assert not at.exception, [e.value for e in at.exception]
    unmeasured = [i.value for i in at.info if "미측정" in i.value]
    assert unmeasured == [], unmeasured
    headers = [h.value for h in at.subheader]
    for h in ("한국 모델 성능 (E2)", "국가 간 일반화 (E3)", "한국 이상거래 유형별 Recall (E6)"):
        assert h in headers
    assert at.radio(key="thr_country").options == ["US", "KR"]
