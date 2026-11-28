"""개인 거래 확인 결과(피드백) 기록 테스트 (이슈 3-4, 설계서 12장 2026-11-28)."""
import pathlib
import sqlite3

import pandas as pd
import pytest

from src import risk
from src.sources import personal
from src.store import FEEDBACK, PersonalStore


def saved_store(tmp_path, n=3):
    st = PersonalStore(tmp_path / "p")
    rows = pd.concat([personal.manual_entry(f"2026-11-{d:02d}", "12:30", 9000 + d, "KRW", "김밥집", card_alias="생활카드")[0]
                      for d in range(1, n + 1)], ignore_index=True)
    st.insert(rows)
    return st, rows["transaction_id"].tolist()


def test_set_update_and_clear(tmp_path):
    st, ids = saved_store(tmp_path)
    st.set_feedback({ids[0]: "확인함", ids[1]: "사기였음"})
    assert st.feedback().set_index("transaction_id")["status"].to_dict() == {ids[0]: "확인함", ids[1]: "사기였음"}
    st.set_feedback({ids[0]: "사기였음", ids[1]: None})  # 바꾸기·지우기
    assert st.feedback().set_index("transaction_id")["status"].to_dict() == {ids[0]: "사기였음"}


def test_rejects_unknown_status_and_unsaved_id(tmp_path):
    st, ids = saved_store(tmp_path)
    with pytest.raises(ValueError, match="허용되지 않는"):
        st.set_feedback({ids[0]: "아마도"})
    with pytest.raises(ValueError, match="저장되지 않은"):
        st.set_feedback({"없는-ID": "확인함"})
    assert st.feedback().empty


def test_kept_on_replace_removed_on_delete_and_wipe(tmp_path):
    st, ids = saved_store(tmp_path)
    st.set_feedback({i: "확인함" for i in ids})
    rows = st.query()
    res, _ = risk.score(rows)
    st.replace(risk.to_store_rows(rows, res))  # 위험도 결과 저장(덮어쓰기)은 기록을 유지
    assert len(st.feedback()) == 3
    st.delete([ids[0]])  # 거래를 지우면 그 기록도 지움
    assert set(st.feedback()["transaction_id"]) == set(ids[1:])
    st.wipe()
    assert st.feedback().empty
    raw = (tmp_path / "p" / "personal.db").read_bytes()
    assert "사기였음".encode() not in raw and "확인함".encode() not in raw


def test_feedback_not_in_transactions_schema(tmp_path):
    st, ids = saved_store(tmp_path)
    st.set_feedback({ids[0]: "사기였음"})
    q = st.query()
    assert q["is_fraud"].isna().all()  # 사용자 표시가 라벨로 바뀌지 않음
    with sqlite3.connect(st.path) as con:
        cols = [r[1] for r in con.execute("PRAGMA table_info(transactions)")]
    assert "status" not in cols


def test_training_and_scoring_code_never_read_feedback():
    for f in ["src/build_data.py", "src/experiments.py", "src/features.py", "src/experiments_country.py",
              "src/build_features_v2.py", "src/features_ext.py", "src/risk.py"]:
        text = pathlib.Path(f).read_text(encoding="utf-8")
        assert "feedback" not in text, f


def test_values():
    assert FEEDBACK == ("확인함", "사기였음")
