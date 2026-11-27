"""개인 거래 로컬 저장소 (SQLite, 설계서 F7·표 10).

저장 위치: 기본값 %LOCALAPPDATA%\\CardFDS\\personal.db  (환경변수 FDS_PERSONAL_DIR로 변경 가능)
[설계 변경 2026-11-27] 설계서의 data/personal/ 대신 프로젝트 밖 로컬 폴더를 기본값으로 함.
  이유: 이 프로젝트 폴더는 OneDrive 안에 있어 data/personal/에 두면 개인 거래가 클라우드로 동기화된다.
- 연구 데이터(data/research/)와 다른 파일·다른 폴더 → 학습 코드가 읽지 않는다.
- 카드번호·CVC·인증정보 열은 테이블에 없다.
- import_log에는 행 내용 없이 시각·파일 해시·건수만 남긴다.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

from .schema import COLUMNS, conform, validate

SQL_TYPE = {"string": "TEXT", "Int64": "INTEGER", "float64": "REAL"}


def default_dir() -> Path:
    base = os.environ.get("FDS_PERSONAL_DIR") or os.path.join(
        os.environ.get("LOCALAPPDATA") or str(Path.home()), "CardFDS")
    return Path(base)


def _restrict_permissions(folder: Path) -> str:
    """Windows: 폴더 상속 권한을 끊고 현재 사용자만 전체 권한을 갖게 한다 (최초 생성 시 1회)."""
    if os.name != "nt":
        os.chmod(folder, 0o700)
        return "chmod 700"
    user = os.environ.get("USERNAME")
    if not user:
        return "skipped (USERNAME 없음)"
    r = subprocess.run(["icacls", str(folder), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F"],
                       capture_output=True, text=True)
    return "ok" if r.returncode == 0 else "failed"


class PersonalStore:
    def __init__(self, folder: Path | str | None = None):
        self.folder = Path(folder) if folder else default_dir()
        first = not self.folder.exists()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.permission_status = _restrict_permissions(self.folder) if first else "unchanged"
        self.path = self.folder / "personal.db"
        self._init()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _init(self):
        cols = ", ".join(f'"{c}" {SQL_TYPE[t]}' + (" PRIMARY KEY" if c == "transaction_id" else "")
                         for c, (t, _) in COLUMNS.items())
        with self._conn() as con:
            con.execute(f"CREATE TABLE IF NOT EXISTS transactions ({cols})")
            con.execute("CREATE TABLE IF NOT EXISTS import_log (imported_at TEXT, file_sha256 TEXT, "
                        "rows_saved INTEGER, rows_error INTEGER, rows_duplicate INTEGER)")

    # ---------------------------------------------------------------- 쓰기
    def insert(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        df = conform(df)[list(COLUMNS)]
        problems = validate(df)
        if problems:
            raise ValueError("저장 전 검증 실패: " + "; ".join(problems))
        rows = [tuple(None if pd.isna(v) else (int(v) if c in ("hour_bucket", "is_fraud") else v)
                      for c, v in zip(df.columns, r)) for r in df.itertuples(index=False)]
        q = f"INSERT INTO transactions VALUES ({','.join('?' * len(COLUMNS))})"
        with self._conn() as con:
            con.executemany(q, rows)
        return len(rows)

    def replace(self, df: pd.DataFrame) -> int:
        """편집된 행을 같은 transaction_id로 덮어쓴다."""
        if df.empty:
            return 0
        self.delete(df["transaction_id"].tolist())
        return self.insert(df)

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        with self._conn() as con:
            cur = con.executemany("DELETE FROM transactions WHERE transaction_id = ?", [(i,) for i in ids])
            return cur.rowcount

    def wipe(self) -> None:
        """전체 삭제. VACUUM으로 파일 안에 남은 페이지까지 정리한다."""
        with self._conn() as con:
            con.execute("DELETE FROM transactions")
            con.execute("DELETE FROM import_log")
        con = self._conn()
        con.execute("VACUUM")
        con.close()

    def log_import(self, file_bytes: bytes, saved: int, errors: int, dups: int) -> None:
        with self._conn() as con:
            con.execute("INSERT INTO import_log VALUES (?,?,?,?,?)",
                        (dt.datetime.now().isoformat(timespec="seconds"),
                         hashlib.sha256(file_bytes).hexdigest(), saved, errors, dups))

    # ---------------------------------------------------------------- 읽기
    def query(self, start: str | None = None, end: str | None = None, card: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM transactions WHERE 1=1", []
        if start:
            q += " AND transaction_date >= ?"
            args.append(str(start))
        if end:
            q += " AND transaction_date <= ?"
            args.append(str(end))
        if card:
            q += " AND card_id = ?"
            args.append(card)
        q += " ORDER BY transaction_date, transaction_time"
        with self._conn() as con:
            df = pd.read_sql_query(q, con, params=args)
        return conform(df) if len(df) else conform(pd.DataFrame(columns=["transaction_id"]))

    def cards(self) -> list[str]:
        with self._conn() as con:
            return [r[0] for r in con.execute("SELECT DISTINCT card_id FROM transactions ORDER BY 1")]

    def count(self) -> int:
        with self._conn() as con:
            return con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
