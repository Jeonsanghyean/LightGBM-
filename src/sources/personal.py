"""개인 카드 거래 입력·업로드 정규화 (설계서 F5·F6).

흐름: read_any → privacy.scan/drop_sensitive → guess_mapping(사용자 확인) → normalize → find_duplicates
- 시각·주소·좌표를 추정해서 채우지 않는다. 시각이 없으면 time_precision='day'.
- 주소에서는 시도명만 로컬 정규식으로 뽑는다 (지오코딩·외부 전송 없음, 설계서 6.3 ①).
- 실제 사기 여부(is_fraud)는 비워 둔다.
"""
from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

import pandas as pd
import yaml

from ..schema import conform, provenance

SYNONYM_PATH = Path(__file__).resolve().parents[2] / "config" / "column_synonyms.yaml"
TARGETS = ["datetime", "date", "time", "amount", "currency", "merchant_name", "merchant_address",
           "transaction_type", "transaction_status", "card_alias"]
CURRENCIES = ["KRW", "USD", "JPY", "EUR", "CNY", "GBP"]
CURRENCY_SYMBOL = {"₩": "KRW", "원": "KRW", "$": "USD", "US$": "USD", "¥": "JPY", "円": "JPY", "엔": "JPY",
                   "€": "EUR", "£": "GBP", "元": "CNY", "위안": "CNY"}

REGIONS = [
    ("서울", r"서울"), ("부산", r"부산"), ("대구", r"대구"), ("인천", r"인천"), ("광주", r"광주광역시|^광주\s|\s광주\s(?!시)"),
    ("대전", r"대전"), ("울산", r"울산"), ("세종", r"세종"), ("경기", r"경기"), ("강원", r"강원"),
    ("충북", r"충북|충청북도"), ("충남", r"충남|충청남도"), ("전북", r"전북|전라북도"), ("전남", r"전남|전라남도"),
    ("경북", r"경북|경상북도"), ("경남", r"경남|경상남도"), ("제주", r"제주"),
]


# ---------------------------------------------------------------- 파일 읽기
def read_any(data: bytes, filename: str) -> pd.DataFrame:
    """CSV(utf-8 / cp949) 또는 Excel을 문자열 표로 읽는다. 값 해석은 normalize에서 한다."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), dtype=str)
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 인코딩을 읽을 수 없습니다 (UTF-8 또는 CP949로 저장해 주세요).")


# ---------------------------------------------------------------- 열 매핑
def _norm(s: str) -> str:
    return re.sub(r"[\s_()\[\]./-]", "", str(s)).lower()


def load_synonyms() -> dict[str, list[str]]:
    with open(SYNONYM_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def guess_mapping(columns) -> dict[str, str | None]:
    syn = load_synonyms()
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for target in TARGETS:
        mapping[target] = None
        for word in syn.get(target, []):
            hit = next((c for c in columns if c not in used and _norm(word) in _norm(c)), None)
            if hit is not None:
                mapping[target] = hit
                used.add(hit)
                break
    if mapping["datetime"] and mapping["date"]:  # 일시 열이 있으면 날짜 열은 쓰지 않음
        used.discard(mapping["date"])
        mapping["date"] = None
    return mapping


# ---------------------------------------------------------------- 값 정규화
def _blank(v) -> bool:
    """None·NaN·pd.NA·빈 문자열을 모두 '값 없음'으로 본다 (pd.NA는 == 비교 시 오류가 나므로 먼저 걸러냄)."""
    return v is None or bool(pd.isna(v)) or str(v).strip() == ""


def parse_date(s) -> str | None:
    if s is None or pd.isna(s):
        return None
    t = str(s).strip()
    m = re.match(r"^(\d{4})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})일?", t)
    if not m:
        m2 = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", t)  # MM/DD/YYYY
        if not m2:
            return None
        y, mo, d = int(m2.group(3)), int(m2.group(1)), int(m2.group(2))
    else:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return pd.Timestamp(year=y, month=mo, day=d).strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_time(s) -> tuple[str | None, str | None]:
    """→ (HH:MM[:SS], 정밀도). 해석할 수 없으면 (None, None)."""
    if s is None or pd.isna(s) or str(s).strip() == "":
        return None, None
    t = str(s).strip()
    pm = bool(re.search(r"오후|pm", t, re.IGNORECASE))
    am = bool(re.search(r"오전|am", t, re.IGNORECASE))
    t = re.sub(r"오전|오후|am|pm", "", t, flags=re.IGNORECASE).strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", t) or re.match(r"^(\d{2})(\d{2})(\d{2})?$", t)
    if not m:
        return None, None
    h, mi, sec = int(m.group(1)), int(m.group(2)), m.group(3)
    if pm and h < 12:
        h += 12
    if am and h == 12:
        h = 0
    if not (0 <= h <= 23 and 0 <= mi <= 59) or (sec is not None and not 0 <= int(sec) <= 59):
        return None, None
    if sec is not None:
        return f"{h:02d}:{mi:02d}:{int(sec):02d}", "second"
    return f"{h:02d}:{mi:02d}", "minute"


def split_datetime(s) -> tuple[str | None, str | None]:
    if s is None or pd.isna(s):
        return None, None
    t = str(s).strip()
    m = re.match(r"^(.*?\d{1,2}일?)[\sT]+(.+)$", t)
    if m and parse_date(m.group(1)):
        return m.group(1), m.group(2)
    return t, None


def parse_amount(s) -> tuple[float | None, str | None, bool]:
    """→ (절댓값 금액, 기호에서 찾은 통화, 음수 여부)."""
    if s is None or pd.isna(s):
        return None, None, False
    t = str(s).strip()
    cur = None
    for sym, code in sorted(CURRENCY_SYMBOL.items(), key=lambda x: -len(x[0])):
        if sym in t:
            cur = code
            break
    code = re.search(r"\b(KRW|USD|JPY|EUR|CNY|GBP)\b", t, re.IGNORECASE)
    if code:
        cur = code.group(1).upper()
    neg = t.startswith("-") or (t.startswith("(") and t.endswith(")")) or "-" in t[:3]
    num = re.sub(r"[^\d.]", "", t)
    if num.count(".") > 1 or num in ("", "."):
        return None, cur, neg
    return float(num), cur, neg


def parse_type(s) -> str | None:
    if s is None or pd.isna(s):
        return None
    t = str(s).lower()
    if re.search(r"온라인|인터넷|online|web|앱|모바일|간편결제", t):
        return "online"
    if re.search(r"오프라인|offline|현장|매장|대면", t):
        return "offline"
    return None


def extract_region(address) -> str | None:
    if address is None or pd.isna(address):
        return None
    a = " " + str(address).strip() + " "
    for name, pat in REGIONS:
        if re.search(pat, a):
            return name
    return None


def normalize(raw: pd.DataFrame, mapping: dict, default_currency: str = "KRW",
              default_card: str = "내 카드") -> tuple[pd.DataFrame, pd.DataFrame]:
    """→ (공통 스키마 거래, 오류 행[원본 행번호·사유·원본 값]). 오류 행은 저장하지 않는다."""
    ok, errors = [], []
    for i, r in raw.iterrows():
        get = lambda k: r[mapping[k]] if mapping.get(k) else None
        date_raw, time_raw = get("date"), get("time")
        if mapping.get("datetime"):
            date_raw, t2 = split_datetime(get("datetime"))
            time_raw = time_raw if time_raw is not None else t2
        reasons = []
        date = parse_date(date_raw)
        if not date:
            reasons.append("날짜 없음" if _blank(date_raw) else "날짜 형식 오류")
        tm, tprec = parse_time(time_raw)
        if not _blank(time_raw) and tm is None:
            reasons.append("시각 형식 오류")
        amount, cur_sym, neg = parse_amount(get("amount"))
        if amount is None:
            reasons.append("금액 형식 오류")
        cur_col = get("currency")
        currency = (str(cur_col).strip().upper() if cur_col is not None and not pd.isna(cur_col) else None) or cur_sym or default_currency
        if currency not in CURRENCIES:
            reasons.append(f"지원하지 않는 통화({currency})")
        if reasons:
            errors.append({"row": i, "reason": ", ".join(reasons),
                           **{k: r[v] for k, v in mapping.items() if v}})
            continue
        status_raw = get("transaction_status")
        cancelled = neg or (status_raw is not None and not pd.isna(status_raw) and "취소" in str(status_raw))
        address = get("merchant_address")
        address = None if address is None or pd.isna(address) or not str(address).strip() else str(address).strip()
        region = extract_region(address)
        merchant = get("merchant_name")
        merchant = None if merchant is None or pd.isna(merchant) or not str(merchant).strip() else str(merchant).strip()
        card = get("card_alias")
        card = str(card).strip() if card is not None and not pd.isna(card) and str(card).strip() else default_card
        prov = {"transaction_date": "user_input", "amount": "user_input"}
        if tm:
            prov.update(transaction_time="user_input", hour_bucket="derived")
        if region:
            prov["region"] = "derived"
        ok.append({
            "transaction_id": str(uuid.uuid4()),
            "source": "personal",
            "country": None,
            "card_id": card,
            "transaction_date": date,
            "transaction_time": tm,
            "hour_bucket": int(tm[:2]) if tm else None,
            "time_precision": tprec or "day",
            "amount": amount,
            "currency": currency,
            "merchant_name": merchant,
            "merchant_address": address,
            "region": region,
            "location_precision": "province" if region else "none",
            "transaction_type": parse_type(get("transaction_type")),
            "transaction_status": "cancelled" if cancelled else "approved",
            "is_fraud": None,
            "provenance": provenance(**prov),
        })
    cols = ["transaction_id"]
    good = conform(pd.DataFrame(ok)) if ok else conform(pd.DataFrame(columns=cols))
    return good, pd.DataFrame(errors)


def manual_entry(date, time_str, amount, currency, merchant=None, address=None, ttype=None,
                 card_alias=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame([{"date": str(date), "time": time_str or None, "amount": str(amount), "currency": currency,
                         "merchant_name": merchant, "merchant_address": address, "transaction_type": ttype,
                         "card_alias": card_alias}])
    mapping = {k: (k if k in raw.columns else None) for k in TARGETS}
    return normalize(raw, mapping, default_currency=currency)


def refresh_derived(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """편집 화면에서 고친 행의 파생 필드(시간대·정밀도·시도)를 다시 계산한다. → (행, 오류 문장)."""
    out = df.copy()
    problems = []
    for i, r in out.iterrows():
        d = parse_date(r["transaction_date"])
        if not d:
            problems.append(f"{i + 1}행: 날짜 형식 오류")
            continue
        out.at[i, "transaction_date"] = d
        tm, prec = parse_time(r["transaction_time"])
        if not _blank(r["transaction_time"]) and tm is None:
            problems.append(f"{i + 1}행: 시각 형식 오류")
            continue
        out.at[i, "transaction_time"] = tm
        out.at[i, "hour_bucket"] = int(tm[:2]) if tm else None
        out.at[i, "time_precision"] = prec or "day"
        addr = r["merchant_address"]
        addr = None if addr is None or pd.isna(addr) or not str(addr).strip() else str(addr).strip()
        out.at[i, "merchant_address"] = addr
        region = extract_region(addr)
        out.at[i, "region"] = region
        out.at[i, "location_precision"] = "province" if region else "none"
    return conform(out), problems


# ---------------------------------------------------------------- 중복
def dup_key(df: pd.DataFrame) -> pd.Series:
    merchant = df["merchant_name"].fillna("").astype(str).str.replace(r"\s", "", regex=True).str.lower()
    return (df["card_id"].astype(str) + "|" + df["transaction_date"].astype(str) + "|"
            + df["transaction_time"].fillna("").astype(str) + "|" + df["amount"].round(2).astype(str)
            + "|" + df["currency"].astype(str) + "|" + merchant + "|" + df["transaction_status"].fillna("").astype(str))


def find_duplicates(new: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.Series:
    """True = 파일 안에서 앞 행과 겹치거나 이미 저장된 거래와 겹침 (중복 의심)."""
    key = dup_key(new)
    dup = key.duplicated(keep="first")
    if existing is not None and len(existing):
        dup |= key.isin(set(dup_key(existing)))
    return dup
