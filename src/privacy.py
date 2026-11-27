"""개인 거래 파일의 민감정보 탐지 (설계서 표 10).

- 카드번호: 13~19자리 숫자(공백·하이픈 제거) 중 Luhn 검사를 통과하는 값
- 주민등록번호 형태: YYMMDD-[1-4]XXXXXX
- 열 이름: CVC/CVV/비밀번호/유효기간/주민/카드번호 등
탐지 결과에는 '어느 열에서 몇 건'만 담고, 값 자체는 담지 않는다 (로그·화면 노출 방지).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

SENSITIVE_NAME = re.compile(
    r"cvc|cvv|cvn|비밀번호|password|\bpin\b|유효기간|expir|주민|resident|카드번호|card ?no\b|card ?number|cc_?num|\bpan\b",
    re.IGNORECASE,
)
DIGITS = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
RRN = re.compile(r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])-?[1-4]\d{6}(?!\d)")
MASKED_CARD = re.compile(r"\d{4}[- ]?[\d*xX]{4}[- ]?[\d*xX]{4}[- ]?\d{4}")


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class ScanResult:
    drop_columns: list[str] = field(default_factory=list)  # 저장 전에 반드시 제거할 열
    findings: list[str] = field(default_factory=list)  # 사용자에게 보여줄 안내 (값 없음)
    masked_card_columns: list[str] = field(default_factory=list)  # 마스킹된 카드번호 열 (별칭 후보)

    @property
    def clean(self) -> bool:
        return not self.drop_columns


def scan(df: pd.DataFrame, max_rows: int = 5000) -> ScanResult:
    res = ScanResult()
    sample = df.head(max_rows)
    for col in df.columns:
        name = str(col)
        if SENSITIVE_NAME.search(name):
            res.drop_columns.append(col)
            res.findings.append(f"'{name}' 열은 열 이름이 민감정보 항목에 해당해 제거합니다.")
            continue
        values = sample[col].dropna().astype(str)
        if values.empty:
            continue
        n_card = sum(any(luhn_ok(m) for m in DIGITS.findall(v)) for v in values)
        n_rrn = int(values.str.contains(RRN).sum())
        if n_card:
            res.drop_columns.append(col)
            res.findings.append(f"'{name}' 열에서 카드번호로 보이는 값 {n_card}건이 발견되어 열 전체를 제거합니다.")
        elif n_rrn:
            res.drop_columns.append(col)
            res.findings.append(f"'{name}' 열에서 주민등록번호 형태의 값 {n_rrn}건이 발견되어 열 전체를 제거합니다.")
        elif values.str.contains(MASKED_CARD).any():
            res.masked_card_columns.append(col)
            res.findings.append(f"'{name}' 열은 마스킹된 카드번호로 보입니다. 저장하지 않고 뒤 4자리만 카드 별칭으로 쓸 수 있습니다.")
    return res


def drop_sensitive(df: pd.DataFrame, res: ScanResult) -> pd.DataFrame:
    return df.drop(columns=res.drop_columns + res.masked_card_columns, errors="ignore")
