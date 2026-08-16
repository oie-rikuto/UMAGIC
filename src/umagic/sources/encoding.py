"""生バイト列の文字コード判定（netkeiba は euc-jp）。"""

from __future__ import annotations

import re

_CANDIDATES = ["euc-jp", "cp932", "utf-8"]


def decode_best(raw: bytes, header_charset: str | None) -> str:
    """文字コードを判定して復号する。

    HTTPヘッダまたは `<meta charset=...>` が明示していれば、それが復号に
    成功する限り最優先で使う。CJK統合漢字の範囲は非常に広く、誤った
    エンコーディングでも偶然「それらしい」文字列に化けて高スコアになる
    ことがある（別エンコーディングのバイト列を誤って解釈した結果が、
    たまたま漢字の Unicode 範囲に収まるケース）。宣言があるのにスコア
    競合で不採用にすると、まさにこの事故が起きる。宣言が無いときだけ、
    ひらがな・カタカナ・漢字の出現数をスコアにした推定にフォールバックする。
    """
    declared = []
    if header_charset:
        declared.append(header_charset)
    m = re.search(rb'charset=["\']?([\w-]+)', raw[:4096], re.I)
    if m:
        declared.append(m.group(1).decode("ascii", "ignore"))

    for enc in dict.fromkeys(c.lower() for c in declared if c):
        try:
            return raw.decode(enc, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue

    best, best_score = "", -1
    for enc in _CANDIDATES:
        try:
            text = raw.decode(enc, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue
        score = len(re.findall(r"[ぁ-んァ-ン一-龥]", text))
        if score > best_score:
            best, best_score = text, score
    if best_score < 0:
        best = raw.decode("utf-8", errors="replace")
    return best
