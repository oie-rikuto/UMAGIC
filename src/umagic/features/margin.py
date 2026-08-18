"""着差(`runners.margin`)の数値化（`docs/spec/003-features.md` 共通規約7 / `D-064`）。"""

from __future__ import annotations

import re

_FRACTION_RE = re.compile(r"^(?:(\d+)\.)?(\d+)/(\d+)$")

_FIXED = {
    "ハナ": 0.05,
    "アタマ": 0.1,
    "クビ": 0.2,
    "大": 10.0,
    "同着": 0.0,
}


def parse_margin(text: str | None) -> float | None:
    """`D-064` の対応表に従い、着差の表記を馬身単位の数値に変換する。

    未知の表記・空文字・`None` は `None`（欠損）を返す。
    """
    if text is None:
        return None
    text = text.strip()
    if text == "":
        return None
    if text in _FIXED:
        return _FIXED[text]

    m = _FRACTION_RE.match(text)
    if m:
        whole = int(m.group(1)) if m.group(1) else 0
        numerator = int(m.group(2))
        denominator = int(m.group(3))
        if denominator == 0:
            return None
        return whole + numerator / denominator

    try:
        return float(text)
    except ValueError:
        return None
