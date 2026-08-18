"""封印セットの範囲計算（`docs/decisions.md` `D-017` / `D-056`）。

封印セットは「まだ開発に使っていない直近N年のG1」で、固定の年ではない。
ここでは範囲の計算のみを持つ。**何回開いたかの計上**（`D-017` が要求する
規律）は本モジュールの範囲外で、`010-backtest.md` 側の責務になる。
"""

from __future__ import annotations

from datetime import date


def sealed_range(today: date, n_years: int = 3) -> tuple[date, date]:
    """`today` から遡って `n_years` 年分の範囲 `[start, today]` を返す。"""
    try:
        start = today.replace(year=today.year - n_years)
    except ValueError:
        # 2/29 起点で n_years 後がうるう年でない場合
        start = today.replace(year=today.year - n_years, day=28)
    return start, today


def is_sealed(race_date: date, grade: str | None, *, today: date, n_years: int = 3) -> bool:
    """そのレースが封印セットに属するか。G1以外は対象外（`D-003`：学習データは封印しない）。"""
    if grade != "G1":
        return False
    start, end = sealed_range(today, n_years)
    return start <= race_date <= end
