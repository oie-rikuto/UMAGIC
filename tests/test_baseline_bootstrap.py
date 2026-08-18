"""回収率のブートストラップ信頼区間（`docs/spec/005-baseline.md` 5節 / `D-078`）。"""

from __future__ import annotations

import math

import polars as pl

from umagic.baseline import bootstrap_roi_ci


def _ledger(rows: list[tuple[int, int]]) -> pl.DataFrame:
    """`rows` は `(stake_yen, payout_yen)` のリスト。"""
    return pl.DataFrame(
        {
            "race_id": list(range(len(rows))),
            "n_bets": [1] * len(rows),
            "n_hits": [1 if p > 0 else 0 for _, p in rows],
            "stake_yen": [s for s, _ in rows],
            "payout_yen": [p for _, p in rows],
        }
    )


def test_single_race_ci_degenerates_to_point_value():
    """観点13: 1レースだけのブートストラップはCIが縮退しても例外を出さない。"""
    ledger = _ledger([(100, 250)])
    lo, hi = bootstrap_roi_ci(ledger, bootstrap_n=200, seed=1)
    assert lo == hi == 2.5


def test_empty_ledger_returns_nan():
    ledger = _ledger([])
    lo, hi = bootstrap_roi_ci(ledger, bootstrap_n=200, seed=1)
    assert math.isnan(lo) and math.isnan(hi)


def test_same_seed_is_bit_exact():
    """観点10: 同じ `seed` で2回実行してもビット完全一致する（`R-021`）。"""
    ledger = _ledger([(100, 0), (100, 250), (100, 100), (100, 0), (100, 500)] * 20)
    result1 = bootstrap_roi_ci(ledger, bootstrap_n=300, seed=42)
    result2 = bootstrap_roi_ci(ledger, bootstrap_n=300, seed=42)
    assert result1 == result2


def test_different_seed_can_differ():
    # 払戻を連番にして percentile の境界が偶然一致しないようにする
    ledger = _ledger([(100, p) for p in range(0, 10000, 37)])
    result1 = bootstrap_roi_ci(ledger, bootstrap_n=300, seed=1)
    result2 = bootstrap_roi_ci(ledger, bootstrap_n=300, seed=2)
    assert result1 != result2


def test_ci_bounds_are_ordered():
    ledger = _ledger([(100, 0), (100, 250), (100, 100), (100, 0), (100, 500)] * 20)
    lo, hi = bootstrap_roi_ci(ledger, bootstrap_n=500, seed=7)
    assert lo <= hi
