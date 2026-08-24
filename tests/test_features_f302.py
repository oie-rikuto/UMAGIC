"""`F-302` 接続点（`docs/spec/003-features.md` / `D-060` / `D-107`）。"""

from __future__ import annotations

from datetime import date

import polars as pl

from umagic.features.f302 import attach_f302


def _base():
    return pl.DataFrame({
        "race_id": [1, 1],
        "horse_id": [10, 20],
        "date": [date(2024, 1, 10), date(2024, 1, 10)],
    })


def test_empty_horse_effects_gives_nan_and_structural_indicator():
    """`horse_effects` が空 → 全行 NaN・unavailable=1。フォールバックを置かない（D-060）。"""
    out = attach_f302(_base(), pl.DataFrame(schema={"horse_id": pl.Int64, "as_of": pl.Date, "effect": pl.Float64}))
    assert out["f302"].is_null().all()
    assert out["f302_unavailable"].to_list() == [1, 1]


def test_matched_rows_get_value_and_indicator_zero():
    """`date` 未満の `as_of` が引かれる（D-107）。"""
    horse_effects = pl.DataFrame({
        "horse_id": [10], "as_of": [date(2024, 1, 1)], "effect": [1.5],
    })
    out = attach_f302(_base(), horse_effects).sort("horse_id")
    assert out["f302"].to_list() == [1.5, None]
    assert out["f302_unavailable"].to_list() == [0, 1]


def test_most_recent_as_of_before_date_is_used():
    """複数の `as_of` があるとき、`date` 未満で最も新しいものが引かれる（013 テスト観点12）。"""
    horse_effects = pl.DataFrame({
        "horse_id": [10, 10, 10],
        "as_of": [date(2023, 1, 1), date(2024, 1, 1), date(2024, 1, 9)],
        "effect": [-9.0, 1.5, 2.5],
    })
    out = attach_f302(_base(), horse_effects).sort("horse_id")
    assert out["f302"].to_list()[0] == 2.5


def test_as_of_on_or_after_date_is_not_used():
    """`date` と同じ、または `date` より後の `as_of` は引かれない（013 テスト観点12）。"""
    horse_effects = pl.DataFrame({
        "horse_id": [10, 10],
        "as_of": [date(2024, 1, 10), date(2024, 1, 20)],  # base の date は 2024-01-10
        "effect": [1.5, 2.5],
    })
    out = attach_f302(_base(), horse_effects).sort("horse_id")
    assert out["f302"].to_list()[0] is None
    assert out["f302_unavailable"].to_list()[0] == 1
