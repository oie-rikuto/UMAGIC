"""`F-302` 接続点（`docs/spec/003-features.md` / `D-060`）。"""

from __future__ import annotations

from datetime import date

import polars as pl

from umagic.features.f302 import attach_f302


def _base():
    return pl.DataFrame({
        "race_id": [1, 1],
        "horse_id": [10, 20],
        "as_of": [date(2024, 1, 1), date(2024, 1, 1)],
    })


def test_empty_horse_effects_gives_nan_and_structural_indicator():
    """`013` 未実装（`horse_effects` が空）→ 全行 NaN・unavailable=1。フォールバックを置かない。"""
    out = attach_f302(_base(), pl.DataFrame(schema={"horse_id": pl.Int64, "as_of": pl.Date, "effect": pl.Float64}))
    assert out["f302"].is_null().all()
    assert out["f302_unavailable"].to_list() == [1, 1]


def test_matched_rows_get_value_and_indicator_zero():
    horse_effects = pl.DataFrame({
        "horse_id": [10], "as_of": [date(2024, 1, 1)], "effect": [1.5],
    })
    out = attach_f302(_base(), horse_effects).sort("horse_id")
    assert out["f302"].to_list() == [1.5, None]
    assert out["f302_unavailable"].to_list() == [0, 1]
