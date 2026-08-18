"""`F-503` 開催週次（`docs/spec/003-features.md` / `D-049` / `D-063`）。

`races.meeting_no` / `races.meeting_day` をそのまま出す。柵移動（A/B/Cコース）
は列として持たない（`D-063`。取得元が無い、`Q-027`）。`race_level`: `True`。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
SELECT ru.race_id, ru.horse_id, r.meeting_no AS f503_meeting_no, r.meeting_day AS f503_meeting_day
FROM base b
JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
JOIN races r ON r.race_id = ru.race_id
"""


def compute_f503(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        return conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")
