"""`F-803` レース基礎情報（`docs/spec/003-features.md` / `D-028`）。

`distance` / `surface` / `direction` / `n_starters` / 季節 / `prize` /
`race_class` / `weight_rule`（`D-049`）。**天候・馬場状態を含めない**
（`F-804` に分離、`D-028`）。`race_level`: `True`。確定時刻は `木曜`。

季節は月から機械的に切る（3-5月=春, 6-8月=夏, 9-11月=秋, 12-2月=冬）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
SELECT
    b.race_id, b.horse_id,
    r.distance AS f803_distance,
    r.surface AS f803_surface,
    r.direction AS f803_direction,
    r.n_starters AS f803_n_starters,
    CASE
        WHEN MONTH(r.date) IN (3, 4, 5) THEN '春'
        WHEN MONTH(r.date) IN (6, 7, 8) THEN '夏'
        WHEN MONTH(r.date) IN (9, 10, 11) THEN '秋'
        ELSE '冬'
    END AS f803_season,
    r.prize AS f803_prize,
    r.race_class AS f803_race_class,
    r.weight_rule AS f803_weight_rule
FROM base b
JOIN races r ON r.race_id = b.race_id
"""


def compute_f803(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        return conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")
