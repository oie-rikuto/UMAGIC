"""`F-603` 馬体重（`docs/spec/003-features.md`）。

`runners.horse_weight` と `weight_diff` をそのまま出す。`horse_weight IS NULL`
（計不）では `NaN`・指示子 `1`。`race_level`: `False`。確定時刻は `当日`。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
SELECT ru.race_id, ru.horse_id, ru.horse_weight AS f603_horse_weight,
       ru.weight_diff AS f603_weight_diff
FROM base b
JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
"""


def compute_f603(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        df = conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")

    is_unavailable = pl.col("f603_horse_weight").is_null()
    return df.with_columns(
        pl.when(is_unavailable).then(1).otherwise(0).alias("f603_horse_weight_unavailable"),
        pl.when(is_unavailable).then(1).otherwise(0).alias("f603_weight_diff_unavailable"),
    )
