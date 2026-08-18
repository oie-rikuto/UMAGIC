"""`F-501` 当日の脚質バイアス（`docs/spec/003-features.md` / `D-010`）。

対象レースより**前に発走した同日・同競馬場・同馬場（芝/ダート）**のレース
（`race_number <` の厳密不等号、`D-010`）の上位3着馬について、4角通過順
（`corners` の最後の要素）を `n_starters` で割った比率の平均を取る。

`F-501`/`F-502` のみが同日レースを参照できる例外規定（`domain-knowledge.md`
5節 原則3）に基づく。`race_level`: `True`（レース内で全馬共通）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id AS target_horse_id,
           r.date, r.course, r.surface, r.race_number
    FROM base b
    JOIN races r ON r.race_id = b.race_id
),
earlier_top3 AS (
    SELECT t.target_race_id, t.target_horse_id,
           CAST(list_extract(ru.corners, len(ru.corners)) AS DOUBLE) / hr.n_starters AS pos_ratio
    FROM targets t
    JOIN races hr ON hr.date = t.date AND hr.course = t.course AND hr.surface = t.surface
                  AND hr.race_number < t.race_number
    JOIN runners ru ON ru.race_id = hr.race_id AND ru.finish_pos <= 3
    WHERE ru.corners IS NOT NULL AND len(ru.corners) > 0
)
SELECT
    t.target_race_id AS race_id,
    t.target_horse_id AS horse_id,
    AVG(e.pos_ratio) AS f501,
    COUNT(e.pos_ratio) AS n_qualifying
FROM targets t
LEFT JOIN earlier_top3 e
  ON e.target_race_id = t.target_race_id AND e.target_horse_id = t.target_horse_id
GROUP BY t.target_race_id, t.target_horse_id
"""


def compute_f501(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        df = conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")

    # 先行レースが0本（1R/2Rなど）→ 構造的欠損
    return df.with_columns(
        pl.when(pl.col("n_qualifying") == 0).then(1).otherwise(0).alias("f501_unavailable")
    ).select(["race_id", "horse_id", "f501", "f501_unavailable"])
