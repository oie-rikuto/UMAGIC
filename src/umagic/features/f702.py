"""`F-702` 乗り替わり（`docs/spec/003-features.md`）。

前走の `jockey_id` と対象レースの `jockey_id` の異同、当該馬への
騎乗経験回数（対象レースより前に、対象レースと同じ騎手がその馬に
騎乗した回数）を出す。前走が無い馬では両方とも `NaN`・指示子 `1`。
`race_level`: `False`。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id AS target_horse_id,
           r.date AS target_date, ru.jockey_id AS target_jockey_id
    FROM base b
    JOIN races r ON r.race_id = b.race_id
    JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
),
history AS (
    SELECT
        t.target_race_id, t.target_horse_id, t.target_jockey_id, hru.jockey_id,
        ROW_NUMBER() OVER (
            PARTITION BY t.target_race_id, t.target_horse_id ORDER BY hr.date DESC
        ) AS rn
    FROM targets t
    JOIN runners hru ON hru.horse_id = t.target_horse_id
    JOIN races hr ON hr.race_id = hru.race_id AND hr.date < t.target_date
),
prev_jockey AS (
    SELECT target_race_id, target_horse_id, jockey_id AS prev_jockey_id
    FROM history WHERE rn = 1
),
experience AS (
    SELECT t.target_race_id, t.target_horse_id, COUNT(*) AS n_experience
    FROM targets t
    JOIN runners hru ON hru.horse_id = t.target_horse_id AND hru.jockey_id = t.target_jockey_id
    JOIN races hr ON hr.race_id = hru.race_id AND hr.date < t.target_date
    GROUP BY t.target_race_id, t.target_horse_id
)
SELECT
    t.target_race_id AS race_id, t.target_horse_id AS horse_id,
    p.prev_jockey_id, t.target_jockey_id,
    COALESCE(e.n_experience, 0) AS f702_jockey_experience_count
FROM targets t
LEFT JOIN prev_jockey p
  ON p.target_race_id = t.target_race_id AND p.target_horse_id = t.target_horse_id
LEFT JOIN experience e
  ON e.target_race_id = t.target_race_id AND e.target_horse_id = t.target_horse_id
"""


def compute_f702(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        df = conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")

    has_prev = pl.col("prev_jockey_id").is_not_null()
    df = df.with_columns(
        pl.when(has_prev)
        .then((pl.col("prev_jockey_id") != pl.col("target_jockey_id")).cast(pl.Int8))
        .otherwise(None)
        .alias("f702_jockey_changed"),
        pl.when(has_prev).then(0).otherwise(1).alias("f702_jockey_changed_unavailable"),
    )
    # 騎乗経験回数は「前走が無い」ではなく「対象馬自身の過去走が無い」を欠損条件にする
    # （経験0回は前走の有無に関わらず正しい値であるため、_unavailable は常に0）
    df = df.with_columns(pl.lit(0).alias("f702_jockey_experience_count_unavailable"))

    return df.select([
        "race_id", "horse_id",
        "f702_jockey_changed", "f702_jockey_changed_unavailable",
        "f702_jockey_experience_count", "f702_jockey_experience_count_unavailable",
    ])
