"""`F-602` ローテーション（`docs/spec/003-features.md`）。

前走からの中週数（`(今走 date − 前走 date) / 7`）と、前走のグレード
（`races.grade`、非重賞なら `NULL`）を出す。

**放牧明け判定・前哨戦直行判定は実装しない。** `domain-knowledge.md` は
言及するが、`003-features.md` は中週数の式のみを与えており、放牧明けの
閾値（何週間空けば「明け」とするか）が仕様に無い。中週数そのものを
Stage 2 に渡せば閾値を暗黙に決め打ちせずに済む。

前走が無い馬では両方とも `NaN`・指示子 `1`。`race_level`: `False`。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id AS target_horse_id,
           r.date AS target_date
    FROM base b
    JOIN races r ON r.race_id = b.race_id
),
history AS (
    SELECT
        t.target_race_id, t.target_horse_id, t.target_date, hr.date AS past_date, hr.grade,
        ROW_NUMBER() OVER (
            PARTITION BY t.target_race_id, t.target_horse_id ORDER BY hr.date DESC
        ) AS rn
    FROM targets t
    JOIN runners ru ON ru.horse_id = t.target_horse_id
    JOIN races hr ON hr.race_id = ru.race_id AND hr.date < t.target_date
)
SELECT
    target_race_id AS race_id, target_horse_id AS horse_id,
    CAST(DATE_DIFF('day', past_date, target_date) AS DOUBLE) / 7.0 AS f602_weeks_since_last,
    grade AS f602_prev_grade
FROM history WHERE rn = 1
"""


def compute_f602(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    conn.register("base", base)
    try:
        prev = conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")

    out = base.join(prev, on=["race_id", "horse_id"], how="left")
    has_prev = pl.col("f602_weeks_since_last").is_not_null()
    return out.with_columns(
        pl.when(has_prev).then(0).otherwise(1).alias("f602_weeks_since_last_unavailable"),
        pl.when(has_prev).then(0).otherwise(1).alias("f602_prev_grade_unavailable"),
    )
