"""`F-601` 反動リスク（`docs/spec/003-features.md` / `D-034` / `Q-032`）。

前走の着順（`finish_pos`）・着差（`margin`、`D-064`）・上がり3F順位
（`F-303` の `last3f_rank_in_race` と同じ算出元）を出す。「高強度だったか」
の指標（`domain-knowledge.md` の人気ベースの式）は `D-002` の過去人気利用
範囲との整合が未確定のため実装しない（`Q-032`）。

`exclude_relegated=True` で `status='降着'` の走を前走候補から除外できる
（`D-034`）。既定は `False`（`finish_pos` は降着後の公式着順であり、
それ自体は有効な値のため）。

前走が無い馬では3列とも `NaN`・指示子 `1`（`003-features.md`）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from umagic.features.f303 import _last3f_rank_all
from umagic.features.margin import parse_margin

_SQL_TEMPLATE = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id AS target_horse_id,
           r.date AS target_date
    FROM base b
    JOIN races r ON r.race_id = b.race_id
),
history AS (
    SELECT
        t.target_race_id, t.target_horse_id, ru.race_id AS past_race_id,
        ru.finish_pos, ru.margin,
        ROW_NUMBER() OVER (
            PARTITION BY t.target_race_id, t.target_horse_id ORDER BY hr.date DESC
        ) AS rn
    FROM targets t
    JOIN runners ru ON ru.horse_id = t.target_horse_id
    JOIN races hr ON hr.race_id = ru.race_id AND hr.date < t.target_date
    {exclude_clause}
)
SELECT target_race_id AS race_id, target_horse_id AS horse_id, past_race_id, finish_pos, margin
FROM history WHERE rn = 1
"""


def compute_f601(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
    exclude_relegated: bool = False,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    exclude_clause = "WHERE ru.status != '降着'" if exclude_relegated else ""
    sql = _SQL_TEMPLATE.format(exclude_clause=exclude_clause)

    conn.register("base", base)
    try:
        prev = conn.execute(sql).pl()
    finally:
        conn.unregister("base")

    rank_all = _last3f_rank_all(conn)
    prev = prev.join(
        rank_all, left_on=["past_race_id", "horse_id"], right_on=["race_id", "horse_id"],
        how="left",
    ).rename({"last_3f_rank": "f601_last3f_rank_prev"})

    prev = prev.with_columns(
        pl.col("finish_pos").cast(pl.Float64).alias("f601_finish_pos_prev"),
        pl.col("margin").map_elements(parse_margin, return_dtype=pl.Float64).alias("f601_margin_prev"),
        pl.lit(1).alias("_has_prev"),
    )

    out = base.join(
        prev.select([
            "race_id", "horse_id", "_has_prev",
            "f601_finish_pos_prev", "f601_margin_prev", "f601_last3f_rank_prev",
        ]),
        on=["race_id", "horse_id"], how="left",
    ).with_columns(pl.col("_has_prev").fill_null(0))

    # 前走が無い馬では3列とも NaN・指示子1（003-features.md。前走が無いこと自体が構造的欠損）
    for col in ["f601_finish_pos_prev", "f601_margin_prev", "f601_last3f_rank_prev"]:
        out = out.with_columns(
            pl.when(pl.col("_has_prev") == 0).then(1).otherwise(0).alias(f"{col}_unavailable")
        )

    return out.drop("_has_prev")
