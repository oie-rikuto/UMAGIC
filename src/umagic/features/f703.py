"""`F-703` 厩舎の勝負度（`docs/spec/003-features.md` / `D-049` / `D-069`）。

遠征フラグは `affiliation`（美浦=`東`/栗東=`西`）と `races.course` の
対応表（`D-069`）で判定する。`地`/`外`/`NULL` は判定できず `NaN`・指示子`1`。

主戦騎手フラグは、対象馬の所属厩舎（`trainer_id`）で最も騎乗数の多い
騎手が、対象レースの騎手と一致するかを見る（`as_of` 未満のデータで判定）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

# D-069: 美浦(東)/栗東(西) と開催競馬場の対応
_HOME_COURSES = {
    "東": {"東京", "中山", "新潟", "福島", "札幌", "函館"},
    "西": {"阪神", "京都", "中京", "小倉"},
}

_TARGET_SQL = """
SELECT b.race_id, b.horse_id, r.course, ru.affiliation, ru.trainer_id, ru.jockey_id,
       r.date AS target_date
FROM base b
JOIN races r ON r.race_id = b.race_id
JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
"""

_MAIN_JOCKEY_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id AS target_horse_id,
           r.date AS target_date, ru.trainer_id
    FROM base b
    JOIN races r ON r.race_id = b.race_id
    JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
    WHERE ru.trainer_id IS NOT NULL
),
mounts AS (
    SELECT t.target_race_id, t.target_horse_id, hru.jockey_id, COUNT(*) AS n_mounts
    FROM targets t
    JOIN runners hru ON hru.trainer_id = t.trainer_id
    JOIN races hr ON hr.race_id = hru.race_id AND hr.date < t.target_date
    WHERE hru.jockey_id IS NOT NULL
    GROUP BY t.target_race_id, t.target_horse_id, hru.jockey_id
),
ranked AS (
    SELECT target_race_id, target_horse_id, jockey_id, n_mounts,
           ROW_NUMBER() OVER (
               PARTITION BY target_race_id, target_horse_id ORDER BY n_mounts DESC, jockey_id
           ) AS rn
    FROM mounts
)
SELECT target_race_id AS race_id, target_horse_id AS horse_id, jockey_id AS main_jockey_id
FROM ranked WHERE rn = 1
"""


def _travel_flag(affiliation: str | None, course: str | None) -> int | None:
    if affiliation not in _HOME_COURSES:
        return None
    return 0 if course in _HOME_COURSES[affiliation] else 1


def compute_f703(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    conn.register("base", base)
    try:
        targets = conn.execute(_TARGET_SQL).pl()
        main_jockey = conn.execute(_MAIN_JOCKEY_SQL).pl()
    finally:
        conn.unregister("base")

    targets = targets.with_columns(
        pl.struct(["affiliation", "course"])
        .map_elements(
            lambda s: _travel_flag(s["affiliation"], s["course"]), return_dtype=pl.Int8,
        )
        .alias("f703_travel_flag")
    )
    targets = targets.with_columns(
        pl.when(pl.col("f703_travel_flag").is_null()).then(1).otherwise(0)
        .alias("f703_travel_flag_unavailable")
    )

    out = targets.join(main_jockey, on=["race_id", "horse_id"], how="left")
    has_main_jockey = pl.col("main_jockey_id").is_not_null()
    out = out.with_columns(
        pl.when(has_main_jockey)
        .then((pl.col("main_jockey_id") == pl.col("jockey_id")).cast(pl.Int8))
        .otherwise(None)
        .alias("f703_main_jockey_flag"),
        pl.when(has_main_jockey).then(0).otherwise(1).alias("f703_main_jockey_flag_unavailable"),
    )

    return out.select([
        "race_id", "horse_id",
        "f703_travel_flag", "f703_travel_flag_unavailable",
        "f703_main_jockey_flag", "f703_main_jockey_flag_unavailable",
    ])
