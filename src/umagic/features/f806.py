"""`F-806` 相手強度（`docs/spec/003-features.md` / `D-135`）。

対象馬の過去走で**同走した他馬の速度指数**（`F-304`）を集約する。

**モデルが原理的に構成できない情報である。** モデルは対象馬自身の過去成績
（`F-601`/`F-304`）を見ているが、その過去走で誰と走ったかは**別の行**にあり、
木がどう分岐しても到達できない。`D-132`（条件付き速度指数）が効かなかったのは
木が自力で交互作用を作れたためで、その対になる着眼（`D-135`）。

**「破った相手」（自分より下位に敗れた相手の平均・最速）は含めない（`D-135`）。**
実測で効かず、列を増やすほど悪化した。`F-601` の着順情報との重複が疑われる。

**過去オッズは使わない。** `Q-006` の暫定方針により過去人気の利用は
`F-701`（騎手）/`F-703`（厩舎）に限定されている。速度指数のみで構成する。

`race_level`: `False`
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from umagic.features.f304 import CLIP, compute_spd

# 対象馬の過去走（`pr.date < t.date`、`D-054` 原則7）で同走した他馬の
# 速度指数を集める。大差負けが平均を壊すため `F-304` と同じ ±CLIP で丸める
_SQL = f"""
WITH t AS (
    SELECT b.race_id, b.horse_id, r.date
    FROM base b JOIN races r ON r.race_id = b.race_id
),
mine AS (
    SELECT t.race_id AS tid, t.horse_id AS thid, ru.race_id AS pid
    FROM t
    JOIN runners ru ON ru.horse_id = t.horse_id
    JOIN races pr ON pr.race_id = ru.race_id AND pr.date < t.date
    WHERE ru.status IN ('出走', '降着')
),
opp AS (
    SELECT m.tid, m.thid, GREATEST(LEAST(s.spd, {CLIP}), -{CLIP}) AS spd_c
    FROM mine m
    JOIN runners o ON o.race_id = m.pid AND o.horse_id <> m.thid
    JOIN spd s ON s.race_id = o.race_id AND s.horse_id = o.horse_id
    WHERE o.status IN ('出走', '降着')
)
SELECT tid AS race_id, thid AS horse_id,
       AVG(spd_c)               AS f806_field_mean,
       CAST(COUNT(*) AS DOUBLE) AS f806_n
FROM opp GROUP BY 1, 2
"""

_SCHEMA = {
    "race_id": pl.Int64, "horse_id": pl.Int64,
    "f806_field_mean": pl.Float64, "f806_n": pl.Float64,
    "f806_unavailable": pl.Int32,
}


def compute_f806(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。

    過去走が無い（＝相手がいない）行は `f806_field_mean` が `NaN`・
    `f806_n=0` になる。**指示子は構造的欠損（過去走はあるが速度指数を
    持つ相手が1頭もいない）を区別する**（`D-058`）。
    """
    base = base.select(["race_id", "horse_id"])
    if base.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    conn.register("spd", compute_spd(conn))
    conn.register("base", base)
    try:
        agg = conn.execute(_SQL).pl()
        past = conn.execute("""
            WITH t AS (
                SELECT b.race_id, b.horse_id, r.date
                FROM base b JOIN races r ON r.race_id = b.race_id
            )
            SELECT t.race_id, t.horse_id, COUNT(*) AS n_past_starts
            FROM t
            JOIN runners ru ON ru.horse_id = t.horse_id
            JOIN races pr ON pr.race_id = ru.race_id AND pr.date < t.date
            WHERE ru.status IN ('出走', '降着')
            GROUP BY 1, 2
        """).pl()
    finally:
        conn.unregister("base")
        conn.unregister("spd")

    out = base.join(agg, on=["race_id", "horse_id"], how="left")
    out = out.join(past, on=["race_id", "horse_id"], how="left").with_columns(
        pl.col("n_past_starts").fill_null(0),
        pl.col("f806_n").fill_null(0.0),
    )
    is_structural = (pl.col("n_past_starts") > 0) & pl.col("f806_field_mean").is_null()
    out = out.with_columns(
        pl.when(pl.col("f806_field_mean").is_null() & is_structural)
        .then(1).otherwise(0).cast(pl.Int32).alias("f806_unavailable")
    )
    return out.select(list(_SCHEMA.keys()))
