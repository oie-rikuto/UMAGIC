"""`F-101` 逃げ意欲スコア（`docs/spec/003-features.md`）。

過去走のうち **1角の通過順が存在するもの**（`corner_nos` に `1` を含む）
だけを使い、`corners[j] / n_starters`（`j` は1角に対応する添字）の
下位分位点を取る。3角・4角での代用は禁止（`D-043`）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from umagic.features.missing import with_unavailable_indicator

DEFAULT_QUANTILE = 0.25  # 既定の下位分位点。実際の値は P-3 で決める（003 未確定）

_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id, r.date AS target_date
    FROM base b
    JOIN races r ON r.race_id = b.race_id
),
-- 1角の有無を問わない、対象行より前の全過去走（「過去走が無い」を判定するため）
past_starts AS (
    SELECT t.target_race_id, t.horse_id, COUNT(*) AS n_past_starts
    FROM targets t
    JOIN runners ru ON ru.horse_id = t.horse_id
    JOIN races hr ON hr.race_id = ru.race_id AND hr.date < t.target_date
    GROUP BY t.target_race_id, t.horse_id
),
-- 1角の通過順が実際に採れる過去走のみ
history AS (
    SELECT
        t.target_race_id,
        t.horse_id,
        CAST(list_extract(ru.corners, list_position(hr.corner_nos, 1)) AS DOUBLE)
            / hr.n_starters AS pos_ratio
    FROM targets t
    JOIN runners ru ON ru.horse_id = t.horse_id
    JOIN races hr ON hr.race_id = ru.race_id AND hr.date < t.target_date
    WHERE hr.corner_nos IS NOT NULL
      AND list_position(hr.corner_nos, 1) IS NOT NULL
      AND ru.corners IS NOT NULL
)
SELECT
    t.target_race_id AS race_id,
    t.horse_id,
    quantile_cont(h.pos_ratio, ?) AS f101,
    COALESCE(p.n_past_starts, 0) AS n_past_starts,
    COUNT(h.pos_ratio) AS n_past_with_corner1
FROM targets t
LEFT JOIN past_starts p
  ON p.target_race_id = t.target_race_id AND p.horse_id = t.horse_id
LEFT JOIN history h
  ON h.target_race_id = t.target_race_id AND h.horse_id = t.horse_id
GROUP BY t.target_race_id, t.horse_id, p.n_past_starts
"""


def compute_f101(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
    quantile: float = DEFAULT_QUANTILE,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        df = conn.execute(_SQL, [quantile]).pl()
    finally:
        conn.unregister("base")

    # 003-features.md の対応表:
    #   1角を含む過去走が1走以上ある → 値, unavailable=0
    #   過去走はあるが1角を含むものが無い → NaN, unavailable=1（構造的）
    #   過去走が無い → NaN, unavailable=0
    # 「構造的」は「過去走はある（n_past_starts>0）のに1角を含む走が
    # 1つも無い（n_past_with_corner1==0）」で判定する
    is_structural = (pl.col("n_past_starts") > 0) & (pl.col("n_past_with_corner1") == 0)
    df = with_unavailable_indicator(df, "f101", is_structural=is_structural)
    return df.drop(["n_past_starts", "n_past_with_corner1"])
