"""`F-808` 騎手×馬のコンビ成績（`docs/spec/003-features.md` / `D-140`）。

`(jockey_id, horse_id)` ペアでの過去の騎乗成績。

**`F-701`（騎手の一般実力）・`F-702`（乗り替わり、回数のみ）・`F-703`
（厩舎の勝負度）のいずれも、このコンビでの過去成績を持たない（`D-140`）。**
`F-702` は「何回乗ったか」を持つが「その騎乗の成績」を持たない。コンビ
での過去成績は他の行（そのペアが一致する過去レース）にまたがる集約で
あり、`jockey_id` の一般実力と馬自身の能力からは導出できない。

**尺度は相対着順（`finish_pos / n_starters`）にする。** `D-108`/`D-116`
（速度指数・ペース適性で標本の少ないエンティティの効果量が無制限に
発散した問題）を避けるため、元から `[0,1]` に有界な量を選んだ。

**「回数」は含めない（`D-140`）。** `F-702` の `f702_jockey_experience_count`
と重複し、足すと一貫して悪化する。

`race_level`: `False`
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

# コンビの過去騎乗（`pr.date < t.date`、`D-054` 原則7）を集約する
_SQL = """
WITH t AS (
    SELECT b.race_id, b.horse_id, r.date, ru.jockey_id AS tj
    FROM base b
    JOIN races r ON r.race_id = b.race_id
    JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
),
combo AS (
    SELECT t.race_id AS tid, t.horse_id AS thid,
           CAST(h.finish_pos AS DOUBLE) / pr.n_starters AS rel_finish,
           CASE WHEN h.finish_pos = 1 THEN 1.0 ELSE 0.0 END AS is_win
    FROM t
    JOIN runners h ON h.horse_id = t.horse_id AND h.jockey_id = t.tj
    JOIN races pr ON pr.race_id = h.race_id AND pr.date < t.date
    WHERE h.status IN ('出走', '降着') AND h.finish_pos IS NOT NULL
)
SELECT tid AS race_id, thid AS horse_id,
       AVG(rel_finish) AS f808_relfinish_mean,
       MIN(rel_finish) AS f808_relfinish_best,
       AVG(is_win)     AS f808_win_rate,
       CAST(COUNT(*) AS DOUBLE) AS n_past_combo
FROM combo GROUP BY 1, 2
"""

_SCHEMA = {
    "race_id": pl.Int64, "horse_id": pl.Int64,
    "f808_relfinish_mean": pl.Float64, "f808_relfinish_best": pl.Float64,
    "f808_win_rate": pl.Float64, "f808_unavailable": pl.Int32,
}


def compute_f808(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。

    コンビでの騎乗歴が無い行（初コンビ、全体の過半数）は全列欠損になる。
    これは構造的欠損ではない（`D-058`）ため `f808_unavailable=0` のまま
    `NaN` にする——「初コンビ」自体は正しい情報で、LightGBM のネイティブ
    な欠損分岐に委ねる。
    """
    base = base.select(["race_id", "horse_id"])
    if base.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    conn.register("base", base)
    try:
        agg = conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")

    out = base.join(agg.drop("n_past_combo"), on=["race_id", "horse_id"], how="left")
    return out.with_columns(pl.lit(0, dtype=pl.Int32).alias("f808_unavailable")).select(
        list(_SCHEMA.keys())
    )
