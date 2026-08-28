"""`F-809` 馬のキャリア成績率（`docs/spec/003-features.md` / `D-145`）。

馬自身の**通算成績率**（勝率・複勝率）。

**実装済み特徴量のどれもこれを持たなかった（`D-145`）。**
`f702_jockey_experience_count` は**騎手**の経験回数、`F-802` は**条件別**
（コース×距離帯）成績、`F-601` は**前走のみ**、`F-602` は中週数と前走
グレードのみ。馬の通算成績という情報カテゴリ自体が欠けていた。

**「経験量」（通算出走数・デビューからの経過日数）は含めない（`D-145`）。**
実測で3 fold中2 foldが有意に悪化した。`F-602`（中週数）や `F-805`（馬齢）
と情報が重なると見られる。`D-135`（`F-806` の「破った相手」）・`D-140`
（`F-808` の「回数」）と同じ構造——**関連情報をまとめて足すと、どれかが
既存列と被って毒になる。**

**尺度は率（`[0,1]` に有界）にする。** `D-108`/`D-116` が示した「標本の
少ないエンティティの推定値が無制限に発散する」問題を構造的に避ける。

`race_level`: `False`
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

# 通算成績は `pr.date < t.date` で厳密に切る（`D-054` 原則7）
_SQL = """
WITH t AS (
    SELECT b.race_id, b.horse_id, r.date
    FROM base b JOIN races r ON r.race_id = b.race_id
),
career AS (
    SELECT t.race_id AS tid, t.horse_id AS thid,
           CAST(COUNT(*) AS DOUBLE) AS n,
           SUM(CASE WHEN h.finish_pos = 1 THEN 1.0 ELSE 0.0 END) AS n_wins,
           SUM(CASE WHEN h.finish_pos <= 3 THEN 1.0 ELSE 0.0 END) AS n_top3
    FROM t
    JOIN runners h ON h.horse_id = t.horse_id
    JOIN races pr ON pr.race_id = h.race_id AND pr.date < t.date
    WHERE h.status IN ('出走', '降着') AND h.finish_pos IS NOT NULL
    GROUP BY 1, 2
)
SELECT tid AS race_id, thid AS horse_id,
       n_wins / n AS f809_win_rate,
       n_top3 / n AS f809_top3_rate
FROM career
"""

_SCHEMA = {
    "race_id": pl.Int64, "horse_id": pl.Int64,
    "f809_win_rate": pl.Float64, "f809_top3_rate": pl.Float64,
}


def compute_f809(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。

    初出走の馬（過去走が無い）は両列とも欠損になる。**指示子は置かない**
    ——「過去走が無い」は `F-601`/`F-602` の `_unavailable` が既に表して
    おり、重ねても寄与しない列が増えるだけである（`D-119` / `D-135`）。
    """
    base = base.select(["race_id", "horse_id"])
    if base.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    conn.register("base", base)
    try:
        agg = conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")

    return base.join(agg, on=["race_id", "horse_id"], how="left").select(list(_SCHEMA.keys()))
