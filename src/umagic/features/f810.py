"""`F-810` 馬主ID（`D-165`、候補段階）。

`F-201` と同じ「生ID」パターン。embedding は作らず `owner_id` をそのまま
出す。カテゴリの扱い（LightGBM のネイティブカテゴリ対応、`cat_smooth`/
`cat_l2` 正則化）は `D-113` が既定を決めている。

**候補段階**: `FEATURE_FNS` にはまだ結線していない。OOF評価で見込みが
あれば `F-201` に統合するか、独立の特徴量として採用するかを判断する
（`D-166` 以降）。

`race_level`: `False`（馬ごとに異なる）。カテゴリ列のため相対化
（`F-901`）は適用しない——採用時は `CATEGORY_COLUMNS` に `f810_owner_id`
を加える。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
SELECT ru.race_id, ru.horse_id, ru.owner_id AS f810_owner_id
FROM base b
JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
"""


def compute_f810(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        return conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")
