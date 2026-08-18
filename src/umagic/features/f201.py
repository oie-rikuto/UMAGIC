"""`F-201` / `F-903` カテゴリID（`docs/spec/003-features.md` / `D-062`）。

embedding は作らない。`sire_id` / `damsire_id` / `jockey_id` / `trainer_id`
のIDをそのまま出す。カテゴリの扱い（LightGBM のネイティブカテゴリ対応）は
`007-stage2-ranker.md` が決める。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
SELECT ru.race_id, ru.horse_id, h.sire_id, h.damsire_id, ru.jockey_id, ru.trainer_id
FROM base b
JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
JOIN horses h ON h.horse_id = ru.horse_id
"""


def compute_f201(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        return conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")
