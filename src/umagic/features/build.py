"""特徴量生成のエントリポイント（`docs/spec/003-features.md` / `004-leakage-test.md`）。

`build_features` は `(race_id, horse_id)` を基底集合とし、登録済みの
`FeatureFn` を1つずつ左結合していく。**集計順序を決定的にする**（`D-055`）
ため、基底集合は常に `(race_id, horse_id)` で整列してから返す。

`race_ids` を指定しないときの対象レースは `date < as_of`。**まだ結果が
確定していない、あるいは as_of 時点で未来のレースを既定の対象に含めない**
ため。特定レース（推論対象）を明示的に含めたいときは `race_ids` で渡す。

基底集合は `出走取消`/`競走除外` を除く（`build_labels()` と同じ条件、
`D-093`/`D-094`）。含めると `F-901`（`relativize()`）の `_rank` 列が
`races.n_starters`（実出走頭数）より多い頭数でランキングされ、比率が
1.0 を超える（`D-109`）。
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import duckdb
import polars as pl

# `出走取消`/`競走除外` を除く（`D-093`/`D-094` の `build_labels()` と同じ
# 条件に揃える）。含めると `races.n_starters`（実出走頭数のみ）を分母に
# 使う `F-901`（`relativize()`）の `_rank` 列が、出走取消・競走除外の
# 馬をランキングに含めたまま計算され、1.0 を超える値になる（`D-109`）。
_RUNNING_STATUSES = "('出走', '降着', '競走中止', '失格')"

_BASE_SQL_BY_RACE_IDS = f"""
    SELECT ru.race_id, ru.horse_id
    FROM runners ru
    JOIN races r USING (race_id)
    WHERE r.race_id = ANY(?) AND ru.status IN {_RUNNING_STATUSES}
    ORDER BY ru.race_id, ru.horse_id
"""

_BASE_SQL_BY_AS_OF = f"""
    SELECT ru.race_id, ru.horse_id
    FROM runners ru
    JOIN races r USING (race_id)
    WHERE r.date < ? AND ru.status IN {_RUNNING_STATUSES}
    ORDER BY ru.race_id, ru.horse_id
"""


class FeatureFn(Protocol):
    """1つの特徴量（または特徴量群）を計算する関数の形。

    `base` は `(race_id, horse_id)` の基底集合（決定的に整列済み）。
    戻り値は `base` のキーの**部分集合または全体**に対する追加列を持つ
    `DataFrame` で、`base` に left join される。計算できない行は
    キーを含めないか、値を `null` にする（`D-058`。行を落とさない）。
    """

    def __call__(
        self, conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
    ) -> pl.DataFrame: ...


def _base_population(
    conn: duckdb.DuckDBPyConnection, *, as_of: date, race_ids: list[int] | None,
) -> pl.DataFrame:
    if race_ids is not None:
        rows = conn.execute(_BASE_SQL_BY_RACE_IDS, [race_ids]).fetchall()
    else:
        rows = conn.execute(_BASE_SQL_BY_AS_OF, [as_of]).fetchall()
    # DuckDB から取得済みの時点で ORDER BY 済みだが、決定性を実装の
    # 詳細（SQL側の並び）に委ねきらないよう、ここでも明示的に整列する（D-055）
    df = pl.DataFrame(rows, schema=["race_id", "horse_id"], orient="row")
    return df.sort(["race_id", "horse_id"])


def build_features(
    conn: duckdb.DuckDBPyConnection,
    *,
    as_of: date,
    race_ids: list[int] | None = None,
    feature_fns: list[FeatureFn] | None = None,
) -> pl.DataFrame:
    """`(race_id, horse_id)` で一意な特徴量行列を返す。

    `feature_fns` は呼び出し側が渡す（`003-features.md` の各 `F-xxx` 実装）。
    骨格の時点では空でよい。
    """
    df = _base_population(conn, as_of=as_of, race_ids=race_ids)

    for fn in feature_fns or []:
        cols = fn(conn, df.select(["race_id", "horse_id"]), as_of=as_of)
        df = df.join(cols, on=["race_id", "horse_id"], how="left")

    # 結合後も基底の順序を保証する（列の追加順に依存しない）
    return df.sort(["race_id", "horse_id"])
