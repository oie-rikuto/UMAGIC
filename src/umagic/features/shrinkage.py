"""縮約（`docs/spec/003-features.md` 共通規約3 / `D-051`）。

すべての集計特徴量に例外なく適用する。`n` が少ない標本ほど `μ_global` に
強く寄せる。`D-051` により階層ベイズは使わず、この単純式に統一する。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl


def shrink(n: float | None, x_bar: float | None, *, k: float, mu_global: float) -> float:
    """`θ = (n·x̄ + k·μ_global) / (n+k)`。標本が無ければ `μ_global` そのもの。"""
    if n is None or x_bar is None or n <= 0:
        return mu_global
    return (n * x_bar + k * mu_global) / (n + k)


def per_row_stat_before(
    conn: duckdb.DuckDBPyConnection,
    base: pl.DataFrame,
    *,
    race_dates: dict[int, date],
    stat_sql: str,
) -> pl.DataFrame:
    """`base` の各 `(race_id, horse_id)` について、**その行自身の対象レース
    日付より前**のデータだけで `stat_sql` を実行し、結果を1列で返す。

    `μ_global` のような共有統計量は、`build_features` 呼び出し単位の
    `as_of` で一括に切ってはならない（`D-054` の追記事項）。この関数は
    その正しい形（対象行ごとの再計算）を一箇所にまとめたもので、
    `F-103` `F-202` `F-303` `F-701` など `μ_global` を使う特徴量が共通で使う。

    `stat_sql` はプレースホルダを1つ持ち、`date` 型のパラメータを1つ
    受け取るクエリで、単一のスカラー値を返すこと（例:
    `"SELECT AVG(ru.time_sec) FROM runners ru JOIN races r USING (race_id) WHERE r.date < ?"`）。

    行数ぶんクエリを発行するため、大規模なバッチでは呼び出し側で
    日付ごとにまとめる等の最適化が要る。ここでは正しさを優先する。
    """
    rows = []
    for race_id, horse_id in base.select(["race_id", "horse_id"]).iter_rows():
        d = race_dates[race_id]
        value = conn.execute(stat_sql, [d]).fetchone()[0]
        rows.append((race_id, horse_id, value))
    return pl.DataFrame(rows, schema=["race_id", "horse_id", "stat"], orient="row")
