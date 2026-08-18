"""`F-302` 補正タイムベースの能力値の接続点（`docs/spec/003-features.md` / `D-060`）。

本体は `013-track-variant.md` が実装する（未作成、`Q-025` 待ち）。ここでは
`003` が持つべき**接続規約のみ**を実装する。`013` 側の実装を待たずに
`P-1` を完結させるため、`013` が未実装の間は常に `horse_effects` が
空で渡される想定で、`f302` は全行 `NaN`・`f302_unavailable=1` になる。

**簡易版のフォールバックを置かない（`D-060`）。** 馬場差を引かない
補正タイムは、速い馬場で走った馬を過大評価する誤った信号になるため、
`NaN` のまま置くほうが安全という判断による。

`race_level`: `False`
"""

from __future__ import annotations

import polars as pl


def attach_f302(df: pl.DataFrame, horse_effects: pl.DataFrame) -> pl.DataFrame:
    """`df` に `f302` / `f302_unavailable` を追加して返す。

    `horse_effects`: `(horse_id, as_of, effect)` を持つ。`013-track-variant.md`
    の `F-301` 馬IDランダム効果の出力（`D-060`）。`df` は `horse_id` と
    `as_of` を持つこと。

    `horse_effects` との結合キーは `(horse_id, as_of)` の完全一致とする。
    より精緻な結合方法（直近の推定を使う asof 結合など）が要るかは `013`
    自体が未実装のため未定（`Q-025`）。ここでは仮の結合方法として書く。
    """
    if horse_effects.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("f302"),
            pl.lit(1).alias("f302_unavailable"),
        )

    joined = df.join(
        horse_effects.select(["horse_id", "as_of", pl.col("effect").alias("f302")]),
        on=["horse_id", "as_of"],
        how="left",
    )
    return joined.with_columns(
        pl.when(pl.col("f302").is_not_null()).then(0).otherwise(1).alias("f302_unavailable")
    )
