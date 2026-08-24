"""`F-302` 補正タイムベースの能力値の接続点（`docs/spec/003-features.md` / `D-060` / `D-107`）。

本体は `013-track-variant.md`（`src/umagic/track_variant.py`）が実装する。
ここでは `003` が持つべき**接続規約のみ**を実装する。`horse_effects` が
空で渡されれば（`013` を呼ばない経路、あるいは学習期間にG1が無い等で
効果が1件も無い場合）、`f302` は全行 `NaN`・`f302_unavailable=1` になる。

**簡易版のフォールバックを置かない（`D-060`）。** 馬場差を引かない
補正タイムは、速い馬場で走った馬を過大評価する誤った信号になるため、
`NaN` のまま置くほうが安全という判断による。

**現時点では馬効果のみを出力する（`D-107`）。** `domain-knowledge.md` が
併記する補正タイムの過去走集計（最高値・直近平均・トレンド）は未実装
（`Q-041`）。

`race_level`: `False`
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl


def attach_f302(df: pl.DataFrame, horse_effects: pl.DataFrame) -> pl.DataFrame:
    """`df` に `f302` / `f302_unavailable` を追加して返す。

    `horse_effects`: `(horse_id, as_of, effect)` を持つ。`013-track-variant.md`
    の `horse_effect_series()` の出力（`D-107`）。`(horse_id, as_of)` で
    一意、`as_of` 昇順に整列済みであること。

    `df` は `horse_id` と `date`（対象レースの日付）を持つこと。**結合は
    `(horse_id, as_of)` の完全一致ではなく as-of 結合**（`date` 未満で
    最も新しい `as_of` の推定値を引く）。`F-202` / `F-701` / `F-704` /
    `F-801` / `F-802` が `join_asof(by=...)` で使っている手法と同型
    （`D-107`）。`date` 未満で唯一つも `as_of` が無い場合は欠損になる。

    `as_of == date`（`date` 以降）は引かない（013 の `as_of` は推定に
    使ったデータの上限であり、対象行自身の日付と同じ `as_of` は対象行の
    情報が推定に混入している可能性を排除できないため）。探索キーを1日
    前にずらすことで「未満」を「以下」の asof 結合として表す
    （`F-103`/`F-202` と同じ手法）。
    """
    if horse_effects.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("f302"),
            pl.lit(1).alias("f302_unavailable"),
        )

    probe = df.with_columns(
        (pl.col("date") - timedelta(days=1)).alias("_asof_date")
    ).sort(["horse_id", "_asof_date"])
    joined = probe.join_asof(
        horse_effects.select(["horse_id", "as_of", pl.col("effect").alias("f302")]).sort(
            ["horse_id", "as_of"]
        ),
        left_on="_asof_date", right_on="as_of", by="horse_id", strategy="backward",
    ).drop(["_asof_date", "as_of"])

    return joined.with_columns(
        pl.when(pl.col("f302").is_not_null()).then(0).otherwise(1).alias("f302_unavailable")
    )
