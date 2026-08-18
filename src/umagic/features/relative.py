"""レース内相対化（`docs/spec/003-features.md` 共通規約4 / `F-901` / `D-021`）。

`race_level=False` の特徴量にのみ適用する。**呼び出し側が対象を選ぶ**
（本モジュールは `race_level` を知らない）。`race_level=True` の特徴量に
適用すると、`F-104 = F-102 × F-103` のようなレース単位変数を含む交互作用
が代数的に潰れる（`D-021`）。
"""

from __future__ import annotations

import polars as pl


def relativize(
    df: pl.DataFrame, value_col: str, *, race_id_col: str = "race_id",
    n_starters_col: str = "n_starters",
) -> pl.DataFrame:
    """`value_col` から `_z`（z-score）と `_rank`（順位/頭数）を追加する。

    `df` は `race_id_col` と `n_starters_col`（分母。`D-012` の実出走頭数）を
    持っていること。`race_std=0`（レース内で全馬同値）のとき `_z` は `NaN`。
    """
    z_col = f"{value_col}_z"
    rank_col = f"{value_col}_rank"

    race_mean = pl.col(value_col).mean().over(race_id_col)
    race_std = pl.col(value_col).std().over(race_id_col)

    return df.with_columns(
        pl.when(race_std == 0)
          .then(None)
          .otherwise((pl.col(value_col) - race_mean) / race_std)
          .alias(z_col),
        (pl.col(value_col).rank(method="average").over(race_id_col)
         / pl.col(n_starters_col)).alias(rank_col),
    )
