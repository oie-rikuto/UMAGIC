"""欠損の表現（`docs/spec/003-features.md` 共通規約5 / `D-058`）。

補完しない。値は `NaN` のまま渡し、**構造的に存在しないのか取得漏れなのか**
を示す指示子列を併設する。「構造的」の判定はドメイン知識そのもの
（例: `F-101` はコースに1角が無いこと）であり、特徴量ごとに異なるため、
呼び出し側が真偽値の式として渡す。
"""

from __future__ import annotations

import polars as pl


def with_unavailable_indicator(
    df: pl.DataFrame, value_col: str, *, is_structural: pl.Expr,
) -> pl.DataFrame:
    """`<value_col>_unavailable` を追加する。

    `value_col` が `null` かつ `is_structural` が真の行だけ `1`。それ以外
    （値がある、または取得漏れで `null`）は `0`（`003-features.md` の
    `F-101` 対応表: 過去走はあるが1角を含むものが無い→`1`、過去走が
    無い→`0`、という区別を一般化したもの）。
    """
    indicator_col = f"{value_col}_unavailable"
    return df.with_columns(
        pl.when(pl.col(value_col).is_null() & is_structural)
          .then(1)
          .otherwise(0)
          .alias(indicator_col)
    )
