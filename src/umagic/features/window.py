"""過去走の集計窓（`docs/spec/003-features.md` 共通規約2 / `D-059`）。

過去走を集計する特徴量は、**全履歴**と**直近N走**の両方を列として出す。
`N` は特徴量ごとの設定値（既定値は `P-3` で決める）。

**入力の `history` は呼び出し側が原則2（`race_date < target_race_date`）で
フィルタ済みであること。** ここでは「全履歴か直近N件か」の窓だけを扱う。
"""

from __future__ import annotations

import polars as pl


def full_and_recent(
    history: pl.DataFrame, *, group_col: str | list[str], date_col: str, value_col: str, n: int,
) -> pl.DataFrame:
    """`group_col` ごとに全履歴と直近 `n` 件の `(count, mean)` を返す。

    `group_col` は単一列名、または複合キー（`["race_id", "horse_id"]` など）の
    列名リストを受け付ける。後者は「同じ馬でも対象行（`as_of` の基点）ごとに
    履歴が違う」ケース（`history` が対象行ごとに再構成済みのとき）に使う。

    戻り値の列: `group_col`, `n_all`, `{value_col}_all`, `n_recent`, `{value_col}_recent`。
    `F-902` の縮約にそのまま渡せるよう `n_*` も出す。標本が無い群は含まれない
    （呼び出し側が `base` に left join することで `NaN` になる）。
    """
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)

    ordered = history.sort(
        group_cols + [date_col], descending=[False] * len(group_cols) + [True]
    )
    ordered = ordered.with_columns(
        pl.int_range(1, pl.len() + 1).over(group_cols).alias("_rn")
    )

    agg_all = (
        ordered.group_by(group_cols)
        .agg(
            pl.len().alias("n_all"),
            pl.col(value_col).mean().alias(f"{value_col}_all"),
        )
    )
    agg_recent = (
        ordered.filter(pl.col("_rn") <= n)
        .group_by(group_cols)
        .agg(
            pl.len().alias("n_recent"),
            pl.col(value_col).mean().alias(f"{value_col}_recent"),
        )
    )
    return agg_all.join(agg_recent, on=group_cols, how="left")
