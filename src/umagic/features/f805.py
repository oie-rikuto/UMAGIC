"""`F-805` 出走馬の基礎情報（`docs/spec/003-features.md` / `D-131`）。

馬齢（`runners.age`）、性別（`runners.sex`）、斤量（`runners.weight_carried`）。
いずれも取り込み済み・欠損ゼロの列だが、特徴量として使われていなかった
（`D-131`。`D-049` は斤量を特徴量にする目的で `weight_rule` をスキーマに
追加していたが、条件変数の側だけが `F-803` として実装され本体が抜けていた）。

**3列そろえて初めて意味を持つ。** 馬齢重量・定量では斤量が馬齢と性別の
関数であり、斤量だけでは「重い/軽い」の意味が定まらない。実測でも
`weight_carried` 単独は改善せず、3列まとめると改善した（`D-131`）。

`f805_sex` は文字列のカテゴリ列として出し、`D-092` の丸めに載せる
（`stage2._CATEGORY_COLS` / `orchestration.CATEGORY_COLUMNS` に登録済み）。
これにより `F-901`（レース内相対化）の対象からも自動的に外れる。

対象レース自身の出走表から取れる値であり、過去走の集約を伴わない。
したがってリークの経路が無い（`D-054` の原則は履歴集計に関するもの）。

`race_level`: `False`
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

# `f805_sex` は文字列のカテゴリ列として出し、`D-092` の丸め
# （`stage2._CATEGORY_COLS` / `orchestration.CATEGORY_COLUMNS`）に載せる。
# 実データの水準は3つ（牡 252,658 / 牝 192,676 / セ 19,430）。

_SQL = """
SELECT race_id, horse_id,
       CAST(age AS DOUBLE) AS f805_age,
       sex AS f805_sex,
       CAST(weight_carried AS DOUBLE) AS f805_weight_carried
FROM runners
WHERE race_id = ANY(?) AND status IN ('出走', '降着', '競走中止', '失格')
"""

_SCHEMA = {
    "race_id": pl.Int64, "horse_id": pl.Int64,
    "f805_age": pl.Float64, "f805_sex": pl.Utf8, "f805_weight_carried": pl.Float64,
}


def compute_f805(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    if base.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    race_ids = base["race_id"].unique().to_list()
    df = conn.execute(_SQL, [race_ids]).pl()
    if df.is_empty():
        return base.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("f805_age"),
            pl.lit(None, dtype=pl.Utf8).alias("f805_sex"),
            pl.lit(None, dtype=pl.Float64).alias("f805_weight_carried"),
        ).select(list(_SCHEMA.keys()))

    out = base.join(df, on=["race_id", "horse_id"], how="left")
    return out.select(list(_SCHEMA.keys()))
