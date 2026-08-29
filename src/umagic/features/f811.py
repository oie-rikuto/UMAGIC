"""`F-811` 配合ニック（父×母父の組み合わせ成績、候補段階、`D-167`）。

**着眼**: 血統理論でいう「ニックス」——特定の父×母父の組み合わせが、
それぞれ単体の平均を上回る産駒成績を出す現象。`F-201`（父ID・母父IDを
生カテゴリとして別々に渡す）・`F-202`（父ごと・母父ごとに別々の条件別
成績）のいずれも、**父と母父の組み合わせそのもの**は持たない。

**`D-136` の2条件**:

  (1) モデルが構成できない: 父×母父はともに高カーディナリティで、
      木が2つのカテゴリ列を順に分岐して同じ組み合わせに到達するには
      多くの分岐と十分な葉サンプルが要る。組み合わせは疎（多くの
      組み合わせが1〜数頭）であり、`D-113` の `cat_smooth` は単一
      カテゴリ列の正則化であって組み合わせの縮約ではない
  (2) 既存列と重複しない: `F-201`/`F-202` は父・母父を独立に扱い、
      組み合わせ効果（相互作用）を持たない

**縮約は `F-202` と同じ式 `θ=(n·x̄+k·μ)/(n+k)`（`D-051`）**。`F-202` は
行ごとの `map_elements` で計算するが、本モジュールはベクトル化した式を
直接使う（挙動は同じ、計算のみ高速）。組み合わせは父単体・母父単体
よりさらに疎なため、`D-108`/`D-116` の発散を避ける設計を踏襲する。
**条件バケツ（芝ダ・距離帯）では条件しない**——父×母父だけでも標本が
薄く、さらに割ると縮約が効かなくなる。

**候補段階**: `FEATURE_FNS` にはまだ結線していない。
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import polars as pl

DEFAULT_K = 5.0
FALLBACK_MU_GLOBAL = 0.5

_TARGET_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id, b.horse_id
    FROM base b
)
SELECT t.race_id, t.horse_id, r.date AS target_date, h.sire_id, h.damsire_id
FROM targets t
JOIN races r ON r.race_id = t.race_id
JOIN horses h ON h.horse_id = t.horse_id
"""

_POPULATION_SQL = """
SELECT h.sire_id, h.damsire_id, r.date,
       CAST(ru.finish_pos AS DOUBLE) / r.n_starters AS perf
FROM runners ru
JOIN races r USING (race_id)
JOIN horses h ON h.horse_id = ru.horse_id
WHERE ru.finish_pos IS NOT NULL AND h.sire_id IS NOT NULL AND h.damsire_id IS NOT NULL
"""

_KEY_COLS = ["race_id", "horse_id"]


def _cumulative_before(pop: pl.DataFrame, *, group_cols: list[str]) -> pl.DataFrame:
    daily = (
        pop.group_by(group_cols + ["date"])
        .agg(pl.col("perf").sum().alias("sum_v"), pl.len().alias("n_v"))
        .sort(group_cols + ["date"])
    )
    return daily.with_columns(
        pl.col("sum_v").cum_sum().over(group_cols).alias("cum_sum"),
        pl.col("n_v").cum_sum().over(group_cols).alias("cum_n"),
    ).select(group_cols + ["date", "cum_sum", "cum_n"])


def _lookup_before(daily: pl.DataFrame, targets: pl.DataFrame, *, group_cols: list[str]) -> pl.DataFrame:
    if daily.is_empty():
        return targets.select(_KEY_COLS).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("cum_sum"),
            pl.lit(None, dtype=pl.Int64).alias("cum_n"),
        )
    probe = targets.with_columns(
        (pl.col("date") - timedelta(days=1)).alias("_asof_date")
    ).sort(group_cols + ["_asof_date"])
    joined = probe.join_asof(
        daily.sort(group_cols + ["date"]),
        left_on="_asof_date", right_on="date", by=group_cols, strategy="backward",
    )
    return joined.select(_KEY_COLS + ["cum_sum", "cum_n"])


def compute_f811(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date, k: float = DEFAULT_K,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        targets = conn.execute(_TARGET_SQL).pl().rename({"target_date": "date"})
        population = conn.execute(_POPULATION_SQL).pl()
    finally:
        conn.unregister("base")

    # μ_global: 血統条件なしの母集団平均（`D-054` 原則7、対象行ごとに前日まで）。
    # `_cumulative_before`/`_lookup_before` は `group_by`/`over` に最低1キーを
    # 要求するため、定数キー `_g` を立てて「全体で1系列」を表現する
    mu_daily = _cumulative_before(population.with_columns(pl.lit(1).alias("_g")),
                                  group_cols=["_g"])
    mu_probe = targets.with_columns(pl.lit(1).alias("_g"))
    mu_looked_up = _lookup_before(mu_daily, mu_probe, group_cols=["_g"])
    mu_looked_up = mu_looked_up.with_columns(
        pl.when(pl.col("cum_n").is_not_null() & (pl.col("cum_n") > 0))
        .then(pl.col("cum_sum") / pl.col("cum_n"))
        .otherwise(FALLBACK_MU_GLOBAL)
        .alias("mu_global")
    ).select(_KEY_COLS + ["mu_global"])

    # ニック（父×母父）の累積
    nick_pop = population.filter(
        pl.col("sire_id").is_not_null() & pl.col("damsire_id").is_not_null()
    )
    nick_daily = _cumulative_before(nick_pop, group_cols=["sire_id", "damsire_id"])
    nick_targets = targets.select(_KEY_COLS + ["sire_id", "damsire_id", "date"])
    nick_looked_up = _lookup_before(nick_daily, nick_targets, group_cols=["sire_id", "damsire_id"])

    out = (
        targets.select(_KEY_COLS + ["sire_id", "damsire_id"])
        .join(mu_looked_up, on=_KEY_COLS, how="left")
        .join(nick_looked_up.rename({"cum_sum": "nick_sum", "cum_n": "nick_n"}),
              on=_KEY_COLS, how="left")
    )

    out = out.with_columns(pl.col("nick_n").fill_null(0))
    out = out.with_columns(
        pl.when(pl.col("nick_n") > 0)
        .then((pl.col("nick_n") * (pl.col("nick_sum") / pl.col("nick_n"))
               + k * pl.col("mu_global")) / (pl.col("nick_n") + k))
        .otherwise(pl.col("mu_global"))
        .alias("f811_nick")
    )

    is_structural = pl.col("sire_id").is_null() | pl.col("damsire_id").is_null()
    out = out.with_columns(
        pl.when(is_structural).then(None).otherwise(pl.col("f811_nick")).alias("f811_nick"),
        pl.when(is_structural).then(1).otherwise(0).alias("f811_nick_unavailable"),
        pl.col("nick_n").cast(pl.Float64).alias("f811_nick_n"),
    )

    return out.select(["race_id", "horse_id", "f811_nick", "f811_nick_n", "f811_nick_unavailable"])
