"""`F-701` 騎手の実力（`docs/spec/003-features.md` / `D-002` / `Q-006`）。

過去レースの人気を使う（対象レースの人気・オッズは使わない、`D-002`の
`F-701`/`F-703` に限った例外）。「人気帯」は人気の値そのもの（1番人気、
2番人気…）を単位とする——区分の境界を新たに設けず、最も細かい粒度を
そのまま使う。

```
residual_i = finish_pos_i − E[finish_pos | popularity_i]   # 母集団は as_of 未満
F-701 = 縮約(mean(residual), n=騎乗数, k, μ_global)
```

`E[finish_pos | popularity]`・`μ_global` はいずれも対象行ごとにその行
自身の日付より前で計算する（`D-054`）。日付順の累積を持つことで、
対象行ごとの Python ループを避けている（`F-103`/`F-202` と同じ手法）。
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import polars as pl

from umagic.features.shrinkage import shrink

DEFAULT_K = 5.0  # 既定値は P-3 で決める（D-051）

_POPULATION_SQL = """
SELECT ru.race_id, ru.horse_id, ru.jockey_id, ru.popularity, ru.finish_pos, r.date
FROM runners ru JOIN races r USING (race_id)
WHERE ru.popularity IS NOT NULL AND ru.finish_pos IS NOT NULL
"""

_TARGET_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id, b.horse_id FROM base b
)
SELECT t.race_id, t.horse_id, r.date, ru.jockey_id
FROM targets t
JOIN races r ON r.race_id = t.race_id
JOIN runners ru ON ru.race_id = t.race_id AND ru.horse_id = t.horse_id
"""

_KEY_COLS = ["race_id", "horse_id"]


def _cumulative_before(df: pl.DataFrame, *, group_cols: list[str], value_col: str) -> pl.DataFrame:
    """`group_cols` ごとに、日付順の累積 `(sum, n)` を返す。`group_cols=[]` なら全体で1系列。"""
    if df.is_empty():
        schema = {c: pl.Int64 for c in group_cols}
        schema.update({"date": pl.Date, "cum_sum": pl.Float64, "cum_n": pl.Int64})
        return pl.DataFrame(schema=schema)
    daily = (
        df.group_by(group_cols + ["date"])
        .agg(pl.col(value_col).sum().alias("sum_v"), pl.len().alias("n_v"))
        .sort(group_cols + ["date"])
    )
    over_cols = group_cols if group_cols else None
    daily = daily.with_columns(
        (pl.col("sum_v").cum_sum().over(over_cols) if over_cols else pl.col("sum_v").cum_sum())
        .alias("cum_sum"),
        (pl.col("n_v").cum_sum().over(over_cols) if over_cols else pl.col("n_v").cum_sum())
        .alias("cum_n"),
    )
    return daily.select(group_cols + ["date", "cum_sum", "cum_n"])


def _lookup_before(
    daily: pl.DataFrame, probe_rows: pl.DataFrame, *, group_cols: list[str], key_cols: list[str],
) -> pl.DataFrame:
    """`probe_rows`（`group_cols` + `date` + `key_cols`）の各行について、
    その `date` **より前**（`D-054`）の累積 `(cum_sum, cum_n)` を返す。
    """
    if daily.is_empty():
        return probe_rows.select(key_cols).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("cum_sum"),
            pl.lit(None, dtype=pl.Int64).alias("cum_n"),
        )
    probe = probe_rows.with_columns(
        (pl.col("date") - timedelta(days=1)).alias("_asof_date")
    ).sort(group_cols + ["_asof_date"]) if group_cols else probe_rows.with_columns(
        (pl.col("date") - timedelta(days=1)).alias("_asof_date")
    ).sort("_asof_date")

    kwargs = {"by": group_cols} if group_cols else {}
    joined = probe.join_asof(
        daily.sort(group_cols + ["date"]) if group_cols else daily.sort("date"),
        left_on="_asof_date", right_on="date", strategy="backward", **kwargs,
    )
    return joined.select(key_cols + ["cum_sum", "cum_n"])


def compute_f701(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date, k: float = DEFAULT_K,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        population = conn.execute(_POPULATION_SQL).pl()
        targets = conn.execute(_TARGET_SQL).pl()
    finally:
        conn.unregister("base")

    # E[finish_pos | popularity]: 人気の値そのものを単位に、対象行(population自身)より前で計算
    pop_daily = _cumulative_before(population, group_cols=["popularity"], value_col="finish_pos")
    pop_probe = population.select(_KEY_COLS + ["popularity", "date"])
    expected = _lookup_before(pop_daily, pop_probe, group_cols=["popularity"], key_cols=_KEY_COLS)
    expected = expected.rename({"cum_sum": "exp_sum", "cum_n": "exp_n"})

    residual_all = population.join(expected, on=_KEY_COLS, how="left").with_columns(
        pl.when(pl.col("exp_n").is_not_null() & (pl.col("exp_n") > 0))
        .then(pl.col("finish_pos") - pl.col("exp_sum") / pl.col("exp_n"))
        .otherwise(None)
        .alias("residual")
    ).filter(pl.col("residual").is_not_null())

    # 騎手ごとの縮約前の生統計（対象行ごとにその日付より前）
    jockey_daily = _cumulative_before(
        residual_all.filter(pl.col("jockey_id").is_not_null()),
        group_cols=["jockey_id"], value_col="residual",
    )
    jockey_probe = targets.select(_KEY_COLS + ["jockey_id", "date"]).filter(
        pl.col("jockey_id").is_not_null()
    )
    jockey_looked_up = _lookup_before(
        jockey_daily, jockey_probe, group_cols=["jockey_id"], key_cols=_KEY_COLS,
    ).rename({"cum_sum": "f701_sum", "cum_n": "f701_n"})

    # μ_global: 人気帯で調整済みの残差の、条件なしの全体平均
    global_daily = _cumulative_before(residual_all, group_cols=[], value_col="residual")
    global_looked_up = _lookup_before(
        global_daily, targets.select(_KEY_COLS + ["date"]), group_cols=[], key_cols=_KEY_COLS,
    )
    global_looked_up = global_looked_up.with_columns(
        pl.when(pl.col("cum_n").is_not_null() & (pl.col("cum_n") > 0))
        .then(pl.col("cum_sum") / pl.col("cum_n"))
        .otherwise(0.0)  # 母集団が全く無い極初期のみ。残差の中立値は0
        .alias("mu_global")
    ).select(_KEY_COLS + ["mu_global"])

    out = targets.join(jockey_looked_up, on=_KEY_COLS, how="left")
    out = out.join(global_looked_up, on=_KEY_COLS, how="left")

    def _shrink_row(s: dict) -> float:
        n = s["f701_n"]
        if n is None or n <= 0:
            return s["mu_global"]
        return shrink(n, s["f701_sum"] / n, k=k, mu_global=s["mu_global"])

    out = out.with_columns(
        pl.struct(["f701_n", "f701_sum", "mu_global"])
        .map_elements(_shrink_row, return_dtype=pl.Float64)
        .alias("f701")
    )

    out = out.with_columns(
        pl.when(pl.col("jockey_id").is_null()).then(None).otherwise(pl.col("f701")).alias("f701"),
        pl.when(pl.col("jockey_id").is_null()).then(1).otherwise(0).alias("f701_unavailable"),
    )

    return out.select(_KEY_COLS + ["f701", "f701_unavailable"])
