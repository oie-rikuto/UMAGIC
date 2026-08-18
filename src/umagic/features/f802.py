"""`F-802` コース・距離適性（`docs/spec/003-features.md`）。

その馬自身の同コース・同距離帯（`D-066`）の過去成績（`D-067`）を縮約する。
`F-202`/`F-704` と同じ形（`(course, distance_band)` バケツごとの
`F-902` 縮約）で、条件バケツの対象エンティティが馬自身になる。
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import polars as pl

from umagic.features.shrinkage import shrink

DEFAULT_K = 5.0  # 既定値は P-3 で決める（D-051）
FALLBACK_MU_GLOBAL = 0.5  # 条件バケツに母集団が無い場合の中立値（finish_pos/n_starters の中央）

_DISTANCE_BAND_CASE = """
    CASE
        WHEN r.distance <= 1400 THEN '短距離'
        WHEN r.distance <= 1800 THEN 'マイル'
        WHEN r.distance <= 2200 THEN '中距離'
        ELSE '長距離'
    END
"""

_TARGET_SQL = f"""
SELECT b.race_id, b.horse_id, r.date, r.course,
       {_DISTANCE_BAND_CASE} AS distance_band
FROM base b
JOIN races r ON r.race_id = b.race_id
"""

_POPULATION_SQL = f"""
SELECT
    ru.horse_id, r.course, {_DISTANCE_BAND_CASE} AS distance_band, r.date,
    CAST(ru.finish_pos AS DOUBLE) / r.n_starters AS perf
FROM runners ru
JOIN races r USING (race_id)
WHERE ru.finish_pos IS NOT NULL
"""

_KEY_COLS = ["race_id", "horse_id"]


def _cumulative_before(pop: pl.DataFrame, *, group_cols: list[str]) -> pl.DataFrame:
    if pop.is_empty():
        schema = {c: pl.Int64 if c.endswith("_id") else pl.Utf8 for c in group_cols}
        schema.update({"date": pl.Date, "cum_sum": pl.Float64, "cum_n": pl.Int64})
        return pl.DataFrame(schema=schema)
    daily = (
        pop.group_by(group_cols + ["date"])
        .agg(pl.col("perf").sum().alias("sum_v"), pl.len().alias("n_v"))
        .sort(group_cols + ["date"])
    )
    daily = daily.with_columns(
        pl.col("sum_v").cum_sum().over(group_cols).alias("cum_sum"),
        pl.col("n_v").cum_sum().over(group_cols).alias("cum_n"),
    )
    return daily.select(group_cols + ["date", "cum_sum", "cum_n"])


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


def compute_f802(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date, k: float = DEFAULT_K,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    conn.register("base", base)
    try:
        targets = conn.execute(_TARGET_SQL).pl()
        population = conn.execute(_POPULATION_SQL).pl()
    finally:
        conn.unregister("base")

    targets = base.join(targets, on=_KEY_COLS, how="left")
    bucket_cols = ["course", "distance_band"]

    mu_daily = _cumulative_before(population.drop("horse_id"), group_cols=bucket_cols)
    mu_looked_up = _lookup_before(mu_daily, targets, group_cols=bucket_cols)
    mu_looked_up = mu_looked_up.with_columns(
        pl.when(pl.col("cum_n").is_not_null() & (pl.col("cum_n") > 0))
        .then(pl.col("cum_sum") / pl.col("cum_n"))
        .otherwise(FALLBACK_MU_GLOBAL)
        .alias("mu_global_c")
    ).select(_KEY_COLS + ["mu_global_c"])

    group_cols = ["horse_id"] + bucket_cols
    daily = _cumulative_before(population, group_cols=group_cols)
    sub_targets = targets.select(_KEY_COLS + bucket_cols + ["date"])
    looked_up = _lookup_before(daily, sub_targets, group_cols=group_cols).rename(
        {"cum_sum": "f802_sum", "cum_n": "f802_n"}
    )

    out = targets.join(mu_looked_up, on=_KEY_COLS, how="left")
    out = out.join(looked_up, on=_KEY_COLS, how="left")

    def _shrink_row(s: dict) -> float:
        n = s["f802_n"]
        if n is None or n <= 0:
            return s["mu_global_c"]
        return shrink(n, s["f802_sum"] / n, k=k, mu_global=s["mu_global_c"])

    out = out.with_columns(
        pl.struct(["f802_n", "f802_sum", "mu_global_c"])
        .map_elements(_shrink_row, return_dtype=pl.Float64)
        .alias("f802")
    )

    return out.select(_KEY_COLS + ["f802"])
