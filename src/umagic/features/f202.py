"""`F-202` 種牡馬の条件別成績（`docs/spec/003-features.md` / `D-066` / `D-067` / `D-068`）。

条件 `c` = (芝/ダート, 距離帯)。**馬場状態は含まない**（`D-068`: `F-2xx` の
締切は木曜だが、対象レースの馬場状態は当日まで分からないため）。父
（`sire_id`）・母父（`damsire_id`）に同じ形を適用する。

`μ_global,c` は条件バケツだけで見た母集団平均を、対象行ごとにその行自身の
レース日付より前で計算する（`D-054`）。日付順の累積を条件バケツ・血統
IDごとに持つことで、対象行ごとの Python ループを避けている（`F-103` と
同じ手法）。
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
WITH targets AS (
    SELECT DISTINCT b.race_id, b.horse_id
    FROM base b
)
SELECT t.race_id, t.horse_id, r.date AS target_date, r.surface,
       {_DISTANCE_BAND_CASE} AS distance_band,
       h.sire_id, h.damsire_id
FROM targets t
JOIN races r ON r.race_id = t.race_id
JOIN horses h ON h.horse_id = t.horse_id
"""

_POPULATION_SQL = f"""
SELECT
    h.sire_id, h.damsire_id, r.date, r.surface,
    {_DISTANCE_BAND_CASE} AS distance_band,
    CAST(ru.finish_pos AS DOUBLE) / r.n_starters AS perf
FROM runners ru
JOIN races r USING (race_id)
JOIN horses h ON h.horse_id = ru.horse_id
WHERE ru.finish_pos IS NOT NULL
"""

_KEY_COLS = ["race_id", "horse_id"]


def _cumulative_before(pop: pl.DataFrame, *, group_cols: list[str]) -> pl.DataFrame:
    """`group_cols` ごとに、日付順の累積 `(sum, n)` を返す。"""
    valid = pop.filter(pl.col(group_cols[0]).is_not_null())
    if valid.is_empty():
        schema = {c: pl.Int64 if c.endswith("_id") else pl.Utf8 for c in group_cols}
        schema.update({"date": pl.Date, "cum_sum": pl.Float64, "cum_n": pl.Int64})
        return pl.DataFrame(schema=schema)

    daily = (
        valid.group_by(group_cols + ["date"])
        .agg(pl.col("perf").sum().alias("sum_v"), pl.len().alias("n_v"))
        .sort(group_cols + ["date"])
    )
    daily = daily.with_columns(
        pl.col("sum_v").cum_sum().over(group_cols).alias("cum_sum"),
        pl.col("n_v").cum_sum().over(group_cols).alias("cum_n"),
    )
    return daily.select(group_cols + ["date", "cum_sum", "cum_n"])


def _lookup_before(daily: pl.DataFrame, targets: pl.DataFrame, *, group_cols: list[str]) -> pl.DataFrame:
    """`targets`（`group_cols` + `date` 列を持つ）の各行について、
    その `date` **より前**（`D-054`）の累積 `(cum_sum, cum_n)` を返す。
    """
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


def _shrunk_column(
    df: pl.DataFrame, *, n_col: str, sum_col: str, mu_col: str, out_col: str, k: float,
) -> pl.DataFrame:
    """`n_col>0` なら縮約値、そうでなければ `μ_global` そのもの（`shrink()` の n=0 と同じ極限）。

    `pl.when().then(map_elements(...))` は `then` 側の式を条件に関わらず
    全行で評価してしまう（polars の既知の挙動）ため、`when`/`otherwise` で
    分岐せず、ラムダ自身に `n_col is None/0` の分岐を持たせる。
    """
    def _shrink_row(s: dict) -> float:
        n = s[n_col]
        if n is None or n <= 0:
            return s[mu_col]
        return shrink(n, s[sum_col] / n, k=k, mu_global=s[mu_col])

    return df.with_columns(
        pl.struct([n_col, sum_col, mu_col])
        .map_elements(_shrink_row, return_dtype=pl.Float64)
        .alias(out_col)
    )


def compute_f202(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date, k: float = DEFAULT_K,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        targets = conn.execute(_TARGET_SQL).pl().rename({"target_date": "date"})
        population = conn.execute(_POPULATION_SQL).pl()
    finally:
        conn.unregister("base")

    bucket_cols = ["surface", "distance_band"]

    # μ_global,c: 条件バケツだけで見た母集団平均
    mu_daily = _cumulative_before(population, group_cols=bucket_cols)
    mu_looked_up = _lookup_before(mu_daily, targets, group_cols=bucket_cols)
    mu_looked_up = mu_looked_up.with_columns(
        pl.when(pl.col("cum_n").is_not_null() & (pl.col("cum_n") > 0))
        .then(pl.col("cum_sum") / pl.col("cum_n"))
        .otherwise(FALLBACK_MU_GLOBAL)
        .alias("mu_global_c")
    ).select(_KEY_COLS + ["mu_global_c"])

    out = targets.join(mu_looked_up, on=_KEY_COLS, how="left")

    for id_col, prefix in [("sire_id", "f202_sire"), ("damsire_id", "f202_damsire")]:
        group_cols = [id_col] + bucket_cols
        daily = _cumulative_before(population, group_cols=group_cols)
        sub_targets = targets.select(_KEY_COLS + [id_col] + bucket_cols + ["date"])
        looked_up = _lookup_before(daily, sub_targets, group_cols=group_cols)
        out = out.join(
            looked_up.rename({"cum_sum": f"{prefix}_sum", "cum_n": f"{prefix}_n"}),
            on=_KEY_COLS, how="left",
        )

    out = _shrunk_column(
        out, n_col="f202_sire_n", sum_col="f202_sire_sum", mu_col="mu_global_c",
        out_col="f202_sire", k=k,
    )
    out = _shrunk_column(
        out, n_col="f202_damsire_n", sum_col="f202_damsire_sum", mu_col="mu_global_c",
        out_col="f202_damsire", k=k,
    )

    # sire_id / damsire_id が NULL の馬では計算できない（D-058 / 003 F-202）
    out = out.with_columns(
        pl.when(pl.col("sire_id").is_null()).then(None).otherwise(pl.col("f202_sire")).alias("f202_sire"),
        pl.when(pl.col("sire_id").is_null()).then(1).otherwise(0).alias("f202_sire_unavailable"),
        pl.when(pl.col("damsire_id").is_null()).then(None).otherwise(pl.col("f202_damsire")).alias("f202_damsire"),
        pl.when(pl.col("damsire_id").is_null()).then(1).otherwise(0).alias("f202_damsire_unavailable"),
    )

    return out.select([
        "race_id", "horse_id",
        "f202_sire", "f202_sire_unavailable",
        "f202_damsire", "f202_damsire_unavailable",
    ])
