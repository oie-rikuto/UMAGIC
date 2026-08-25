"""`F-103` ペース適性（`docs/spec/003-features.md` / `D-061` / `D-065`）。

対象馬の過去走ごとに、その過去走のペース指数 `p_i`（自馬を除く出走馬の
`F-101` 分布から算出、`D-061` / `D-065`）と着差 `d_i`（`D-064`）の組を集め、
`d_i ~ p_i` の回帰係数 `β` を `F-902`（`D-051`）で縮約する。

`μ_global`（縮約先の全体平均）は対象行ごとに、その行自身のレース日付
より前のデータだけで計算する（`D-054` の追記事項）。全馬・全レースに
ついて生の `β` を計算し、日付順の累積平均として持つことで、対象行ごとの
Python ループを避けている。
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import polars as pl

from umagic.features.f101 import DEFAULT_QUANTILE, compute_f101
from umagic.features.margin import parse_margin
from umagic.features.missing import with_unavailable_indicator
from umagic.features.shrinkage import shrink

DEFAULT_K = 5.0  # 縮約の強さ。既定値は P-3 で決める（D-051 と同じ扱い）
FALLBACK_MU_GLOBAL = 0.0  # 縮約対象となる生の β が1件も無い場合の中立値（ペース適性なし）

# `f101_all`（呼び出し側が事前に登録した `race_id, horse_id, f101` テーブル）を
# 使って、`base` の各行について「対象馬の過去走ごとの (p_i, 着差の生テキスト)」を返す。
_PACE_PAIRS_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id AS target_horse_id,
           r.date AS target_date
    FROM base b
    JOIN races r ON r.race_id = b.race_id
),
past_races AS (
    SELECT t.target_race_id, t.target_horse_id, ru.race_id AS past_race_id,
           ru.margin AS own_margin
    FROM targets t
    JOIN runners ru ON ru.horse_id = t.target_horse_id
    JOIN races hr ON hr.race_id = ru.race_id AND hr.date < t.target_date
),
other_horses AS (
    SELECT pr.target_race_id, pr.target_horse_id, pr.past_race_id, f.f101
    FROM past_races pr
    JOIN runners oru ON oru.race_id = pr.past_race_id AND oru.horse_id != pr.target_horse_id
    JOIN f101_all f ON f.race_id = oru.race_id AND f.horse_id = oru.horse_id
    WHERE f.f101 IS NOT NULL
),
race_pace AS (
    -- D-061/D-065: 自馬を除く他馬の F-101 平均を反転した値
    SELECT target_race_id, target_horse_id, past_race_id, 1 - AVG(f101) AS p_i
    FROM other_horses
    GROUP BY target_race_id, target_horse_id, past_race_id
)
SELECT
    pr.target_race_id AS race_id,
    pr.target_horse_id AS horse_id,
    pr.past_race_id,
    pr.own_margin,
    rp.p_i
FROM past_races pr
LEFT JOIN race_pace rp
  ON rp.target_race_id = pr.target_race_id
 AND rp.target_horse_id = pr.target_horse_id
 AND rp.past_race_id = pr.past_race_id
"""


def _pace_pairs(conn: duckdb.DuckDBPyConnection, base: pl.DataFrame) -> pl.DataFrame:
    """`f101_all` が登録済みであること。列: `race_id, horse_id, past_race_id, own_margin, p_i`。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        return conn.execute(_PACE_PAIRS_SQL).pl()
    finally:
        conn.unregister("base")


MIN_VAR_X = 1e-6  # D-102: この値未満の分散では beta を計算しない（数値不安定性のガード）


def _ols_slope(xs: list[float], ys: list[float]) -> float | None:
    """`ys ~ xs` の単回帰の傾き。標本数2未満、または `xs` の分散が `MIN_VAR_X`
    未満なら `None`（`D-102`）。

    `beta = cov_xy / var_x` は `var_x` で割るため、分散がゼロに近いほど
    誤差が増幅される（悪条件の回帰）。`p_i` は DuckDB の並列集計
    （`AVG()`）を経由しており、実行のたびに最終桁レベルの浮動小数点誤差
    （`~1e-14` 程度、通常は無害）が乗る。`var_x` が極小だとこの誤差が
    `beta` に大きく増幅され、`beta` は全馬共通の `mu_global`（累積平均）
    に混ざるため、少数の不安定な `beta` の誤差が多数の行の `f103` に
    伝播する（`R-021` 違反の実例、`Q-038` で発見）。**`var_x == 0` だけ
    弾ぐ既存のガードでは、ゼロに近いが非ゼロの分散を防げなかった。**
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x < MIN_VAR_X:
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov_xy / var_x


_BETA_SCHEMA = {
    "race_id": pl.Int64, "horse_id": pl.Int64, "beta": pl.Float64,
    "n_past_races": pl.Int64, "n_valid_pairs": pl.Int64,
}


def _raw_beta(pairs: pl.DataFrame) -> pl.DataFrame:
    """`(race_id, horse_id)` ごとに縮約前の生の `β` を計算する。

    `p_i` と `d_i`（着差の数値化, `D-064`）がともに揃う過去走が2走未満、
    または `p_i` の分散が `MIN_VAR_X` 未満の場合は `beta=None`
    （`F-902` の縮約対象外、`D-102`）。
    過去走が1件も無い `(race_id, horse_id)` は `pairs` に現れないため、
    呼び出し側で対象母集団に left join して `n_past_races=0` を復元すること。

    `_ols_slope()` と同じ二パスの式を Polars でベクトル化している
    （`D-117`）。逐次和ではなくなるが、実データで結果が一致することと
    `R-021`（run-to-run の再現性）を保つことを実測で確認している。
    """
    if pairs.is_empty():
        return pl.DataFrame(schema=_BETA_SCHEMA)

    keys = ["race_id", "horse_id"]
    df = pairs.with_columns(
        pl.col("own_margin").map_elements(parse_margin, return_dtype=pl.Float64).alias("d_i")
    )
    # 出力の行順を Python ループ版（グループの初出順）に合わせる
    n_past = df.group_by(keys, maintain_order=True).agg(
        pl.len().cast(pl.Int64).alias("n_past_races")
    )

    # 二パス（平均を引いてから二乗和）にして `_ols_slope()` の式に揃える。
    # `sum(x²)-n·x̄²` の一パス版は打ち切り誤差が大きく、`D-102` が
    # 問題にしている var_x 極小の領域で挙動が変わる
    valid = df.filter(pl.col("p_i").is_not_null() & pl.col("d_i").is_not_null())
    valid = valid.with_columns(
        (pl.col("p_i") - pl.col("p_i").mean().over(keys)).alias("_dx"),
        (pl.col("d_i") - pl.col("d_i").mean().over(keys)).alias("_dy"),
    )
    agg = valid.group_by(keys, maintain_order=True).agg(
        pl.len().cast(pl.Int64).alias("n_valid_pairs"),
        (pl.col("_dx") ** 2).sum().alias("_var_x"),
        (pl.col("_dx") * pl.col("_dy")).sum().alias("_cov_xy"),
    )
    agg = agg.with_columns(
        pl.when((pl.col("n_valid_pairs") >= 2) & (pl.col("_var_x") >= MIN_VAR_X))
        .then(pl.col("_cov_xy") / pl.col("_var_x"))
        .otherwise(None)
        .alias("beta")
    )

    out = n_past.join(agg, on=keys, how="left").with_columns(
        pl.col("n_valid_pairs").fill_null(0)
    )
    return out.select(list(_BETA_SCHEMA.keys()))


def _mu_global_daily(beta_dated: pl.DataFrame) -> pl.DataFrame:
    """日付ごとの `β` を累積し、その日付**まで**の合計・件数を返す（列: `date, cum_sum, cum_n`）。"""
    valid = beta_dated.filter(pl.col("beta").is_not_null()).select(["date", "beta"])
    if valid.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "cum_sum": pl.Float64, "cum_n": pl.Int64})
    daily = (
        valid.group_by("date")
        .agg(pl.col("beta").sum().alias("sum_beta"), pl.len().alias("n_beta"))
        .sort("date")
        .with_columns(
            pl.col("sum_beta").cum_sum().alias("cum_sum"),
            pl.col("n_beta").cum_sum().alias("cum_n"),
        )
        .select(["date", "cum_sum", "cum_n"])
    )
    return daily


def _mu_global_for_dates(daily: pl.DataFrame, target_dates: pl.DataFrame) -> pl.DataFrame:
    """`target_dates`（列 `race_id, horse_id, date`）の各行について、
    その `date` **より前**（`D-054`）の累積平均 `mu_global` を返す。
    """
    if daily.is_empty() or target_dates.is_empty():
        return target_dates.select(["race_id", "horse_id"]).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("mu_global")
        )

    # 「date より前」を asof(<=) で表すため、探索キーを1日前にずらす
    probe = (
        target_dates.with_columns((pl.col("date") - timedelta(days=1)).alias("_asof_date"))
        .sort("_asof_date")
    )
    joined = probe.join_asof(
        daily.sort("date"), left_on="_asof_date", right_on="date", strategy="backward",
    )
    joined = joined.with_columns(
        pl.when(pl.col("cum_n").is_not_null() & (pl.col("cum_n") > 0))
        .then(pl.col("cum_sum") / pl.col("cum_n"))
        .otherwise(None)
        .alias("mu_global")
    )
    return joined.select(["race_id", "horse_id", "mu_global"])


def compute_f103(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
    quantile: float = DEFAULT_QUANTILE, k: float = DEFAULT_K,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])

    all_pairs_base = conn.execute(
        "SELECT DISTINCT race_id, horse_id FROM runners ORDER BY race_id, horse_id"
    ).pl()
    # F-101 は全馬について1回だけ計算し、他馬のペース指数の参照元として使い回す
    f101_all = compute_f101(conn, all_pairs_base, as_of=as_of, quantile=quantile)
    conn.register("f101_all", f101_all.select(["race_id", "horse_id", "f101"]))
    try:
        target_pairs = _pace_pairs(conn, base)
        pop_pairs = _pace_pairs(conn, all_pairs_base)
    finally:
        conn.unregister("f101_all")

    target_beta = _raw_beta(target_pairs)
    pop_beta = _raw_beta(pop_pairs)

    dates = conn.execute("SELECT race_id, date FROM races").pl()

    # 過去走が1件も無い対象行は target_pairs に現れないため、base に left join して復元する
    target_beta_full = base.join(target_beta, on=["race_id", "horse_id"], how="left").with_columns(
        pl.col("n_past_races").fill_null(0),
        pl.col("n_valid_pairs").fill_null(0),
    )
    target_dated = target_beta_full.join(dates, on="race_id", how="left")

    pop_beta_dated = pop_beta.join(dates, on="race_id", how="left")
    daily = _mu_global_daily(pop_beta_dated)
    mu = _mu_global_for_dates(daily, target_dated.select(["race_id", "horse_id", "date"]))

    joined = target_dated.join(mu, on=["race_id", "horse_id"], how="left").with_columns(
        pl.col("mu_global").fill_null(FALLBACK_MU_GLOBAL)
    )

    # beta が無ければ縮約せず NaN のまま（shrink() の n<=0 フォールバックに頼らない）
    joined = joined.with_columns(
        pl.when(pl.col("beta").is_not_null())
        .then(
            pl.struct(["n_valid_pairs", "beta", "mu_global"]).map_elements(
                lambda s: shrink(s["n_valid_pairs"], s["beta"], k=k, mu_global=s["mu_global"]),
                return_dtype=pl.Float64,
            )
        )
        .otherwise(None)
        .alias("f103")
    )

    is_structural = (pl.col("n_past_races") > 0) & pl.col("beta").is_null()
    out = with_unavailable_indicator(joined, "f103", is_structural=is_structural)
    return out.select(["race_id", "horse_id", "f103", "f103_unavailable"])
