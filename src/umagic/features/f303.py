"""`F-303` 上がり3F関連（`docs/spec/003-features.md`）。

`F-301`（馬場差推定、`013-track-variant.md`）に依存しないため、本仕様で
実装する。過去走の上がり3F（`runners.last_3f`）について、全履歴・直近N走
の縮約平均（`D-051` / `D-059`）と、過去走それぞれでのレース内上がり順位
（`last_3f` が小さいほど上位）の平均を出す。

`last3f_recent`（直近N走）の「N走」は、**`last_3f` が記録されている過去走
に限った直近N走**とする（`F-101` の「1角を含む過去走のみ」と同じ、対象
条件を満たす履歴だけを母集団にする既存の扱いに合わせた）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from umagic.features.missing import with_unavailable_indicator
from umagic.features.relative import relativize
from umagic.features.shrinkage import per_row_stat_before, shrink
from umagic.features.window import full_and_recent

DEFAULT_N_RECENT = 5  # 既定値は P-3 で決める（D-059）
DEFAULT_K = 5.0  # 既定値は P-3 で決める（D-051）

_MU_GLOBAL_SQL = (
    "SELECT AVG(ru.last_3f) FROM runners ru JOIN races r USING (race_id) "
    "WHERE r.date < ? AND ru.last_3f IS NOT NULL"
)

_PAST_STARTS_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id AS target_horse_id,
           r.date AS target_date
    FROM base b JOIN races r ON r.race_id = b.race_id
)
SELECT t.target_race_id AS race_id, t.target_horse_id AS horse_id, COUNT(*) AS n_past_starts
FROM targets t
JOIN runners ru ON ru.horse_id = t.target_horse_id
JOIN races hr ON hr.race_id = ru.race_id AND hr.date < t.target_date
GROUP BY t.target_race_id, t.target_horse_id
"""

_HISTORY_SQL = """
WITH targets AS (
    SELECT DISTINCT b.race_id AS target_race_id, b.horse_id AS target_horse_id,
           r.date AS target_date
    FROM base b JOIN races r ON r.race_id = b.race_id
)
SELECT t.target_race_id AS race_id, t.target_horse_id AS horse_id,
       ru.race_id AS past_race_id, hr.date AS past_date, ru.last_3f
FROM targets t
JOIN runners ru ON ru.horse_id = t.target_horse_id
JOIN races hr ON hr.race_id = ru.race_id AND hr.date < t.target_date
WHERE ru.last_3f IS NOT NULL
"""

_ALL_RUNNERS_LAST3F_SQL = """
SELECT ru.race_id, ru.horse_id, ru.last_3f, r.n_starters
FROM runners ru JOIN races r USING (race_id)
WHERE ru.last_3f IS NOT NULL
"""


def _last3f_rank_all(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """DB内の全 `(race_id, horse_id)` について、そのレース内での上がり3F順位/頭数を返す。"""
    df = conn.execute(_ALL_RUNNERS_LAST3F_SQL).pl()
    if df.is_empty():
        return pl.DataFrame(schema={"race_id": pl.Int64, "horse_id": pl.Int64, "last_3f_rank": pl.Float64})
    out = relativize(df, "last_3f", race_id_col="race_id", n_starters_col="n_starters")
    return out.select(["race_id", "horse_id", "last_3f_rank"])


def compute_f303(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
    n_recent: int = DEFAULT_N_RECENT, k: float = DEFAULT_K,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    conn.register("base", base)
    try:
        past_starts = conn.execute(_PAST_STARTS_SQL).pl()
        history = conn.execute(_HISTORY_SQL).pl()
    finally:
        conn.unregister("base")

    rank_all = _last3f_rank_all(conn)
    history = history.join(
        rank_all, left_on=["past_race_id", "horse_id"], right_on=["race_id", "horse_id"],
        how="left",
    )

    out = base.join(past_starts, on=["race_id", "horse_id"], how="left").with_columns(
        pl.col("n_past_starts").fill_null(0)
    )

    if history.is_empty():
        window = pl.DataFrame(schema={
            "race_id": pl.Int64, "horse_id": pl.Int64, "n_all": pl.Int64,
            "last_3f_all": pl.Float64, "n_recent": pl.Int64, "last_3f_recent": pl.Float64,
        })
        rank_avg = pl.DataFrame(schema={
            "race_id": pl.Int64, "horse_id": pl.Int64, "last3f_rank_in_race": pl.Float64,
        })
    else:
        window = full_and_recent(
            history, group_col=["race_id", "horse_id"], date_col="past_date",
            value_col="last_3f", n=n_recent,
        )
        rank_avg = (
            history.group_by(["race_id", "horse_id"])
            .agg(pl.col("last_3f_rank").mean().alias("last3f_rank_in_race"))
        )

    out = out.join(window, on=["race_id", "horse_id"], how="left").with_columns(
        pl.col("n_all").fill_null(0), pl.col("n_recent").fill_null(0),
    )
    out = out.join(rank_avg, on=["race_id", "horse_id"], how="left")

    dates = conn.execute("SELECT race_id, date FROM races").pl()
    out = out.join(dates, on="race_id", how="left")
    mu = per_row_stat_before(
        conn, out.select(["race_id", "horse_id"]),
        race_dates=dict(zip(out["race_id"].to_list(), out["date"].to_list())),
        stat_sql=_MU_GLOBAL_SQL,
    ).rename({"stat": "mu_global"})
    out = out.join(mu, on=["race_id", "horse_id"], how="left")

    # x̄（last_3f_all / last_3f_recent）が無ければ縮約せず NaN のまま
    # （shrink() の n<=0 フォールバックに頼らない。F-103 と同じ理由）
    out = out.with_columns(
        pl.when(pl.col("last_3f_all").is_not_null())
        .then(
            pl.struct(["n_all", "last_3f_all", "mu_global"]).map_elements(
                lambda s: shrink(s["n_all"], s["last_3f_all"], k=k, mu_global=s["mu_global"]),
                return_dtype=pl.Float64,
            )
        )
        .otherwise(None)
        .alias("last3f_all"),
        pl.when(pl.col("last_3f_recent").is_not_null())
        .then(
            pl.struct(["n_recent", "last_3f_recent", "mu_global"]).map_elements(
                lambda s: shrink(s["n_recent"], s["last_3f_recent"], k=k, mu_global=s["mu_global"]),
                return_dtype=pl.Float64,
            )
        )
        .otherwise(None)
        .alias("last3f_recent"),
    )

    is_structural = (pl.col("n_past_starts") > 0) & (pl.col("n_all") == 0)
    out = with_unavailable_indicator(out, "last3f_all", is_structural=is_structural)
    out = with_unavailable_indicator(out, "last3f_recent", is_structural=is_structural)
    out = with_unavailable_indicator(out, "last3f_rank_in_race", is_structural=is_structural)

    return out.select([
        "race_id", "horse_id",
        "last3f_all", "last3f_all_unavailable",
        "last3f_recent", "last3f_recent_unavailable",
        "last3f_rank_in_race", "last3f_rank_in_race_unavailable",
    ])
