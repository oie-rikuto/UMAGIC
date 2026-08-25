"""`F-304` 中央値ベースの速度指数（`docs/spec/003-features.md` / `D-123`）。

`F-301`/`F-302` と同じ「馬場差で補正した走破時計」を、混合効果モデルでは
なく**中央値**で推定する別ルート。`F-301` は推定が破綻して既定無効
（`D-110`）であり、こちらは標本の少ない馬で効果量が発散する問題
（`D-108`）が構造的に起きない。

    base(surface, distance, class_key, month)  その月より前の time_sec の平均・標準偏差
    resid = time_sec − base
    tv(date, course, surface)                  その日その場その馬場の resid の中央値
    spd   = −(resid − tv) / base_sd            正が速い

`class_key` は `COALESCE(grade, race_class, 'その他')`。`race_class` は
`G1`〜`G3`・`L` を `'オープン'` に丸めるため（`D-111`）`grade` を優先する。

**リーク防止（`D-054` 原則7）**: `base` は**厳密にその月より前**の集計だけを
使う。`tv` はその過去レース当日のデータのみを使い、対象レースより前の
日付なので対象行から見れば全て過去である。馬ごとの集約は
`race_date < target_race_date` で厳密に切る。

`race_level`: `False`

**指示子は `fspd_unavailable` の1列だけ置く。** 値列4本は「過去走に速度指数が
1件でもあるか」で同時に決まり、列ごとに指示子を作ると同一内容の列が4本
並ぶ（`D-119` が示したとおり、寄与しない列を増やすのは害になる）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from umagic.features.missing import with_unavailable_indicator

CLIP = 5.0  # 大差負けが平均を壊すため、集約前にこの絶対値でクリップする
MIN_BASE_N = 20  # 基準タイムの推定に要する最小標本数
MIN_TV_N = 20  # 馬場差の推定に要する最小標本数

_VALUE_COLS = ("fspd_best", "fspd_mean", "fspd_last", "fspd_recent3", "fspd_n")

# 全出走行の速度指数。`as_of` に依存しない（行ごとに自身の月・当日で切るため）
_SPD_SQL = f"""
WITH r AS (
    SELECT ru.race_id, ru.horse_id, CAST(ru.time_sec AS DOUBLE) AS t,
           ra.date, ra.course, ra.surface, ra.distance,
           DATE_TRUNC('month', ra.date) AS mo,
           COALESCE(ra.grade, ra.race_class, 'その他') AS class_key
    FROM runners ru JOIN races ra USING (race_id)
    WHERE ru.status IN ('出走', '降着') AND ru.time_sec IS NOT NULL
),
monthly AS (
    SELECT surface, distance, class_key, mo,
           SUM(t) AS s, SUM(t*t) AS ss, COUNT(*) AS n
    FROM r GROUP BY 1,2,3,4
),
cum AS (
    SELECT surface, distance, class_key, mo,
           SUM(s) OVER w AS cs, SUM(ss) OVER w AS css, SUM(n) OVER w AS cn
    FROM monthly
    WINDOW w AS (PARTITION BY surface, distance, class_key ORDER BY mo
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
base AS (
    SELECT surface, distance, class_key, mo,
           cs / cn AS base_t,
           CASE WHEN cn > 1 THEN sqrt(GREATEST(css/cn - (cs/cn)*(cs/cn), 0.0)) END AS base_sd
    FROM cum WHERE cn IS NOT NULL AND cn >= {MIN_BASE_N}
),
-- class_key に履歴が無い場合のフォールバック（新設クラスの初月など）
monthly_sd AS (
    SELECT surface, distance, mo, SUM(t) AS s, SUM(t*t) AS ss, COUNT(*) AS n
    FROM r GROUP BY 1,2,3
),
cum_sd AS (
    SELECT surface, distance, mo,
           SUM(s) OVER w AS cs, SUM(ss) OVER w AS css, SUM(n) OVER w AS cn
    FROM monthly_sd
    WINDOW w AS (PARTITION BY surface, distance ORDER BY mo
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
base_fallback AS (
    SELECT surface, distance, mo,
           cs / cn AS base_t, sqrt(GREATEST(css/cn - (cs/cn)*(cs/cn), 0.0)) AS base_sd
    FROM cum_sd WHERE cn IS NOT NULL AND cn >= {MIN_BASE_N}
),
resid AS (
    SELECT r.race_id, r.horse_id, r.date, r.course, r.surface,
           r.t - COALESCE(b.base_t, f.base_t) AS e,
           COALESCE(b.base_sd, f.base_sd) AS base_sd
    FROM r
    LEFT JOIN base b
      ON b.surface = r.surface AND b.distance = r.distance
     AND b.class_key = r.class_key AND b.mo = r.mo
    LEFT JOIN base_fallback f
      ON f.surface = r.surface AND f.distance = r.distance AND f.mo = r.mo
    WHERE COALESCE(b.base_t, f.base_t) IS NOT NULL
),
tv AS (
    SELECT date, course, surface, MEDIAN(e) AS tv, COUNT(*) AS tv_n
    FROM resid GROUP BY 1,2,3
)
SELECT resid.race_id, resid.horse_id,
       -(resid.e - tv.tv) / resid.base_sd AS spd
FROM resid JOIN tv USING (date, course, surface)
WHERE resid.base_sd IS NOT NULL AND resid.base_sd > 0 AND tv.tv_n >= {MIN_TV_N}
"""

# 対象行ごとに `race_date < target_race_date` で厳密に切る（`D-054` 原則7）
_AGG_SQL = f"""
WITH t AS (
    SELECT b.race_id, b.horse_id, r.date
    FROM base b JOIN races r ON r.race_id = b.race_id
),
past AS (
    SELECT t.race_id, t.horse_id,
           GREATEST(LEAST(s.spd, {CLIP}), -{CLIP}) AS spd_c,
           ROW_NUMBER() OVER (
               PARTITION BY t.race_id, t.horse_id ORDER BY pr.date DESC, pr.race_id DESC
           ) AS rn
    FROM t
    JOIN spd s ON s.horse_id = t.horse_id
    JOIN races pr ON pr.race_id = s.race_id AND pr.date < t.date
)
SELECT race_id, horse_id,
       MAX(spd_c)                            AS fspd_best,
       AVG(spd_c)                            AS fspd_mean,
       MAX(CASE WHEN rn = 1 THEN spd_c END)  AS fspd_last,
       AVG(CASE WHEN rn <= 3 THEN spd_c END) AS fspd_recent3,
       CAST(COUNT(*) AS DOUBLE)              AS fspd_n
FROM past GROUP BY 1, 2
"""

# 「過去走はあるが速度指数が1件も無い」を構造的欠損と区別するための母数
_PAST_STARTS_SQL = """
WITH t AS (
    SELECT b.race_id, b.horse_id, r.date
    FROM base b JOIN races r ON r.race_id = b.race_id
)
SELECT t.race_id, t.horse_id, COUNT(*) AS n_past_starts
FROM t
JOIN runners ru ON ru.horse_id = t.horse_id
JOIN races pr ON pr.race_id = ru.race_id AND pr.date < t.date
GROUP BY 1, 2
"""

_SCHEMA = {
    "race_id": pl.Int64, "horse_id": pl.Int64,
    "fspd_best": pl.Float64, "fspd_mean": pl.Float64, "fspd_last": pl.Float64,
    "fspd_recent3": pl.Float64, "fspd_n": pl.Float64, "fspd_unavailable": pl.Int32,
}


def compute_spd(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """全出走行の速度指数 `(race_id, horse_id, spd)` を返す。`as_of` に依存しない。"""
    return conn.execute(_SPD_SQL).pl()


def compute_f304(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    if base.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    spd = compute_spd(conn)
    conn.register("spd", spd)
    conn.register("base", base)
    try:
        agg = conn.execute(_AGG_SQL).pl()
        past = conn.execute(_PAST_STARTS_SQL).pl()
    finally:
        conn.unregister("base")
        conn.unregister("spd")

    out = base.join(agg, on=["race_id", "horse_id"], how="left")
    out = out.join(past, on=["race_id", "horse_id"], how="left").with_columns(
        pl.col("n_past_starts").fill_null(0),
        pl.col("fspd_n").fill_null(0.0),
    )
    is_structural = (pl.col("n_past_starts") > 0) & pl.col("fspd_best").is_null()
    out = with_unavailable_indicator(out, "fspd_best", is_structural=is_structural)
    out = out.rename({"fspd_best_unavailable": "fspd_unavailable"})
    return out.select(list(_SCHEMA.keys()))
