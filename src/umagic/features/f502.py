"""`F-502` 当日時計傾向（`docs/domain-knowledge.md` `F-5xx` / `D-010` / `D-148`）。

対象レースより**前に発走した同日・同競馬場・同馬場（芝/ダート）**のレース
（`race_number <` の厳密不等号、`D-010`）の全出走馬について、`F-304`
（`D-123`）と同じ基準タイム `base(surface, distance, class_key, month)`
からの残差 `e = time_sec - base_t` を集め、その**中央値**を当日の時計傾向
とする。正なら基準より遅い時計（時計の掛かる馬場）、負なら速い馬場。

`F-304` の `tv`（`date, course, surface` 単位の中央値）と同じ量だが、
`tv` は対象レースの前後を区別せずその日**全体**を集計するため、対象レース
自身に使うと `D-010` に違反する（後続レースの結果が混入する）。この
モジュールは EARLIER 側だけを集計する独立の計算として持つ（`D-148` の
実装方針）。コードは `F-304` と共有しない——`F-301`/`F-304` が互いに
独立した「基準タイム」の実装を持つのと同じ形（`D-148`）。

**リーク防止（`D-054` 原則7 / `D-010`）**: `base` は F-304 と同じく
**厳密にその月より前**の集計だけを使う。集約対象は
`(date, course, surface)` が対象レースと一致し `race_number` が厳密に
小さいレースの出走行のみで、対象レース自身・後続レース・別競馬場・
別馬場は一切含まない。

`F-501`/`F-502` のみが同日レースを参照できる例外規定
（`domain-knowledge.md` 5節 原則3）に基づく。`race_level`: `True`
（レース内で全馬共通のため `F-901` の相対化は適用しない、`D-021`）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

MIN_BASE_N = 20  # 基準タイムの推定に要する最小標本数（F-304 と同値）

# 全出走行の残差 e = time_sec - base_t。`as_of` に依存しない
# （行ごとに自身の月・当日で切るため、F-304 の `_SPD_SQL` と同じ構造）
_RESID_SQL = f"""
WITH r AS (
    SELECT ru.race_id, ru.horse_id, CAST(ru.time_sec AS DOUBLE) AS t,
           ra.date, ra.course, ra.surface, ra.distance, ra.race_number,
           DATE_TRUNC('month', ra.date) AS mo,
           COALESCE(ra.grade, ra.race_class, 'その他') AS class_key
    FROM runners ru JOIN races ra USING (race_id)
    WHERE ru.status IN ('出走', '降着') AND ru.time_sec IS NOT NULL
),
monthly AS (
    SELECT surface, distance, class_key, mo, SUM(t) AS s, COUNT(*) AS n
    FROM r GROUP BY 1,2,3,4
),
cum AS (
    SELECT surface, distance, class_key, mo,
           SUM(s) OVER w AS cs, SUM(n) OVER w AS cn
    FROM monthly
    WINDOW w AS (PARTITION BY surface, distance, class_key ORDER BY mo
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
base AS (
    SELECT surface, distance, class_key, mo, cs / cn AS base_t
    FROM cum WHERE cn IS NOT NULL AND cn >= {MIN_BASE_N}
),
-- class_key に履歴が無い場合のフォールバック（F-304 と同じ、新設クラスの初月など）
monthly_fb AS (
    SELECT surface, distance, mo, SUM(t) AS s, COUNT(*) AS n
    FROM r GROUP BY 1,2,3
),
cum_fb AS (
    SELECT surface, distance, mo, SUM(s) OVER w AS cs, SUM(n) OVER w AS cn
    FROM monthly_fb
    WINDOW w AS (PARTITION BY surface, distance ORDER BY mo
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
base_fallback AS (
    SELECT surface, distance, mo, cs / cn AS base_t
    FROM cum_fb WHERE cn IS NOT NULL AND cn >= {MIN_BASE_N}
)
SELECT r.race_id, r.horse_id, r.date, r.course, r.surface, r.race_number,
       r.t - COALESCE(b.base_t, f.base_t) AS e
FROM r
LEFT JOIN base b
  ON b.surface = r.surface AND b.distance = r.distance
 AND b.class_key = r.class_key AND b.mo = r.mo
LEFT JOIN base_fallback f
  ON f.surface = r.surface AND f.distance = r.distance AND f.mo = r.mo
WHERE COALESCE(b.base_t, f.base_t) IS NOT NULL
"""

# レース単位で1回だけ集約する（F-501 と違い、対象を (race_id, horse_id) では
# なく race_id で引く。同一レース内の全馬で値が同じため、頭数分の重複join を
# 避ける）
_AGG_SQL = """
WITH targets AS (
    SELECT DISTINCT r.race_id, r.date, r.course, r.surface, r.race_number
    FROM base b JOIN races r ON r.race_id = b.race_id
),
earlier AS (
    SELECT t.race_id, resid.e
    FROM targets t
    JOIN resid
      ON resid.date = t.date AND resid.course = t.course
     AND resid.surface = t.surface AND resid.race_number < t.race_number
)
SELECT t.race_id, MEDIAN(e2.e) AS f502, COUNT(e2.e) AS n_qualifying
FROM targets t
LEFT JOIN earlier e2 ON e2.race_id = t.race_id
GROUP BY t.race_id
"""

_SCHEMA = {
    "race_id": pl.Int64, "horse_id": pl.Int64,
    "f502": pl.Float64, "f502_unavailable": pl.Int32,
}


def compute_f502(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    if base.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    conn.register("base", base)
    try:
        resid = conn.execute(_RESID_SQL).pl()
        conn.register("resid", resid)
        try:
            agg = conn.execute(_AGG_SQL).pl()
        finally:
            conn.unregister("resid")
    finally:
        conn.unregister("base")

    out = base.join(agg, on="race_id", how="left").with_columns(
        pl.col("n_qualifying").fill_null(0)
    )
    # 先行レースが0本（1Rなど）、または先行レースの出走馬に基準タイムが
    # 1件も無い → 構造的欠損（F-501 と同じ扱い）
    out = out.with_columns(
        pl.when(pl.col("n_qualifying") == 0).then(None).otherwise(pl.col("f502")).alias("f502"),
        pl.when(pl.col("n_qualifying") == 0).then(1).otherwise(0).alias("f502_unavailable"),
    )
    return out.select(list(_SCHEMA.keys()))
