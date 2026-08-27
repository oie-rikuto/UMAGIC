"""`F-807` 前走からの条件替わり（`docs/spec/003-features.md` / `D-137`）。

前走の条件（距離・馬場・コース・頭数）と、今回との変化。

**`F-601`/`F-602` は前走の着順・着差・上がり順位・週数・グレードを持つが、
どんな条件のレースだったかを持たない（`D-137`）。** そのため距離延長/短縮、
芝→ダート替わりといった競馬予想の基本要素が表現できていなかった。
`F-704`（騎手 × コース・距離帯）と `F-802`（馬自身の条件別成績）は条件
**適性**を持つが、**前走からの変化**は別物である。

**生の値と差分の両方を持つ（`D-137`）。** 実測では生だけでも木がかなり
処理できたが（`−0.0053`）、明示的な差分にも上乗せの価値があった
（両方で `−0.0064`）。

`f807_prev_surface` / `f807_prev_course` は文字列のカテゴリ列として出し、
`D-092` の丸め（`stage2._CATEGORY_COLS` / `orchestration.CATEGORY_COLUMNS`）
に載せる。これにより `F-901`（レース内相対化）の対象からも自動的に外れる。

`race_level`: `False`
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

# 直近の過去走（`pr.date < t.date`、`D-054` 原則7）を1本引く。
# 同日複数走は無いが、決定的な順序のため `race_id` を第2キーにする
_SQL = """
WITH t AS (
    SELECT b.race_id, b.horse_id, r.date,
           r.distance AS cur_d, r.surface AS cur_s, r.direction AS cur_dir,
           r.course AS cur_c, r.n_starters AS cur_n
    FROM base b JOIN races r ON r.race_id = b.race_id
),
ranked AS (
    SELECT t.race_id, t.horse_id, t.cur_d, t.cur_s, t.cur_dir, t.cur_c, t.cur_n,
           pr.distance AS prev_d, pr.surface AS prev_s, pr.direction AS prev_dir,
           pr.course AS prev_c, pr.n_starters AS prev_n,
           ROW_NUMBER() OVER (
               PARTITION BY t.race_id, t.horse_id ORDER BY pr.date DESC, pr.race_id DESC
           ) AS rn
    FROM t
    JOIN runners ru ON ru.horse_id = t.horse_id
    JOIN races pr ON pr.race_id = ru.race_id AND pr.date < t.date
    WHERE ru.status IN ('出走', '降着', '競走中止', '失格')
)
SELECT race_id, horse_id,
       CAST(prev_d AS DOUBLE)               AS f807_prev_distance,
       prev_s                               AS f807_prev_surface,
       prev_c                               AS f807_prev_course,
       CAST(prev_n AS DOUBLE)               AS f807_prev_n_starters,
       CAST(cur_d - prev_d AS DOUBLE)       AS f807_distance_diff,
       CASE WHEN cur_s <> prev_s THEN 1.0 ELSE 0.0 END              AS f807_surface_changed,
       CASE WHEN cur_dir IS DISTINCT FROM prev_dir THEN 1.0 ELSE 0.0 END AS f807_direction_changed,
       CASE WHEN cur_c <> prev_c THEN 1.0 ELSE 0.0 END              AS f807_course_changed,
       CAST(cur_n - prev_n AS DOUBLE)       AS f807_n_starters_diff
FROM ranked WHERE rn = 1
"""

_SCHEMA = {
    "race_id": pl.Int64, "horse_id": pl.Int64,
    "f807_prev_distance": pl.Float64, "f807_prev_surface": pl.Utf8,
    "f807_prev_course": pl.Utf8, "f807_prev_n_starters": pl.Float64,
    "f807_distance_diff": pl.Float64, "f807_surface_changed": pl.Float64,
    "f807_direction_changed": pl.Float64, "f807_course_changed": pl.Float64,
    "f807_n_starters_diff": pl.Float64,
}


def compute_f807(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。

    **過去走が無い行は全列が欠損になる。** 指示子は置かない——「前走が無い」
    は `F-601`/`F-602` の `_unavailable` が既に表しており、同じ情報を重ねても
    寄与しない列が増えるだけである（`D-119` / `D-135`）。
    """
    base = base.select(["race_id", "horse_id"])
    if base.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    conn.register("base", base)
    try:
        df = conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")

    out = base.join(df, on=["race_id", "horse_id"], how="left")
    return out.select(list(_SCHEMA.keys()))
