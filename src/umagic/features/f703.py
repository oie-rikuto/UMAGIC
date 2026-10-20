"""`F-703` 厩舎の勝負度（`docs/spec/003-features.md` / `D-049` / `D-069`）。

遠征フラグは `affiliation`（美浦=`東`/栗東=`西`）と `races.course` の
対応表（`D-069`）で判定する。`地`/`外`/`NULL` は判定できず `NaN`・指示子`1`。

主戦騎手フラグは、対象馬の所属厩舎（`trainer_id`）で最も騎乗数の多い
騎手が、対象レースの騎手と一致するかを見る（`as_of` 未満のデータで判定）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

# D-069: 美浦(東)/栗東(西) と開催競馬場の対応
_HOME_COURSES = {
    "東": {"東京", "中山", "新潟", "福島", "札幌", "函館"},
    "西": {"阪神", "京都", "中京", "小倉"},
}

_TARGET_SQL = """
SELECT b.race_id, b.horse_id, r.course, ru.affiliation, ru.trainer_id, ru.jockey_id,
       r.date AS target_date
FROM base b
JOIN races r ON r.race_id = b.race_id
JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
"""

# D-209: 主戦騎手は「(調教師, 対象日)」だけで決まる。出走馬ごとに過去の全騎乗と結合すると
# 全履歴で約7億組になり一時領域があふれるため、(調教師, 騎手, 日付)の累積騎乗数を作って
# 対象日より前の最新値を引く。1文にまとめるとDuckDBの実行計画が悪化するので段階実行する。
_MAIN_JOCKEY_STAGES = [
    ("_f703_tg", """
        SELECT DISTINCT ru.trainer_id, r.date AS target_date
        FROM base b
        JOIN races r ON r.race_id = b.race_id
        JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
        WHERE ru.trainer_id IS NOT NULL
    """),
    ("_f703_cum", """
        SELECT trainer_id, jockey_id, date,
               SUM(c) OVER (PARTITION BY trainer_id, jockey_id ORDER BY date) AS n_mounts
        FROM (
            SELECT hru.trainer_id, hru.jockey_id, hr.date, COUNT(*) AS c
            FROM runners hru
            JOIN races hr ON hr.race_id = hru.race_id
            WHERE hru.jockey_id IS NOT NULL AND hru.trainer_id IS NOT NULL
            GROUP BY hru.trainer_id, hru.jockey_id, hr.date
        )
    """),
    ("_f703_cand", """
        SELECT tg.trainer_id, tg.target_date, tj.jockey_id
        FROM _f703_tg tg
        JOIN (SELECT DISTINCT trainer_id, jockey_id FROM _f703_cum) tj
          ON tj.trainer_id = tg.trainer_id
    """),
    # target_date > date（厳密）: 対象日当日の騎乗は数えない
    ("_f703_prior", """
        SELECT c.trainer_id, c.target_date, c.jockey_id, cum.n_mounts
        FROM _f703_cand c
        ASOF JOIN _f703_cum cum
          ON c.trainer_id = cum.trainer_id AND c.jockey_id = cum.jockey_id
         AND c.target_date > cum.date
    """),
    ("_f703_main", """
        SELECT trainer_id, target_date, jockey_id AS main_jockey_id
        FROM (
            SELECT trainer_id, target_date, jockey_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY trainer_id, target_date ORDER BY n_mounts DESC, jockey_id
                   ) AS rn
            FROM _f703_prior
        )
        WHERE rn = 1
    """),
]

_MAIN_JOCKEY_SQL = """
SELECT t.race_id, t.horse_id, m.main_jockey_id
FROM (
    SELECT DISTINCT b.race_id, b.horse_id, r.date AS target_date, ru.trainer_id
    FROM base b
    JOIN races r ON r.race_id = b.race_id
    JOIN runners ru ON ru.race_id = b.race_id AND ru.horse_id = b.horse_id
    WHERE ru.trainer_id IS NOT NULL
) t
JOIN _f703_main m ON m.trainer_id = t.trainer_id AND m.target_date = t.target_date
"""


def _main_jockeys(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """`base`（登録済みビュー）の各出走馬の主戦騎手を返す。一時テーブルは必ず破棄する。"""
    try:
        for name, sql in _MAIN_JOCKEY_STAGES:
            conn.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {sql}")
        return conn.execute(_MAIN_JOCKEY_SQL).pl()
    finally:
        for name, _ in reversed(_MAIN_JOCKEY_STAGES):
            conn.execute(f"DROP TABLE IF EXISTS {name}")


def _travel_flag(affiliation: str | None, course: str | None) -> int | None:
    if affiliation not in _HOME_COURSES:
        return None
    return 0 if course in _HOME_COURSES[affiliation] else 1


def compute_f703(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    base = base.select(["race_id", "horse_id"])
    conn.register("base", base)
    try:
        targets = conn.execute(_TARGET_SQL).pl()
        main_jockey = _main_jockeys(conn)
    finally:
        conn.unregister("base")

    targets = targets.with_columns(
        pl.struct(["affiliation", "course"])
        .map_elements(
            lambda s: _travel_flag(s["affiliation"], s["course"]), return_dtype=pl.Int8,
        )
        .alias("f703_travel_flag")
    )
    targets = targets.with_columns(
        pl.when(pl.col("f703_travel_flag").is_null()).then(1).otherwise(0)
        .alias("f703_travel_flag_unavailable")
    )

    out = targets.join(main_jockey, on=["race_id", "horse_id"], how="left")
    has_main_jockey = pl.col("main_jockey_id").is_not_null()
    out = out.with_columns(
        pl.when(has_main_jockey)
        .then((pl.col("main_jockey_id") == pl.col("jockey_id")).cast(pl.Int8))
        .otherwise(None)
        .alias("f703_main_jockey_flag"),
        pl.when(has_main_jockey).then(0).otherwise(1).alias("f703_main_jockey_flag_unavailable"),
    )

    return out.select([
        "race_id", "horse_id",
        "f703_travel_flag", "f703_travel_flag_unavailable",
        "f703_main_jockey_flag", "f703_main_jockey_flag_unavailable",
    ])
