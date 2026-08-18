"""`F-804` 当日の天候・馬場状態（`docs/spec/003-features.md` / `D-029` / `Q-021`）。

`weather`（実測）・`weather_forecast`（予報、`D-029`）・`track_condition`
（順序尺度 `良=0/稍重=1/重=2/不良=3`、`001-schema.md`）を出す。

**暫定/本命どちらを使うかはこの関数の責務ではない。** `weather` と
`weather_forecast` を別列のまま両方渡し、どちらを使うかは呼び出し側
（`FeatureRegistry.columns_for(route)` および予測パイプライン）が決める
（`D-029`: 実測と予報を同じ列に混ぜるとリークするため分離済み）。

**`weather_forecast` は木曜時点で入手できるが、`weather`/`track_condition`
は当日まで確定しない。** 本関数の3列は本来異なる `timing` を持つべきだが、
`FeatureSpec` は1つの特徴量に1つの `timing` しか持てず、`F-804` を
レジストリに登録する際にどう扱うかは未整理（`P-1` ではどの `F-xxx` も
まだレジストリに実登録していないため、現時点でブロックしない）。

`race_level`: `True`。確定時刻は `当日`（`weather_forecast` を除く）。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

_SQL = """
SELECT
    b.race_id, b.horse_id,
    r.weather AS f804_weather,
    r.weather_forecast AS f804_weather_forecast,
    CASE r.track_condition
        WHEN '良' THEN 0
        WHEN '稍重' THEN 1
        WHEN '重' THEN 2
        WHEN '不良' THEN 3
        ELSE NULL
    END AS f804_track_condition
FROM base b
JOIN races r ON r.race_id = b.race_id
"""


def compute_f804(
    conn: duckdb.DuckDBPyConnection, base: pl.DataFrame, *, as_of: date,
) -> pl.DataFrame:
    """`FeatureFn` 互換。`base` は `(race_id, horse_id)` を持つ。"""
    conn.register("base", base.select(["race_id", "horse_id"]))
    try:
        return conn.execute(_SQL).pl()
    finally:
        conn.unregister("base")
