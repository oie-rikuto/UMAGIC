"""`per_row_stat_before` の dtype 安定性（`Q-047` 段階②で発見、`D-176`）。

先頭100行超（polars の既定 `infer_schema_length`）が `NULL` の後に実数値が
続くケースで `ComputeError` にならないことを確認する。JRAは同日に複数場が
並行開催されるため先頭行に実数値が混ざりやすく顕在化していなかったが、
1場のみで1日あたりの行数が多い母集団（大井のようなNAR）では、データ
ベースの立ち上がり直後に先頭100行が丸ごと `NULL` になりうる。
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from umagic.features.shrinkage import per_row_stat_before


def test_per_row_stat_before_handles_many_leading_nulls():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE runners (race_id BIGINT, last_3f DOUBLE)")
    conn.execute("CREATE TABLE races (race_id BIGINT, date DATE)")
    # 1件だけ実データを入れる（2023-01-01以降にのみ存在）。それより前を
    # 問い合わせる行は全て NULL になる
    conn.execute("INSERT INTO races VALUES (99999, '2023-01-01')")
    conn.execute("INSERT INTO runners VALUES (99999, 40.168456)")

    # 先頭120行（>100）は全て「これより前のデータが無い」= NULL になる行
    n_null_rows = 120
    race_ids = list(range(1, n_null_rows + 1))
    race_dates = {rid: date(2022, 1, 1) for rid in race_ids}
    # 最後の1行だけ、実データより後の日付にして NULL でない値を引かせる
    race_ids.append(100000)
    race_dates[100000] = date(2023, 1, 2)

    base = pl.DataFrame({"race_id": race_ids, "horse_id": [1] * len(race_ids)})
    result = per_row_stat_before(
        conn, base, race_dates=race_dates,
        stat_sql="SELECT AVG(ru.last_3f) FROM runners ru JOIN races r USING (race_id) "
                 "WHERE r.date < ? AND ru.last_3f IS NOT NULL",
    )

    assert result["stat"].dtype == pl.Float64
    assert result.filter(pl.col("race_id") == 100000)["stat"][0] == 40.168456
    assert result.filter(pl.col("race_id") == 1)["stat"][0] is None
