"""`F-804` 当日の天候・馬場状態（`docs/spec/003-features.md` / `D-029`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f804 import compute_f804

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def test_weather_and_track_condition_ordinal(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "weather, weather_forecast, track_condition, n_entries, n_starters, "
        "source, fetched_at) VALUES (1, ?, '東京', 1, 2000, '芝', "
        "'晴', '曇', '稍重', 1, 1, 'netkeiba_jra', ?)",
        [date(2020, 1, 1), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (10, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (1, 10, 1, '出走', 1, 100.0, 'netkeiba_jra', ?)", [NOW],
    )

    base = pl.DataFrame({"race_id": [1], "horse_id": [10]})
    out = compute_f804(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["f804_weather"].to_list()[0] == "晴"
    assert row["f804_weather_forecast"].to_list()[0] == "曇"
    assert row["f804_track_condition"].to_list()[0] == 1  # 稍重


def test_null_track_condition(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (2, ?, '東京', 1, 2000, '芝', 1, 1, 'netkeiba_jra', ?)",
        [date(2020, 1, 1), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (20, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (2, 20, 1, '出走', 1, 100.0, 'netkeiba_jra', ?)", [NOW],
    )

    base = pl.DataFrame({"race_id": [2], "horse_id": [20]})
    out = compute_f804(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)
    assert row["f804_track_condition"].to_list()[0] is None
