"""`F-602` ローテーション（`docs/spec/003-features.md`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f602 import compute_f602

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, grade=None):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, grade, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', 1, 1, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, grade, NOW],
    )


def _runner(conn, race_id, horse_id):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, 1, '出走', 1, 100.0, 'netkeiba_jra', ?)",
        [race_id, horse_id, NOW],
    )


def test_weeks_since_last_and_prev_grade(conn):
    _race(conn, 1, date(2020, 1, 1), grade="G1")
    _runner(conn, 1, 10)

    _race(conn, 2, date(2020, 1, 15))  # 対象（2週間後）
    _runner(conn, 2, 10)

    base = pl.DataFrame({"race_id": [2], "horse_id": [10]})
    out = compute_f602(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["f602_weeks_since_last"].to_list()[0] == 2.0
    assert row["f602_prev_grade"].to_list()[0] == "G1"
    assert row["f602_weeks_since_last_unavailable"].to_list()[0] == 0


def test_no_previous_race_is_structural(conn):
    _race(conn, 10, date(2020, 1, 1))
    _runner(conn, 10, 20)

    base = pl.DataFrame({"race_id": [10], "horse_id": [20]})
    out = compute_f602(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)
    assert row["f602_weeks_since_last"].to_list()[0] is None
    assert row["f602_weeks_since_last_unavailable"].to_list()[0] == 1
    assert row["f602_prev_grade_unavailable"].to_list()[0] == 1
