"""`F-702` 乗り替わり（`docs/spec/003-features.md`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f702 import compute_f702

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', 1, 1, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, NOW],
    )


def _runner(conn, race_id, horse_id, jockey_id):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "jockey_id, source, fetched_at) VALUES (?, ?, 1, '出走', 1, 100.0, ?, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, jockey_id, NOW],
    )


def test_jockey_changed_and_experience(conn):
    _race(conn, 1, date(2020, 1, 1))
    _runner(conn, 1, 10, jockey_id=100)

    _race(conn, 2, date(2020, 2, 1))  # 対象。同じ騎手が2度目の騎乗
    _runner(conn, 2, 10, jockey_id=100)

    base = pl.DataFrame({"race_id": [2], "horse_id": [10]})
    out = compute_f702(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["f702_jockey_changed"].to_list()[0] == 0
    assert row["f702_jockey_experience_count"].to_list()[0] == 1


def test_jockey_change_detected(conn):
    _race(conn, 10, date(2020, 1, 1))
    _runner(conn, 10, 20, jockey_id=100)

    _race(conn, 11, date(2020, 2, 1))  # 対象。乗り替わり
    _runner(conn, 11, 20, jockey_id=200)

    base = pl.DataFrame({"race_id": [11], "horse_id": [20]})
    out = compute_f702(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)
    assert row["f702_jockey_changed"].to_list()[0] == 1
    assert row["f702_jockey_experience_count"].to_list()[0] == 0  # 新騎手なので経験0


def test_no_previous_race_is_structural(conn):
    _race(conn, 20, date(2020, 1, 1))
    _runner(conn, 20, 30, jockey_id=100)

    base = pl.DataFrame({"race_id": [20], "horse_id": [30]})
    out = compute_f702(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 30)
    assert row["f702_jockey_changed"].to_list()[0] is None
    assert row["f702_jockey_changed_unavailable"].to_list()[0] == 1
    assert row["f702_jockey_experience_count"].to_list()[0] == 0
