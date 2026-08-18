"""`F-603` 馬体重（`docs/spec/003-features.md`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f603 import compute_f603

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _setup(conn, race_id, horse_id, horse_weight, weight_diff):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', 1, 2000, '芝', 1, 1, 'netkeiba_jra', ?)",
        [race_id, date(2020, 1, 1), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
        [horse_id, NOW],
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "horse_weight, weight_diff, source, fetched_at) VALUES (?, ?, 1, '出走', 1, 100.0, "
        "?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_weight, weight_diff, NOW],
    )


def test_passthrough(conn):
    _setup(conn, 1, 10, 480, -4)
    base = pl.DataFrame({"race_id": [1], "horse_id": [10]})
    out = compute_f603(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["f603_horse_weight"].to_list()[0] == 480
    assert row["f603_weight_diff"].to_list()[0] == -4
    assert row["f603_horse_weight_unavailable"].to_list()[0] == 0


def test_null_weight_is_unavailable(conn):
    _setup(conn, 1, 10, None, None)
    base = pl.DataFrame({"race_id": [1], "horse_id": [10]})
    out = compute_f603(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["f603_horse_weight"].to_list()[0] is None
    assert row["f603_horse_weight_unavailable"].to_list()[0] == 1
    assert row["f603_weight_diff_unavailable"].to_list()[0] == 1
