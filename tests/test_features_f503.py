"""`F-503` 開催週次（`docs/spec/003-features.md` / `D-049` / `D-063`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f503 import compute_f503

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def test_passthrough_meeting_no_and_day(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, meeting_no, meeting_day, source, fetched_at) "
        "VALUES (1, ?, '東京', 1, 2000, '芝', 1, 1, 3, 8, 'netkeiba_jra', ?)",
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
    out = compute_f503(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["f503_meeting_no"].to_list()[0] == 3
    assert row["f503_meeting_day"].to_list()[0] == 8


def test_integration_with_build_features(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, meeting_no, meeting_day, source, fetched_at) "
        "VALUES (2, ?, '東京', 1, 2000, '芝', 1, 1, 1, 1, 'netkeiba_jra', ?)",
        [date(2020, 1, 1), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (20, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (2, 20, 1, '出走', 1, 100.0, 'netkeiba_jra', ?)", [NOW],
    )

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[2], feature_fns=[compute_f503])
    assert "f503_meeting_no" in df.columns and "f503_meeting_day" in df.columns
