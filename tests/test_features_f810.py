"""`F-810` 馬主ID passthrough（`docs/domain-knowledge.md` / `D-165`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f810 import compute_f810

NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)


def test_passthrough_owner_id(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (1, ?, '東京', 1, 2000, '芝', 1, 1, 'netkeiba_jra', ?)", [date(2020, 1, 1), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (10, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "owner_id, source, fetched_at) VALUES (1, 10, 1, '出走', 1, 100.0, "
        "500, 'netkeiba_jra', ?)", [NOW],
    )

    base = pl.DataFrame({"race_id": [1], "horse_id": [10]})
    out = compute_f810(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["f810_owner_id"].to_list()[0] == 500


def test_null_owner_passthrough(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (1, ?, '東京', 1, 2000, '芝', 1, 1, 'netkeiba_jra', ?)", [date(2020, 1, 1), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (10, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (1, 10, 1, '出走', 1, 100.0, 'netkeiba_jra', ?)", [NOW],
    )

    base = pl.DataFrame({"race_id": [1], "horse_id": [10]})
    out = compute_f810(conn, base, as_of=date(2025, 1, 1))
    assert out["f810_owner_id"].to_list()[0] is None


def test_integration_with_build_features(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (1, ?, '東京', 1, 2000, '芝', 1, 1, 'netkeiba_jra', ?)", [date(2020, 1, 1), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (10, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "owner_id, source, fetched_at) VALUES (1, 10, 1, '出走', 1, 100.0, "
        "500, 'netkeiba_jra', ?)", [NOW],
    )

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[1], feature_fns=[compute_f810])
    assert "f810_owner_id" in df.columns
