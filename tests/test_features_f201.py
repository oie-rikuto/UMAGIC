"""`F-201` カテゴリID passthrough（`docs/spec/003-features.md` / `D-062`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f201 import compute_f201

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def test_passthrough_ids(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (1, ?, '東京', 1, 2000, '芝', 1, 1, 'netkeiba_jra', ?)", [date(2020, 1, 1), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (10, 'h', NULL, 900, NULL, 901, 'netkeiba_jra', ?)", [NOW]
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "jockey_id, trainer_id, source, fetched_at) VALUES (1, 10, 1, '出走', 1, 100.0, "
        "700, 800, 'netkeiba_jra', ?)", [NOW],
    )

    base = pl.DataFrame({"race_id": [1], "horse_id": [10]})
    out = compute_f201(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["sire_id"].to_list()[0] == 900
    assert row["damsire_id"].to_list()[0] == 901
    assert row["jockey_id"].to_list()[0] == 700
    assert row["trainer_id"].to_list()[0] == 800


def test_null_sire_passthrough(conn):
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
    out = compute_f201(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["sire_id"].to_list()[0] is None
