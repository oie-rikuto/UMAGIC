"""`F-803` レース基礎情報（`docs/spec/003-features.md` / `D-028`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f803 import compute_f803

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def test_passthrough_and_season(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "direction, n_entries, n_starters, prize, race_class, weight_rule, "
        "source, fetched_at) VALUES (1, ?, '東京', 1, 2000, '芝', '左', 8, 8, "
        "10000, '3勝クラス', '定量', 'netkeiba_jra', ?)",
        [date(2020, 4, 15), NOW],
    )
    conn.execute(
        "INSERT INTO horses VALUES (10, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (1, 10, 1, '出走', 1, 100.0, 'netkeiba_jra', ?)", [NOW],
    )

    base = pl.DataFrame({"race_id": [1], "horse_id": [10]})
    out = compute_f803(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)
    assert row["f803_distance"].to_list()[0] == 2000
    assert row["f803_surface"].to_list()[0] == "芝"
    assert row["f803_direction"].to_list()[0] == "左"
    assert row["f803_n_starters"].to_list()[0] == 8
    assert row["f803_season"].to_list()[0] == "春"
    assert row["f803_prize"].to_list()[0] == 10000
    assert row["f803_race_class"].to_list()[0] == "3勝クラス"
    assert row["f803_weight_rule"].to_list()[0] == "定量"
