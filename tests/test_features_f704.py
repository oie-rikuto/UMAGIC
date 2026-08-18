"""`F-704` 騎手のコース適性（`docs/spec/003-features.md`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f704 import DEFAULT_K, compute_f704
from umagic.features.shrinkage import shrink

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)

J1, J2 = 701, 702


def _race(conn, race_id, race_date, course, distance=2000, n_starters=4):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, course, race_id, distance, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, finish_pos, jockey_id):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "jockey_id, source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, ?, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, finish_pos, jockey_id, NOW],
    )


def test_shrink_toward_course_bucket_mu_global(conn):
    # J1: 東京・中距離で2回、いずれも1着
    _race(conn, 1, date(2020, 1, 1), "東京", 2000)
    _runner(conn, 1, 10, 1, J1)
    _race(conn, 2, date(2020, 2, 1), "東京", 2000)
    _runner(conn, 2, 11, 1, J1)

    # J2: 東京・中距離で1回、4着
    _race(conn, 3, date(2020, 1, 15), "東京", 2000)
    _runner(conn, 3, 12, 4, J2)

    # 対象: J1騎乗、東京・中距離
    _race(conn, 4, date(2020, 3, 1), "東京", 2000)
    _runner(conn, 4, 20, None, J1)

    base = pl.DataFrame({"race_id": [4], "horse_id": [20]})
    out = compute_f704(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)

    # μ_global,c(東京,中距離) = (0.25+0.25+1.0)/3 = 0.5
    mu = 0.5
    expected = shrink(2, 0.25, k=DEFAULT_K, mu_global=mu)
    assert abs(row["f704"].to_list()[0] - expected) < 1e-9
    assert row["f704_unavailable"].to_list()[0] == 0


def test_null_jockey_is_unavailable(conn):
    _race(conn, 10, date(2020, 1, 1), "東京", 2000)
    _runner(conn, 10, 30, None, None)

    base = pl.DataFrame({"race_id": [10], "horse_id": [30]})
    out = compute_f704(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 30)
    assert row["f704"].to_list()[0] is None
    assert row["f704_unavailable"].to_list()[0] == 1


def test_integration_with_build_features(conn):
    _race(conn, 20, date(2020, 1, 1), "東京", 2000)
    _runner(conn, 20, 40, None, None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[20], feature_fns=[compute_f704])
    assert "f704" in df.columns
