"""`F-802` コース・距離適性（`docs/spec/003-features.md`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f802 import DEFAULT_K, compute_f802
from umagic.features.shrinkage import shrink

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, course="東京", distance=2000):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, '芝', 4, 4, 'netkeiba_jra', ?)",
        [race_id, race_date, course, race_id, distance, NOW],
    )


def _runner(conn, race_id, horse_id, finish_pos):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, finish_pos, NOW],
    )


def test_shrink_toward_bucket_mu_global(conn):
    course, distance = "東京", 2000
    # 対象馬 H 自身: 同コース・同距離帯で2走とも1着
    _race(conn, 1, date(2020, 1, 1), course, distance)
    _runner(conn, 1, 10, 1)
    _race(conn, 2, date(2020, 2, 1), course, distance)
    _runner(conn, 2, 10, 1)

    # 他馬: 同条件で4着
    _race(conn, 3, date(2020, 1, 15), course, distance)
    _runner(conn, 3, 11, 4)

    _race(conn, 4, date(2020, 3, 1), course, distance)
    _runner(conn, 4, 10, None)

    base = pl.DataFrame({"race_id": [4], "horse_id": [10]})
    out = compute_f802(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)

    mu = (1 / 4 + 1 / 4 + 1.0) / 3
    expected = shrink(2, 1 / 4, k=DEFAULT_K, mu_global=mu)
    assert abs(row["f802"].to_list()[0] - expected) < 1e-9


def test_integration_with_build_features(conn):
    _race(conn, 20, date(2020, 1, 1))
    _runner(conn, 20, 40, None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[20], feature_fns=[compute_f802])
    assert "f802" in df.columns
