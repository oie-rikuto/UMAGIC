"""`F-801` 枠順バイアス（`docs/spec/003-features.md` / `D-070`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f801 import DEFAULT_K, compute_f801
from umagic.features.shrinkage import shrink

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, course="東京", distance=2000, n_starters=8):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, course, race_id, distance, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, finish_pos, frame):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, frame, status, finish_pos, "
        "time_sec, source, fetched_at) VALUES (?, ?, ?, ?, '出走', ?, 100.0, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, frame, finish_pos, NOW],
    )


def test_shrink_toward_bucket_mu_global(conn):
    course, distance, n_starters = "東京", 2000, 8
    # 同条件・1枠: 2走とも1着
    _race(conn, 1, date(2020, 1, 1), course, distance, n_starters)
    _runner(conn, 1, 10, 1, frame=1)
    _race(conn, 2, date(2020, 2, 1), course, distance, n_starters)
    _runner(conn, 2, 11, 1, frame=1)

    # 同条件・8枠: 1走で最下位
    _race(conn, 3, date(2020, 1, 15), course, distance, n_starters)
    _runner(conn, 3, 12, 8, frame=8)

    # 対象: 1枠、同条件
    _race(conn, 4, date(2020, 3, 1), course, distance, n_starters)
    _runner(conn, 4, 20, None, frame=1)

    base = pl.DataFrame({"race_id": [4], "horse_id": [20]})
    out = compute_f801(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)

    # x̄(1枠) = (1/8+1/8)/2 = 0.125, μ_global,c = (0.125+0.125+1.0)/3 = 0.41666...
    mu = (1 / 8 + 1 / 8 + 1.0) / 3
    expected = shrink(2, 1 / 8, k=DEFAULT_K, mu_global=mu)
    assert abs(row["f801"].to_list()[0] - expected) < 1e-9
    assert row["f801_unavailable"].to_list()[0] == 0


def test_null_frame_is_unavailable(conn):
    _race(conn, 10, date(2020, 1, 1))
    _runner(conn, 10, 30, None, frame=None)

    base = pl.DataFrame({"race_id": [10], "horse_id": [30]})
    out = compute_f801(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 30)
    assert row["f801"].to_list()[0] is None
    assert row["f801_unavailable"].to_list()[0] == 1


def test_integration_with_build_features(conn):
    _race(conn, 20, date(2020, 1, 1))
    _runner(conn, 20, 40, None, frame=1)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[20], feature_fns=[compute_f801])
    assert "f801" in df.columns
