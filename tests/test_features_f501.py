"""`F-501` 当日の脚質バイアス（`docs/spec/003-features.md` / `D-010`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f501 import compute_f501

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, course, surface, race_number, n_starters=4):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, ?, ?, 2000, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, course, race_number, surface, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, finish_pos, corners):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "corners, source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, ?, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, finish_pos, corners, NOW],
    )


def test_uses_only_earlier_same_day_same_course_same_surface_top3(conn):
    d = date(2020, 1, 1)
    # 対象と同日・同競馬場・同馬場、対象より前(1R)の上位3頭
    _race(conn, 1, d, "東京", "芝", 1)
    _runner(conn, 1, 101, 1, [1, 1, 1, 1])  # 4角 pos=1/4=0.25
    _runner(conn, 1, 102, 2, [1, 2, 2, 2])  # pos=2/4=0.5
    _runner(conn, 1, 103, 3, [1, 2, 3, 4])  # pos=4/4=1.0
    _runner(conn, 1, 104, 4, [1, 4, 4, 4])  # 4着なので対象外

    # 別競馬場（同日・同race_number以前）→ 含めない
    _race(conn, 2, d, "阪神", "芝", 1)
    _runner(conn, 2, 201, 1, [1, 1, 1, 1])

    # 同競馬場・別馬場（ダート、2R）→ 含めない
    _race(conn, 3, d, "東京", "ダート", 2)
    _runner(conn, 3, 301, 1, [1, 1, 1, 1])

    # 対象レース（3R）
    _race(conn, 4, d, "東京", "芝", 3)
    _runner(conn, 4, 401, None, None)

    # 対象より後（4R）→ リークするので含めない
    _race(conn, 5, d, "東京", "芝", 4)
    _runner(conn, 5, 501, 1, [1, 1, 1, 1])

    base = pl.DataFrame({"race_id": [4], "horse_id": [401]})
    out = compute_f501(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 401)

    expected = (0.25 + 0.5 + 1.0) / 3
    assert abs(row["f501"].to_list()[0] - expected) < 1e-9
    assert row["f501_unavailable"].to_list()[0] == 0


def test_first_race_of_day_is_unavailable(conn):
    d = date(2020, 1, 1)
    _race(conn, 10, d, "東京", "芝", 1)
    _runner(conn, 10, 1001, None, None)

    base = pl.DataFrame({"race_id": [10], "horse_id": [1001]})
    out = compute_f501(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 1001)
    assert row["f501"].to_list()[0] is None
    assert row["f501_unavailable"].to_list()[0] == 1


def test_integration_with_build_features(conn):
    d = date(2020, 1, 1)
    _race(conn, 20, d, "東京", "芝", 1)
    _runner(conn, 20, 2001, None, None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[20], feature_fns=[compute_f501])
    assert "f501" in df.columns and "f501_unavailable" in df.columns
