"""`F-303` 上がり3F関連（`docs/spec/003-features.md`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f303 import DEFAULT_K, compute_f303
from umagic.features.shrinkage import shrink

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, n_starters):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, last_3f):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "last_3f, source, fetched_at) VALUES (?, ?, ?, '出走', 1, 100.0, ?, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, last_3f, NOW],
    )


def test_full_recent_and_rank(conn):
    _race(conn, 1, date(2020, 1, 1), 3)
    _runner(conn, 1, 10, 35.0)  # H
    _runner(conn, 1, 11, 33.0)  # A（H より速い）
    _runner(conn, 1, 12, 37.0)  # B（H より遅い）

    _race(conn, 2, date(2020, 2, 1), 2)
    _runner(conn, 2, 10, 34.0)  # H
    _runner(conn, 2, 13, 36.0)  # C（H より遅い）

    _race(conn, 3, date(2020, 3, 1), 1)
    _runner(conn, 3, 10, None)  # 対象レース

    base = pl.DataFrame({"race_id": [3], "horse_id": [10]})
    out = compute_f303(conn, base, as_of=date(2025, 1, 1), n_recent=1)
    row = out.filter(pl.col("horse_id") == 10)

    # μ_global = (35+33+37+34+36)/5 = 35.0（対象日より前の全 last_3f の平均）
    mu = 35.0
    expected_all = shrink(2, 34.5, k=DEFAULT_K, mu_global=mu)
    expected_recent = shrink(1, 34.0, k=DEFAULT_K, mu_global=mu)  # 直近1走=R2
    expected_rank = ((2 / 3) + (1 / 2)) / 2  # R1: 2位/3頭, R2: 1位/2頭

    assert abs(row["last3f_all"].to_list()[0] - expected_all) < 1e-9
    assert abs(row["last3f_recent"].to_list()[0] - expected_recent) < 1e-9
    assert abs(row["last3f_rank_in_race"].to_list()[0] - expected_rank) < 1e-9
    assert row["last3f_all_unavailable"].to_list()[0] == 0
    assert row["last3f_recent_unavailable"].to_list()[0] == 0
    assert row["last3f_rank_in_race_unavailable"].to_list()[0] == 0


def test_no_history_is_not_structural(conn):
    _race(conn, 100, date(2020, 1, 1), 1)
    _runner(conn, 100, 20, None)

    base = pl.DataFrame({"race_id": [100], "horse_id": [20]})
    out = compute_f303(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)
    assert row["last3f_all"].to_list()[0] is None
    assert row["last3f_all_unavailable"].to_list()[0] == 0


def test_past_races_without_last3f_is_structural(conn):
    """過去走はあるが `last_3f` が1件も記録されていない → NaN, unavailable=1。"""
    _race(conn, 200, date(2019, 1, 1), 1)
    _runner(conn, 200, 30, None)  # last_3f 欠損の過去走

    _race(conn, 201, date(2020, 1, 1), 1)
    _runner(conn, 201, 30, None)  # 対象レース

    base = pl.DataFrame({"race_id": [201], "horse_id": [30]})
    out = compute_f303(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 30)
    assert row["last3f_all"].to_list()[0] is None
    assert row["last3f_all_unavailable"].to_list()[0] == 1
    assert row["last3f_recent_unavailable"].to_list()[0] == 1
    assert row["last3f_rank_in_race_unavailable"].to_list()[0] == 1


def test_integration_with_build_features(conn):
    _race(conn, 300, date(2020, 1, 1), 1)
    _runner(conn, 300, 40, None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[300], feature_fns=[compute_f303])
    assert "last3f_all" in df.columns and "last3f_rank_in_race" in df.columns
