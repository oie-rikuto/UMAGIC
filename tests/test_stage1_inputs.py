"""Stage 1 の入力（`docs/spec/006-stage1-pace.md` 2節 / `D-089`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

from umagic.stage1 import FEATURE_COLS, build_inputs

NOW = datetime(2026, 8, 21, tzinfo=timezone.utc)


def _race(conn, race_id, race_date=date(2020, 1, 1), n_starters=6, corner_nos=None,
          weather="晴", track_condition="良"):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "direction, race_class, n_entries, n_starters, corner_nos, weather, "
        "track_condition, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', '左', '3勝クラス', ?, ?, ?, ?, ?, "
        "'netkeiba_jra', ?)",
        [race_id, race_date, race_id, n_starters, n_starters, corner_nos, weather,
         track_condition, NOW],
    )


def _runner(conn, race_id, horse_id, number, status="出走", finish_pos=1, corners=None):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "corners, source, fetched_at) VALUES (?, ?, ?, ?, ?, 100.0, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, number, status, finish_pos, corners, NOW],
    )


def _seed_past_f101(conn, horse_id, ratio, past_race_id, n_starters_past=20):
    """`horse_id` の `F-101` が `ratio` になる過去走を1本仕込む。"""
    _race(conn, past_race_id, date(2019, 1, 1), n_starters=n_starters_past,
          corner_nos=[1, 2, 3, 4])
    pos = round(ratio * n_starters_past)
    _runner(conn, past_race_id, horse_id, 1, corners=[pos, pos, pos, pos])


def test_all_horses_missing_f101(conn):
    """観点8: F-101が全馬欠損（過去走無し）→ min/mean/q25がNaN、n_missing=n_starters。"""
    _race(conn, 1, n_starters=3, corner_nos=[1, 2, 3, 4])
    for h in (10, 11, 12):
        _runner(conn, 1, h, h % 100)

    out = build_inputs(conn, [1], as_of=date(2025, 1, 1))
    row = out.filter(out["race_id"] == 1)
    assert row["f101_min"].to_list()[0] is None
    assert row["f101_mean"].to_list()[0] is None
    assert row["f101_q25"].to_list()[0] is None
    assert row["f101_n_missing"].to_list()[0] == 3


def test_distribution_shape_differs_at_same_mean(conn):
    """観点9: 「逃げ馬1頭+差し馬多数」と「先行馬揃い」で平均が同値でも min/q25 が異なる（D-089の動機）。"""
    # レースA: 逃げ馬1頭(0.05) + 差し馬5頭(0.95) → 平均=0.8, 最小=0.05
    _race(conn, 1, n_starters=6, corner_nos=[1, 2, 3, 4])
    _seed_past_f101(conn, 100, 0.05, past_race_id=900)
    _runner(conn, 1, 100, 1)
    for i, h in enumerate((101, 102, 103, 104, 105)):
        _seed_past_f101(conn, h, 0.95, past_race_id=901 + i)
        _runner(conn, 1, h, i + 2)

    # レースB: 先行馬6頭が揃って0.8 → 平均=0.8, 最小=0.8
    _race(conn, 2, n_starters=6, corner_nos=[1, 2, 3, 4])
    for i, h in enumerate((200, 201, 202, 203, 204, 205)):
        _seed_past_f101(conn, h, 0.8, past_race_id=910 + i)
        _runner(conn, 2, h, i + 1)

    out = build_inputs(conn, [1, 2], as_of=date(2025, 1, 1)).sort("race_id")
    mean_a, mean_b = out["f101_mean"].to_list()
    min_a, min_b = out["f101_min"].to_list()
    assert abs(mean_a - mean_b) < 1e-6  # 平均は同値
    assert abs(min_a - min_b) > 0.1     # 最小値ははっきり異なる
    assert min_a < 0.1
    assert abs(min_b - 0.8) < 1e-6


def test_no_lap_columns_in_inputs(conn):
    """観点12: laps 由来の列を1つも含まない（原則5）。"""
    _race(conn, 1, n_starters=4, corner_nos=[1, 2, 3, 4])
    for h in (10, 11, 12, 13):
        _runner(conn, 1, h, h % 100)
    out = build_inputs(conn, [1], as_of=date(2025, 1, 1))
    assert "lap_sec" not in out.columns
    assert "furlong_no" not in out.columns
    for c in out.columns:
        assert "lap" not in c.lower()


def test_no_odds_or_popularity_in_inputs(conn):
    """観点13: odds_win / popularity を1つも含まない（R-018）。"""
    _race(conn, 1, n_starters=4, corner_nos=[1, 2, 3, 4])
    for h in (10, 11, 12, 13):
        _runner(conn, 1, h, h % 100)
    out = build_inputs(conn, [1], as_of=date(2025, 1, 1))
    assert "odds_win" not in out.columns
    assert "popularity" not in out.columns


def test_feature_cols_subset_of_output(conn):
    _race(conn, 1, n_starters=4, corner_nos=[1, 2, 3, 4])
    for h in (10, 11, 12, 13):
        _runner(conn, 1, h, h % 100)
    out = build_inputs(conn, [1], as_of=date(2025, 1, 1))
    assert set(FEATURE_COLS) <= set(out.columns)
