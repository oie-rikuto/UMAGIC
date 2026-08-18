"""`F-601` 反動リスク（`docs/spec/003-features.md` / `D-034`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f601 import compute_f601

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, n_starters=2):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, finish_pos, margin=None, status="出走", last_3f=None):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "margin, last_3f, source, fetched_at) VALUES (?, ?, ?, ?, ?, 100.0, ?, ?, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, status, finish_pos, margin, last_3f, NOW],
    )


def test_uses_most_recent_previous_race(conn):
    _race(conn, 1, date(2020, 1, 1))
    _runner(conn, 1, 10, 2, margin="1/2", last_3f=35.0)
    _runner(conn, 1, 11, 1, margin=None, last_3f=33.0)  # last3f順位を作るための対戦相手

    _race(conn, 2, date(2020, 3, 1))  # 対象レース
    _runner(conn, 2, 10, None)

    base = pl.DataFrame({"race_id": [2], "horse_id": [10]})
    out = compute_f601(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)

    assert row["f601_finish_pos_prev"].to_list()[0] == 2
    assert row["f601_margin_prev"].to_list()[0] == 0.5
    assert row["f601_last3f_rank_prev"].to_list()[0] == 1.0  # 2頭中2位 → 2/2
    for col in ["f601_finish_pos_prev", "f601_margin_prev", "f601_last3f_rank_prev"]:
        assert row[f"{col}_unavailable"].to_list()[0] == 0


def test_no_previous_race_is_structural(conn):
    _race(conn, 10, date(2020, 1, 1))
    _runner(conn, 10, 20, None)

    base = pl.DataFrame({"race_id": [10], "horse_id": [20]})
    out = compute_f601(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)
    for col in ["f601_finish_pos_prev", "f601_margin_prev", "f601_last3f_rank_prev"]:
        assert row[col].to_list()[0] is None
        assert row[f"{col}_unavailable"].to_list()[0] == 1


def test_exclude_relegated_falls_back_to_earlier_race(conn):
    _race(conn, 20, date(2020, 1, 1))
    _runner(conn, 20, 30, 3, margin="1")  # 通常の前々走

    _race(conn, 21, date(2020, 2, 1))
    _runner(conn, 21, 30, 4, margin=None, status="降着")  # 直近走は降着（着差欄は空）

    _race(conn, 22, date(2020, 3, 1))  # 対象
    _runner(conn, 22, 30, None)

    base = pl.DataFrame({"race_id": [22], "horse_id": [30]})

    out_default = compute_f601(conn, base, as_of=date(2025, 1, 1))
    row_default = out_default.filter(pl.col("horse_id") == 30)
    assert row_default["f601_finish_pos_prev"].to_list()[0] == 4  # 既定は降着走をそのまま使う

    out_excluded = compute_f601(conn, base, as_of=date(2025, 1, 1), exclude_relegated=True)
    row_excluded = out_excluded.filter(pl.col("horse_id") == 30)
    assert row_excluded["f601_finish_pos_prev"].to_list()[0] == 3  # 1つ前の走に遡る


def test_integration_with_build_features(conn):
    _race(conn, 30, date(2020, 1, 1))
    _runner(conn, 30, 40, None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[30], feature_fns=[compute_f601])
    assert "f601_finish_pos_prev" in df.columns
