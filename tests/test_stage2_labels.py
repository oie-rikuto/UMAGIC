"""Stage 2 のラベルと学習母集団（`docs/spec/007-stage2-ranker.md` 1〜3節 / `D-093` `D-094`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.stage2 import build_labels, race_group

NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


def _race(conn, race_id, n_starters=4):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, date(2020, 1, 1), race_id, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, number, status="出走", finish_pos=None):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, ?, ?, ?, 100.0, 'netkeiba_jra', ?)",
        [race_id, horse_id, number, status, finish_pos, NOW],
    )


def test_finish_pos_maps_to_staged_labels(conn):
    """観点3: finish_pos 1,2,3,4 → label 3,2,1,0。"""
    _race(conn, 1, n_starters=4)
    for i, (h, fp) in enumerate([(10, 1), (11, 2), (12, 3), (13, 4)]):
        _runner(conn, 1, h, i + 1, finish_pos=fp)

    out = build_labels(conn, [1]).sort("horse_id")
    assert out["label"].to_list() == [3, 2, 1, 0]


def test_dead_heat_both_get_top_label(conn):
    """観点4: 1着同着2頭 → 両方 label=3。"""
    _race(conn, 1, n_starters=2)
    _runner(conn, 1, 10, 1, finish_pos=1)
    _runner(conn, 1, 11, 2, finish_pos=1)

    out = build_labels(conn, [1])
    assert out["label"].to_list() == [3, 3]


def test_suspended_horse_gets_zero_label_and_is_included(conn):
    """観点5: 競走中止は label=0 で学習に含まれる（D-094）。"""
    _race(conn, 1, n_starters=2)
    _runner(conn, 1, 10, 1, status="出走", finish_pos=1)
    _runner(conn, 1, 11, 2, status="競走中止", finish_pos=None)

    out = build_labels(conn, [1]).sort("horse_id")
    assert out["horse_id"].to_list() == [10, 11]
    assert out["label"].to_list() == [3, 0]


def test_scratched_horse_excluded(conn):
    """観点6: 出走取消は学習（build_labels）に現れない（D-094）。"""
    _race(conn, 1, n_starters=2)
    _runner(conn, 1, 10, 1, status="出走", finish_pos=1)
    _runner(conn, 1, 11, 2, status="出走取消", finish_pos=None)

    out = build_labels(conn, [1])
    assert out["horse_id"].to_list() == [10]


def test_group_sum_matches_row_count(conn):
    """観点7: group の合計が学習行数と一致する。"""
    _race(conn, 1, n_starters=3)
    _race(conn, 2, n_starters=5)
    for i, h in enumerate((10, 11, 12)):
        _runner(conn, 1, h, i + 1, finish_pos=i + 1)
    for i, h in enumerate((20, 21, 22, 23, 24)):
        _runner(conn, 2, h, i + 1, finish_pos=i + 1)

    labels = build_labels(conn, [1, 2]).sort(["race_id", "horse_id"])
    group = race_group(labels)
    assert group.sum() == labels.height
    assert group.to_list() == [3, 5]


def test_empty_race_ids(conn):
    out = build_labels(conn, [])
    assert out.is_empty()
