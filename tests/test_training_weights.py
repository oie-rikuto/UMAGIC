"""`sample_weight`（`docs/spec/014-training-pipeline.md` 2節 / `D-081`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

from umagic.training import sample_weights

NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)


def _race(conn, race_id, grade=None):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "grade, n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, 1, 1, 'netkeiba_jra', ?)",
        [race_id, date(2020, 1, 1), race_id, grade, NOW],
    )


def test_class_weights_applied(conn):
    """観点9: G1のレースが指定重み、G2/G3/L/NULLが1.0。"""
    _race(conn, 1, "G1")
    _race(conn, 2, "G2")
    _race(conn, 3, None)

    out = sample_weights(
        conn, [1, 2, 3], class_weights={"G1": 5.0, "G2": 3.0, "G3": 2.0, "L": 1.5},
    ).sort("race_id")
    assert out["sample_weight"].to_list() == [5.0, 3.0, 1.0]


def test_unknown_class_falls_back_to_one(conn):
    """観点10: class_weights に無いクラスは1.0にフォールバックする。例外にしない。"""
    _race(conn, 1, "JpnI")  # Q-034: class_weights にキーが無い想定のクラス
    out = sample_weights(conn, [1], class_weights={"G1": 5.0})
    assert out["sample_weight"].to_list() == [1.0]


def test_empty_race_ids(conn):
    out = sample_weights(conn, [], class_weights={"G1": 5.0})
    assert out.is_empty()
