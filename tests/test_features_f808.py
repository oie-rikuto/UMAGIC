"""`F-808` 騎手×馬のコンビ成績（`D-140`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f808 import compute_f808

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, race_number, n_starters):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_number, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, jockey_id, *, finish_pos, status="出走"):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, jockey_id, status, finish_pos, "
        "source, fetched_at) VALUES (?, ?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, jockey_id, status, finish_pos, NOW],
    )


def test_first_time_combo_is_all_null(conn):
    """初コンビ（過去の騎乗歴が無い）は全列欠損だが `unavailable=0`。"""
    _race(conn, 1, date(2020, 3, 1), 1, 4)
    _runner(conn, 1, 10, jockey_id=1, finish_pos=1)

    out = compute_f808(conn, pl.DataFrame({"race_id": [1], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f808_relfinish_mean"] is None
    assert row["f808_win_rate"] is None
    assert row["f808_unavailable"] == 0


def test_aggregates_only_matching_jockey(conn):
    """同じ馬でも別の騎手の騎乗は集約に入らない。"""
    _race(conn, 1, date(2020, 1, 1), 1, 4)
    _runner(conn, 1, 10, jockey_id=1, finish_pos=1)   # 対象騎手・1着
    _race(conn, 2, date(2020, 2, 1), 1, 4)
    _runner(conn, 2, 10, jockey_id=2, finish_pos=4)   # 別騎手・4着（入らない）
    _race(conn, 3, date(2020, 3, 1), 1, 4)
    _runner(conn, 3, 10, jockey_id=1, finish_pos=1)   # 対象レース（同じ騎手）

    out = compute_f808(conn, pl.DataFrame({"race_id": [3], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f808_relfinish_mean"] == 0.25   # 1/4（レース1のみ）
    assert row["f808_win_rate"] == 1.0


def test_relfinish_is_bounded_and_win_rate_matches(conn):
    _race(conn, 1, date(2020, 1, 1), 1, 5)
    _runner(conn, 1, 10, jockey_id=1, finish_pos=3)
    _race(conn, 2, date(2020, 2, 1), 1, 4)
    _runner(conn, 2, 10, jockey_id=1, finish_pos=1)
    _race(conn, 3, date(2020, 3, 1), 1, 4)
    _runner(conn, 3, 10, jockey_id=1, finish_pos=1)

    out = compute_f808(conn, pl.DataFrame({"race_id": [3], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    # 過去2走: 3/5=0.6, 1/4=0.25 → mean=0.425, best=0.25, win_rate=1/2=0.5
    assert row["f808_relfinish_mean"] == 0.425
    assert row["f808_relfinish_best"] == 0.25
    assert row["f808_win_rate"] == 0.5
    assert 0.0 <= row["f808_relfinish_mean"] <= 1.0


def test_excludes_same_day_and_future(conn):
    """`race_date < target_race_date` で厳密に切る（`D-054` 原則7）。"""
    _race(conn, 1, date(2020, 3, 1), 1, 4)
    _runner(conn, 1, 10, jockey_id=1, finish_pos=1)   # 同日の別レース
    _race(conn, 2, date(2020, 3, 1), 2, 4)
    _runner(conn, 2, 10, jockey_id=1, finish_pos=1)   # 対象レース（同日）

    out = compute_f808(conn, pl.DataFrame({"race_id": [2], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    assert out.row(0, named=True)["f808_relfinish_mean"] is None


def test_empty_base_returns_schema(conn):
    out = compute_f808(conn, pl.DataFrame({"race_id": [], "horse_id": []}),
                       as_of=date(2025, 1, 1))
    assert out.is_empty()
    for c in ("f808_relfinish_mean", "f808_relfinish_best", "f808_win_rate", "f808_unavailable"):
        assert c in out.columns
