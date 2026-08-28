"""`F-809` 馬のキャリア成績率（`D-145`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f809 import compute_f809

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, race_number, n_starters=8):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_number, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, finish_pos, *, status="出走"):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, "
        "source, fetched_at) VALUES (?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, status, finish_pos, NOW],
    )


def test_first_start_is_null(conn):
    """初出走は両列とも欠損（指示子は置かない）。"""
    _race(conn, 1, date(2020, 3, 1), 1)
    _runner(conn, 1, 10, finish_pos=1)

    out = compute_f809(conn, pl.DataFrame({"race_id": [1], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f809_win_rate"] is None
    assert row["f809_top3_rate"] is None


def test_win_and_top3_rates(conn):
    """勝率・複勝率が過去走から正しく計算される。"""
    for i, pos in enumerate([1, 5, 3, 8], start=1):
        _race(conn, i, date(2020, i, 1), 1)
        _runner(conn, i, 10, finish_pos=pos)
    _race(conn, 9, date(2020, 6, 1), 1)
    _runner(conn, 9, 10, finish_pos=1)  # 対象レース

    out = compute_f809(conn, pl.DataFrame({"race_id": [9], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    # 過去4走: 1着,5着,3着,8着 → 勝率1/4=0.25、複勝率2/4=0.5
    assert row["f809_win_rate"] == 0.25
    assert row["f809_top3_rate"] == 0.5


def test_rates_are_bounded(conn):
    """率は [0,1] に有界（`D-108`/`D-116` の発散問題を構造的に避ける）。"""
    for i in range(1, 4):
        _race(conn, i, date(2020, i, 1), 1)
        _runner(conn, i, 10, finish_pos=1)
    _race(conn, 9, date(2020, 6, 1), 1)
    _runner(conn, 9, 10, finish_pos=1)

    out = compute_f809(conn, pl.DataFrame({"race_id": [9], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f809_win_rate"] == 1.0
    assert row["f809_top3_rate"] == 1.0
    assert 0.0 <= row["f809_win_rate"] <= 1.0


def test_excludes_same_day_and_future(conn):
    """`race_date < target_race_date` で厳密に切る（`D-054` 原則7）。"""
    _race(conn, 1, date(2020, 1, 1), 1)
    _runner(conn, 1, 10, finish_pos=1)
    _race(conn, 2, date(2020, 6, 1), 1)
    _runner(conn, 2, 10, finish_pos=8)   # 同日の別レース
    _race(conn, 3, date(2020, 6, 1), 2)
    _runner(conn, 3, 10, finish_pos=1)   # 対象レース（同日）

    out = compute_f809(conn, pl.DataFrame({"race_id": [3], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    # 同日（race 2）は入らないので過去は race 1 のみ → 勝率1.0
    assert out.row(0, named=True)["f809_win_rate"] == 1.0


def test_scratched_past_race_is_excluded(conn):
    """`出走取消` など着順の無い過去走は母数に入れない。"""
    _race(conn, 1, date(2020, 1, 1), 1)
    _runner(conn, 1, 10, finish_pos=1)
    _race(conn, 2, date(2020, 2, 1), 1)
    _runner(conn, 2, 10, finish_pos=None, status="出走取消")
    _race(conn, 9, date(2020, 6, 1), 1)
    _runner(conn, 9, 10, finish_pos=1)

    out = compute_f809(conn, pl.DataFrame({"race_id": [9], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    # 有効な過去走は race 1 のみ → 勝率1.0（取消を母数に入れると0.5になる）
    assert out.row(0, named=True)["f809_win_rate"] == 1.0


def test_empty_base_returns_schema(conn):
    out = compute_f809(conn, pl.DataFrame({"race_id": [], "horse_id": []}),
                       as_of=date(2025, 1, 1))
    assert out.is_empty()
    for c in ("f809_win_rate", "f809_top3_rate"):
        assert c in out.columns
