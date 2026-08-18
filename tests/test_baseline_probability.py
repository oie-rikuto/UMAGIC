"""市場確率の確率指標（`docs/spec/005-baseline.md` 2節）。"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

from umagic.baseline import probability_metrics

NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)


def _race(conn, race_id, n_starters=3):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, date(2020, 1, 1), race_id, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, number, odds_win, finish_pos, popularity, status="出走"):
    horse_id = race_id * 100 + number
    conn.execute(
        "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
        [horse_id, NOW],
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "odds_win, popularity, source, fetched_at) VALUES (?, ?, ?, ?, ?, 100.0, ?, ?, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, number, status, finish_pos, odds_win, popularity, NOW],
    )


def test_normalization_sums_to_one_and_metrics(conn):
    """観点1: オッズ [2.0, 4.0, 4.0] → p = [0.5, 0.25, 0.25]。"""
    _race(conn, 1)
    _runner(conn, 1, 1, 2.0, finish_pos=1, popularity=1)
    _runner(conn, 1, 2, 4.0, finish_pos=2, popularity=2)
    _runner(conn, 1, 3, 4.0, finish_pos=3, popularity=3)

    m = probability_metrics(conn, [1], population="all")
    assert m.n_races == 1
    assert m.n_runners == 3
    assert abs(m.log_loss - (-math.log(0.5))) < 1e-9
    assert abs(m.brier - ((0.5 - 1) ** 2 + 0.25**2 + 0.25**2)) < 1e-9
    assert m.top1_hit_rate == 1.0
    assert m.top3_hit_rate == 1.0


def test_dead_heat_splits_label(conn):
    """観点2: 1着同着2頭 → y = [0.5, 0.5]、Σy = 1。"""
    _race(conn, 2, n_starters=2)
    _runner(conn, 2, 1, 3.0, finish_pos=1, popularity=1)
    _runner(conn, 2, 2, 6.0, finish_pos=1, popularity=2)

    m = probability_metrics(conn, [2], population="all")
    # p = [2/3, 1/3]（1/3 / (1/3+1/6) = 2/3, 1/6 / (1/2) = 1/3）
    p1, p2 = 2 / 3, 1 / 3
    y1 = y2 = 0.5
    expected_log_loss = -(y1 * math.log(p1) + y2 * math.log(p2))
    expected_brier = (p1 - y1) ** 2 + (p2 - y2) ** 2
    assert abs(m.log_loss - expected_log_loss) < 1e-9
    assert abs(m.brier - expected_brier) < 1e-9


def test_scratched_horse_excluded_from_normalization(conn):
    """観点3: `出走取消` 馬は分母に含まれない。"""
    _race(conn, 3)
    _runner(conn, 3, 1, 2.0, finish_pos=1, popularity=1)
    _runner(conn, 3, 2, 4.0, finish_pos=2, popularity=2)
    _runner(conn, 3, 3, None, finish_pos=None, popularity=None, status="出走取消")

    m = probability_metrics(conn, [3], population="all")
    assert m.n_runners == 2
    assert abs(m.log_loss - (-math.log(1 / 3 / (1 / 3 + 1 / 6)))) < 1e-9


def test_suspended_horse_included_and_never_wins(conn):
    """観点4: `競走中止` 馬は分母に含まれ、的中しない扱いになる。"""
    _race(conn, 4)
    _runner(conn, 4, 1, 2.0, finish_pos=1, popularity=1)
    _runner(conn, 4, 2, 4.0, finish_pos=None, popularity=2, status="競走中止")
    _runner(conn, 4, 3, 8.0, finish_pos=3, popularity=3)

    m = probability_metrics(conn, [4], population="all")
    assert m.n_runners == 3  # 中止馬も分母に含まれる


def test_empty_race_ids_returns_nan(conn):
    m = probability_metrics(conn, [], population="all")
    assert m.n_races == 0
    assert math.isnan(m.log_loss)
