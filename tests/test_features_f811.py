"""`F-811` 配合ニック（候補段階、`D-167`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f811 import DEFAULT_K, compute_f811

NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)

SIRE1, DAMSIRE1 = 901, 801
SIRE2, DAMSIRE2 = 902, 802


def _race(conn, race_id, race_date, n_starters=4):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, n_starters, n_starters, NOW],
    )


def _horse(conn, horse_id, sire_id, damsire_id):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, ?, NULL, ?, 'netkeiba_jra', ?)",
            [horse_id, sire_id, damsire_id, NOW],
        )


def _runner(conn, race_id, horse_id, finish_pos):
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, finish_pos, NOW],
    )


def test_shrink_toward_mu_global(conn):
    # SIRE1×DAMSIRE1 の産駒2頭: いずれも1着（成績が良い＝relfinishが低い）
    _horse(conn, 10, SIRE1, DAMSIRE1)
    _race(conn, 1, date(2020, 1, 1))
    _runner(conn, 1, 10, 1)  # perf=1/4=0.25

    _horse(conn, 11, SIRE1, DAMSIRE1)
    _race(conn, 2, date(2020, 2, 1))
    _runner(conn, 2, 11, 1)

    # 無関係な血統の母集団（μ_global を非自明にする）
    _horse(conn, 20, SIRE2, DAMSIRE2)
    _race(conn, 3, date(2020, 1, 15))
    _runner(conn, 3, 20, 3)  # perf=3/4=0.75

    # 対象: SIRE1×DAMSIRE1 の3頭目
    _horse(conn, 12, SIRE1, DAMSIRE1)
    _race(conn, 4, date(2020, 3, 1))
    _runner(conn, 4, 12, 2)

    base = pl.DataFrame({"race_id": [4], "horse_id": [12]})
    out = compute_f811(conn, base, as_of=date(2025, 1, 1))
    row = out.row(0, named=True)

    # 手計算: mu_global（前日まで全体）= (0.25+0.25+0.75)/3 = 0.41667
    # nick(SIRE1×DAMSIRE1) 前日までの2件平均 = 0.25、n=2
    mu_global = (0.25 + 0.25 + 0.75) / 3
    expected = (2 * 0.25 + DEFAULT_K * mu_global) / (2 + DEFAULT_K)
    assert abs(row["f811_nick"] - expected) < 1e-6
    assert row["f811_nick_n"] == 2.0
    assert row["f811_nick_unavailable"] == 0


def test_no_prior_nick_falls_back_to_mu_global(conn):
    _horse(conn, 20, SIRE2, DAMSIRE2)
    _race(conn, 1, date(2020, 1, 1))
    _runner(conn, 1, 20, 2)  # perf=0.5

    # 対象: 初めて現れる SIRE1×DAMSIRE1（前例なし）
    _horse(conn, 30, SIRE1, DAMSIRE1)
    _race(conn, 2, date(2020, 2, 1))
    _runner(conn, 2, 30, 1)

    base = pl.DataFrame({"race_id": [2], "horse_id": [30]})
    out = compute_f811(conn, base, as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f811_nick_n"] == 0.0
    assert abs(row["f811_nick"] - 0.5) < 1e-6  # 前日までの全体平均そのもの


def test_missing_sire_or_damsire_is_structural(conn):
    _horse(conn, 40, None, None)
    _race(conn, 1, date(2020, 1, 1))
    _runner(conn, 1, 40, 1)

    base = pl.DataFrame({"race_id": [1], "horse_id": [40]})
    out = compute_f811(conn, base, as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f811_nick"] is None
    assert row["f811_nick_unavailable"] == 1


def test_aggregation_excludes_same_day_and_future(conn):
    """集約は `race_date < target_race_date` で厳密に切る（`D-054` 原則7）。"""
    _horse(conn, 50, SIRE1, DAMSIRE1)
    _race(conn, 1, date(2020, 3, 1))
    _runner(conn, 1, 50, 1)  # 同日の別レース → 入らない

    _horse(conn, 51, SIRE1, DAMSIRE1)
    _race(conn, 2, date(2020, 3, 2))
    _runner(conn, 2, 51, 1)  # 対象より後 → 入らない

    _horse(conn, 52, SIRE1, DAMSIRE1)
    _race(conn, 3, date(2020, 3, 1))
    _runner(conn, 3, 52, 4)  # 対象レース自身

    base = pl.DataFrame({"race_id": [3], "horse_id": [52]})
    out = compute_f811(conn, base, as_of=date(2025, 1, 1))
    assert out.row(0, named=True)["f811_nick_n"] == 0.0


def test_integration_with_build_features(conn):
    _horse(conn, 60, SIRE1, DAMSIRE1)
    _race(conn, 1, date(2020, 1, 1))
    _runner(conn, 1, 60, 1)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[1], feature_fns=[compute_f811])
    assert {"f811_nick", "f811_nick_n", "f811_nick_unavailable"} <= set(df.columns)
