"""`F-304` 中央値ベースの速度指数（`docs/spec/003-features.md` / `D-123`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f304 import CLIP, MIN_BASE_N, MIN_TV_N, compute_f304, compute_spd

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)

# `races` は `(date, course, race_number)` に一意制約を持つ。
# `MIN_BASE_N` / `MIN_TV_N` は**出走行数**を数えるため、レース数ではなく
# 1レースあたりの頭数で満たす
_BULK = max(MIN_BASE_N, MIN_TV_N)


def _race(conn, race_id, race_date, race_number, *, course="東京", surface="芝",
          distance=2000, race_class="未勝利", grade=None, n=2):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "race_class, grade, n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, course, race_number, distance, surface,
         race_class, grade, n, n, NOW],
    )


def _runner(conn, race_id, horse_id, time_sec, *, status="出走"):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, ?, ?, 1, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, status, time_sec, NOW],
    )


def _bulk_race(conn, race_id, race_date, race_number, time_sec, *, n=_BULK, spread=0.5):
    """基準タイム／馬場差の標本数を満たすだけの頭数を持つ1レースを作る。

    `spread` でタイムにばらつきを与える。全頭同着だと基準の標準偏差が
    `0` になり、`f304` が `base_sd > 0` で弾いてしまうため。平均は
    `time_sec` に一致させる（`i - (n-1)/2` が対称になる）。
    """
    _race(conn, race_id, race_date, race_number, n=n)
    for i in range(n):
        t = time_sec + spread * (i - (n - 1) / 2)
        _runner(conn, race_id, race_id * 1000 + i, t)


def test_no_history_is_not_structural(conn):
    """過去走が無い馬は `unavailable=0`（`F-101` と同じ扱い）。"""
    _race(conn, 1, date(2020, 3, 1), 1, n=1)
    _runner(conn, 1, 10, 120.0)

    out = compute_f304(conn, pl.DataFrame({"race_id": [1], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["fspd_best"] is None
    assert row["fspd_n"] == 0.0
    assert row["fspd_unavailable"] == 0


def test_past_starts_without_speed_figure_is_structural(conn):
    """過去走はあるが速度指数が作れない（基準が無い）と `unavailable=1`。"""
    _race(conn, 1, date(2020, 1, 1), 1, n=1)
    _runner(conn, 1, 10, 120.0)  # 前月の履歴が無いので基準が作れない
    _race(conn, 2, date(2020, 2, 1), 1, n=1)
    _runner(conn, 2, 10, 121.0)

    out = compute_f304(conn, pl.DataFrame({"race_id": [2], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["fspd_best"] is None
    assert row["fspd_unavailable"] == 1


def test_faster_than_baseline_gives_positive_spd(conn):
    """基準より速く走れば `spd` は正になる（符号の向き）。"""
    _bulk_race(conn, 10, date(2020, 1, 10), 1, 120.0)   # 基準の母数（前月）
    _bulk_race(conn, 20, date(2020, 2, 10), 1, 120.0)   # 当日の馬場差の母数
    _race(conn, 21, date(2020, 2, 10), 2, n=1)
    _runner(conn, 21, 777, 118.0)                        # 基準より2秒速い

    spd = compute_spd(conn)
    v = spd.filter(pl.col("horse_id") == 777)["spd"]
    assert v.len() == 1
    assert v[0] > 0


def test_slower_than_baseline_gives_negative_spd(conn):
    _bulk_race(conn, 10, date(2020, 1, 10), 1, 120.0)
    _bulk_race(conn, 20, date(2020, 2, 10), 1, 120.0)
    _race(conn, 21, date(2020, 2, 10), 2, n=1)
    _runner(conn, 21, 777, 123.0)  # 基準より3秒遅い

    v = compute_spd(conn).filter(pl.col("horse_id") == 777)["spd"]
    assert v.len() == 1
    assert v[0] < 0


def test_aggregation_excludes_same_day_and_future(conn):
    """集約は `race_date < target_race_date` で厳密に切る（`D-054` 原則7）。"""
    _bulk_race(conn, 10, date(2020, 1, 10), 1, 120.0)
    _bulk_race(conn, 20, date(2020, 2, 10), 1, 120.0)
    _race(conn, 21, date(2020, 2, 10), 2, n=1)
    _runner(conn, 21, 777, 118.0)                        # 過去走（対象より前）

    _bulk_race(conn, 30, date(2020, 3, 1), 1, 120.0)     # 同日の馬場差の母数
    _race(conn, 31, date(2020, 3, 1), 2, n=1)
    _runner(conn, 31, 777, 110.0)                        # 同日の別レース
    _race(conn, 32, date(2020, 3, 1), 3, n=1)
    _runner(conn, 32, 777, 119.0)                        # 対象レース

    out = compute_f304(conn, pl.DataFrame({"race_id": [32], "horse_id": [777]}),
                       as_of=date(2025, 1, 1))
    # 同日（31）も自身（32）も入らない。集約対象は 2/10 の1走だけ
    assert out.row(0, named=True)["fspd_n"] == 1.0


def test_clip_bounds_aggregate(conn):
    """集約前に `±CLIP` でクリップする。"""
    _bulk_race(conn, 10, date(2020, 1, 10), 1, 120.0)
    _race(conn, 11, date(2020, 1, 10), 2, n=1)
    _runner(conn, 11, 999, 120.5)                        # 基準の分散を非ゼロにする
    _bulk_race(conn, 20, date(2020, 2, 10), 1, 120.0)
    _race(conn, 21, date(2020, 2, 10), 2, n=1)
    _runner(conn, 21, 777, 60.0)                         # 極端に速い

    _bulk_race(conn, 30, date(2020, 3, 1), 1, 120.0)
    _race(conn, 31, date(2020, 3, 1), 2, n=1)
    _runner(conn, 31, 777, 119.0)                        # 対象レース

    out = compute_f304(conn, pl.DataFrame({"race_id": [31], "horse_id": [777]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["fspd_best"] <= CLIP
    assert row["fspd_mean"] <= CLIP


def test_empty_base_returns_schema(conn):
    out = compute_f304(conn, pl.DataFrame({"race_id": [], "horse_id": []}),
                       as_of=date(2025, 1, 1))
    assert out.is_empty()
    assert "fspd_best" in out.columns
    assert "fspd_unavailable" in out.columns
