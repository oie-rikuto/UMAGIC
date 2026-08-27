"""`F-806` 相手強度 / `F-807` 条件替わり（`D-135` / `D-137`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f304 import MIN_BASE_N, MIN_TV_N
from umagic.features.f806 import compute_f806
from umagic.features.f807 import compute_f807

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)
_BULK = max(MIN_BASE_N, MIN_TV_N)


def _race(conn, race_id, race_date, race_number, *, course="東京", surface="芝",
          distance=2000, direction="左", n=2):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "direction, race_class, n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, '未勝利', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, course, race_number, distance, surface, direction, n, n, NOW],
    )


def _runner(conn, race_id, horse_id, *, time_sec=None, finish_pos=1, status="出走"):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, status, finish_pos, time_sec, NOW],
    )


def _bulk(conn, race_id, race_date, race_number, t, *, n=_BULK, spread=0.5):
    """`F-304` の速度指数が成立する母数を作る。"""
    _race(conn, race_id, race_date, race_number, n=n)
    for i in range(n):
        _runner(conn, race_id, race_id * 1000 + i,
                time_sec=t + spread * (i - (n - 1) / 2), finish_pos=i + 1)


# ---------------------------------------------------------------------------
# F-806
# ---------------------------------------------------------------------------

def test_f806_no_history_is_not_structural(conn):
    _race(conn, 1, date(2020, 3, 1), 1, n=1)
    _runner(conn, 1, 10, time_sec=120.0)
    out = compute_f806(conn, pl.DataFrame({"race_id": [1], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f806_field_mean"] is None
    assert row["f806_n"] == 0.0
    assert row["f806_unavailable"] == 0  # 過去走が無いのは構造的欠損ではない


def test_f806_aggregates_opponents_not_self(conn):
    """自分自身は相手に含めない。"""
    _bulk(conn, 10, date(2020, 1, 10), 1, 120.0)
    _bulk(conn, 20, date(2020, 2, 10), 1, 120.0)
    _race(conn, 21, date(2020, 2, 10), 2, n=2)
    _runner(conn, 21, 777, time_sec=118.0, finish_pos=1)   # 対象馬（速い）
    _runner(conn, 21, 888, time_sec=125.0, finish_pos=2)   # 相手（遅い）

    _bulk(conn, 30, date(2020, 3, 1), 1, 120.0)
    _race(conn, 31, date(2020, 3, 1), 2, n=1)
    _runner(conn, 31, 777, time_sec=119.0)

    out = compute_f806(conn, pl.DataFrame({"race_id": [31], "horse_id": [777]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f806_n"] == 1.0  # 相手は 888 の1頭だけ（自分は数えない）
    assert row["f806_field_mean"] < 0  # 遅い相手なので負


def test_f806_excludes_same_day_and_future(conn):
    """`race_date < target_race_date` で厳密に切る（`D-054` 原則7）。"""
    _bulk(conn, 10, date(2020, 1, 10), 1, 120.0)
    _bulk(conn, 20, date(2020, 2, 10), 1, 120.0)
    _race(conn, 21, date(2020, 2, 10), 2, n=2)
    _runner(conn, 21, 777, time_sec=118.0, finish_pos=1)
    _runner(conn, 21, 888, time_sec=125.0, finish_pos=2)

    _bulk(conn, 30, date(2020, 3, 1), 1, 120.0)
    _race(conn, 31, date(2020, 3, 1), 2, n=2)          # 同日の別レース
    _runner(conn, 31, 777, time_sec=110.0, finish_pos=1)
    _runner(conn, 31, 999, time_sec=111.0, finish_pos=2)
    _race(conn, 32, date(2020, 3, 1), 3, n=1)          # 対象レース（同日）
    _runner(conn, 32, 777, time_sec=119.0)

    out = compute_f806(conn, pl.DataFrame({"race_id": [32], "horse_id": [777]}),
                       as_of=date(2025, 1, 1))
    assert out.row(0, named=True)["f806_n"] == 1.0  # 2/10 の1頭だけ。同日は入らない


# ---------------------------------------------------------------------------
# F-807
# ---------------------------------------------------------------------------

def test_f807_picks_most_recent_past_race(conn):
    _race(conn, 1, date(2020, 1, 1), 1, distance=1200, surface="ダート", course="中山", n=1)
    _runner(conn, 1, 10)
    _race(conn, 2, date(2020, 2, 1), 1, distance=1600, surface="芝", course="東京", n=1)
    _runner(conn, 2, 10)
    _race(conn, 3, date(2020, 3, 1), 1, distance=2000, surface="芝", course="東京", n=1)
    _runner(conn, 3, 10)

    out = compute_f807(conn, pl.DataFrame({"race_id": [3], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f807_prev_distance"] == 1600.0       # 直近（2/1）を引く
    assert row["f807_prev_surface"] == "芝"
    assert row["f807_distance_diff"] == 400.0        # 2000 − 1600
    assert row["f807_surface_changed"] == 0.0
    assert row["f807_course_changed"] == 0.0


def test_f807_detects_surface_and_course_change(conn):
    _race(conn, 1, date(2020, 2, 1), 1, distance=1200, surface="ダート",
          course="中山", direction="右", n=1)
    _runner(conn, 1, 10)
    _race(conn, 2, date(2020, 3, 1), 1, distance=1200, surface="芝",
          course="東京", direction="左", n=1)
    _runner(conn, 2, 10)

    out = compute_f807(conn, pl.DataFrame({"race_id": [2], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f807_surface_changed"] == 1.0
    assert row["f807_course_changed"] == 1.0
    assert row["f807_direction_changed"] == 1.0
    assert row["f807_distance_diff"] == 0.0


def test_f807_no_history_is_all_null(conn):
    """前走が無い行は全列が欠損（指示子は置かない。`F-601`/`F-602` が持つ）。"""
    _race(conn, 1, date(2020, 3, 1), 1, n=1)
    _runner(conn, 1, 10)
    out = compute_f807(conn, pl.DataFrame({"race_id": [1], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f807_prev_distance"] is None
    assert row["f807_distance_diff"] is None


def test_f807_excludes_same_day(conn):
    """同日のレースは前走に取らない（`D-054` 原則7）。"""
    _race(conn, 1, date(2020, 2, 1), 1, distance=1400, n=1)
    _runner(conn, 1, 10)
    _race(conn, 2, date(2020, 3, 1), 1, distance=1800, n=1)   # 同日の別レース
    _runner(conn, 2, 10)
    _race(conn, 3, date(2020, 3, 1), 2, distance=2000, n=1)   # 対象
    _runner(conn, 3, 10)

    out = compute_f807(conn, pl.DataFrame({"race_id": [3], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    assert out.row(0, named=True)["f807_prev_distance"] == 1400.0


def test_f807_empty_base_returns_schema(conn):
    out = compute_f807(conn, pl.DataFrame({"race_id": [], "horse_id": []}),
                       as_of=date(2025, 1, 1))
    assert out.is_empty()
    assert "f807_distance_diff" in out.columns
    assert out.schema["f807_prev_surface"] == pl.Utf8
