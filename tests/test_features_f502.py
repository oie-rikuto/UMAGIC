"""`F-502` 当日時計傾向（`docs/domain-knowledge.md` `F-5xx` / `D-010` / `D-148`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f502 import MIN_BASE_N, compute_f502

NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)

# `races` は `(date, course, race_number)` に一意制約を持つ。
# `MIN_BASE_N` は**出走行数**を数えるため、レース数ではなく1レースあたりの頭数で満たす
_BULK = MIN_BASE_N


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


def _bulk_race(conn, race_id, race_date, race_number, time_sec, *,
                course="東京", surface="芝", n=_BULK, spread=0.5):
    """基準タイムの標本数を満たすだけの頭数を持つ1レースを作る。"""
    _race(conn, race_id, race_date, race_number, course=course, surface=surface, n=n)
    for i in range(n):
        t = time_sec + spread * (i - (n - 1) / 2)
        _runner(conn, race_id, race_id * 1000 + i, t)


def test_uses_only_earlier_same_day_same_course_same_surface(conn):
    d = date(2020, 2, 10)
    # 基準タイムの母数（前月）
    _bulk_race(conn, 10, date(2020, 1, 10), 1, 120.0)

    # 対象と同日・同競馬場・同馬場、対象より前(1R)。基準より2秒遅い
    _bulk_race(conn, 20, d, 1, 122.0)

    # 別競馬場（同日・同race_number以前）→ 含めない
    _bulk_race(conn, 21, d, 1, 90.0, course="阪神")

    # 同競馬場・別馬場（ダート、2R）→ 含めない
    _bulk_race(conn, 22, d, 2, 90.0, surface="ダート")

    # 対象より後（4R）→ リークするので含めない
    _bulk_race(conn, 23, d, 4, 90.0)

    # 対象レース（3R）
    _race(conn, 30, d, 3, n=1)
    _runner(conn, 30, 9001, None)

    base = pl.DataFrame({"race_id": [30], "horse_id": [9001]})
    out = compute_f502(conn, base, as_of=date(2025, 1, 1))
    row = out.row(0, named=True)

    # 基準より遅い(+2.0)レースだけが集計対象 → 残差の中央値は正
    assert row["f502"] is not None
    assert row["f502"] > 0
    assert row["f502_unavailable"] == 0


def test_first_race_of_day_is_unavailable(conn):
    d = date(2020, 2, 10)
    _bulk_race(conn, 10, date(2020, 1, 10), 1, 120.0)
    _race(conn, 20, d, 1, n=1)
    _runner(conn, 20, 9001, None)

    base = pl.DataFrame({"race_id": [20], "horse_id": [9001]})
    out = compute_f502(conn, base, as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f502"] is None
    assert row["f502_unavailable"] == 1


def test_race_level_is_shared_across_horses_in_the_race(conn):
    d = date(2020, 2, 10)
    _bulk_race(conn, 10, date(2020, 1, 10), 1, 120.0)
    _bulk_race(conn, 20, d, 1, 121.0)

    _race(conn, 30, d, 2, n=2)
    _runner(conn, 30, 9001, None)
    _runner(conn, 30, 9002, None)

    base = pl.DataFrame({"race_id": [30, 30], "horse_id": [9001, 9002]})
    out = compute_f502(conn, base, as_of=date(2025, 1, 1))
    vals = out.sort("horse_id")["f502"].to_list()
    assert vals[0] == vals[1]


def test_no_prior_month_history_is_structural(conn):
    """基準タイムが無いレースの出走馬は残差が作れず、実質的に先行レース無しと同じ扱いになる。"""
    d = date(2020, 1, 10)  # 前月の履歴が無い
    _bulk_race(conn, 10, d, 1, 120.0)
    _race(conn, 20, d, 2, n=1)
    _runner(conn, 20, 9001, None)

    base = pl.DataFrame({"race_id": [20], "horse_id": [9001]})
    out = compute_f502(conn, base, as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f502"] is None
    assert row["f502_unavailable"] == 1


def test_empty_base_returns_schema(conn):
    out = compute_f502(conn, pl.DataFrame({"race_id": [], "horse_id": []}),
                        as_of=date(2025, 1, 1))
    assert out.is_empty()
    assert "f502" in out.columns
    assert "f502_unavailable" in out.columns


def test_integration_with_build_features(conn):
    d = date(2020, 2, 10)
    _bulk_race(conn, 10, date(2020, 1, 10), 1, 120.0)
    _bulk_race(conn, 20, d, 1, 121.0)
    _race(conn, 30, d, 2, n=1)
    _runner(conn, 30, 9001, None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[30], feature_fns=[compute_f502])
    assert "f502" in df.columns and "f502_unavailable" in df.columns
