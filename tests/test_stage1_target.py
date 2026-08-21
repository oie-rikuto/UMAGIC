"""Stage 1 の目的変数（`docs/spec/006-stage1-pace.md` 1節 / `D-087` `D-091`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

from umagic.stage1 import build_target

NOW = datetime(2026, 8, 21, tzinfo=timezone.utc)


def _race(conn, race_id, distance, n_starters=8):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, ?, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, date(2020, 1, 1), race_id, distance, n_starters, n_starters, NOW],
    )


def _laps(conn, race_id, values):
    for i, v in enumerate(values, start=1):
        conn.execute(
            "INSERT INTO laps (race_id, furlong_no, lap_sec, source, fetched_at) "
            "VALUES (?, ?, ?, 'netkeiba_jra', ?)",
            [race_id, i, v, NOW],
        )


def test_uniform_pace_gives_zero(conn):
    """観点1: 10本 [12.0]*10（2000m）→ f102_actual=0.0。"""
    _race(conn, 1, distance=2000)
    _laps(conn, 1, [12.0] * 10)
    out = build_target(conn, [1])
    row = out.filter(out["race_id"] == 1)
    assert row["f102_actual"].to_list()[0] == 0.0
    assert row["n_laps"].to_list()[0] == 10


def test_fast_front_slow_finish_is_positive(conn):
    """観点2: 前半が速く上がりが遅い → f102_actual > 0（ハイペース）。"""
    _race(conn, 1, distance=2000)
    _laps(conn, 1, [10.0] * 7 + [14.0] * 3)
    out = build_target(conn, [1])
    assert out["f102_actual"].to_list()[0] > 0


def test_slow_front_fast_finish_is_negative(conn):
    """観点3: 前半が遅く上がりが速い → f102_actual < 0（スローペース）。"""
    _race(conn, 1, distance=2000)
    _laps(conn, 1, [14.0] * 7 + [10.0] * 3)
    out = build_target(conn, [1])
    assert out["f102_actual"].to_list()[0] < 0


def test_odd_distance_matches_even_distance_at_same_pace(conn):
    """観点4: 2400m(12本)と2500m(13本、1本目100m)が同じペースなら同じ値に揃う（D-087の追記）。"""
    _race(conn, 1, distance=2400)
    _laps(conn, 1, [12.0] * 12)  # 200mごと一律12.0秒/200m

    _race(conn, 2, distance=2500)
    _laps(conn, 2, [6.0] + [12.0] * 12)  # 1本目は100m分=6.0秒、以降は200mごと12.0秒

    out = build_target(conn, [1, 2]).sort("race_id")
    f102 = out["f102_actual"].to_list()
    assert f102[0] == 0.0
    assert abs(f102[1] - 0.0) < 1e-9
    assert abs(f102[0] - f102[1]) < 1e-9


def test_race_without_laps_is_excluded(conn):
    """観点5: laps が1行も無いレースは build_target() の戻り値に行が現れない。"""
    _race(conn, 1, distance=2000)
    out = build_target(conn, [1])
    assert out.is_empty()


def test_race_with_three_or_fewer_laps_is_excluded(conn):
    """観点6: N<4（3本以下）のレースは前半が作れず除外される。"""
    _race(conn, 1, distance=600)
    _laps(conn, 1, [12.0, 12.0, 12.0])
    out = build_target(conn, [1])
    assert out.is_empty()


def test_empty_race_ids():
    import duckdb
    from umagic.schema import create_schema

    c = duckdb.connect()
    create_schema(c)
    out = build_target(c, [])
    assert out.is_empty()
    c.close()
