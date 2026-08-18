"""対象母集団の抽出（`docs/spec/005-baseline.md` 1節 / `D-071` / `D-076`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

from umagic.baseline import target_races

NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)
TODAY = date(2026, 8, 19)


def _race(conn, race_id, race_date, grade, n_starters=8):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "grade, n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, grade, n_starters, n_starters, NOW],
    )


def test_all_population_excludes_only_sealed_g1(conn):
    _race(conn, 1, date(2015, 1, 1), "G1")  # 古いG1（非封印）
    _race(conn, 2, date(2025, 1, 1), "G1")  # 直近3年以内のG1（封印）
    _race(conn, 3, date(2025, 1, 1), None)  # 直近だが非G1（封印対象外）

    result = target_races(conn, population="all", today=TODAY)
    assert set(result.races["race_id"].to_list()) == {1, 3}
    assert result.n_sealed_g1_excluded == 1


def test_g1_population_only_returns_non_sealed_g1(conn):
    _race(conn, 1, date(2015, 1, 1), "G1")
    _race(conn, 2, date(2025, 1, 1), "G1")
    _race(conn, 3, date(2025, 1, 1), None)

    result = target_races(conn, population="g1", today=TODAY)
    assert result.races["race_id"].to_list() == [1]
    assert result.n_sealed_g1_excluded == 1


def test_no_races_at_all(conn):
    result = target_races(conn, population="all", today=TODAY)
    assert result.races.is_empty()
    assert result.n_sealed_g1_excluded == 0


def test_sorted_by_race_id(conn):
    _race(conn, 30, date(2020, 1, 1), None)
    _race(conn, 10, date(2020, 1, 1), None)
    _race(conn, 20, date(2020, 1, 1), None)

    result = target_races(conn, population="all", today=TODAY)
    assert result.races["race_id"].to_list() == [10, 20, 30]
