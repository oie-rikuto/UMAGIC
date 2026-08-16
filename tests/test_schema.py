"""`001-schema.md` の「テスト観点」8件。"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from tests.conftest import NOW
from umagic.schema import TABLE_NAMES, create_schema


def test_all_tables_created():
    c = duckdb.connect()
    create_schema(c)
    names = {r[0] for r in c.execute(
        "SELECT table_name FROM information_schema.tables"
    ).fetchall()}
    assert set(TABLE_NAMES) <= names


def _insert_race(conn, race_id=1, n_entries=16, n_starters=16, corner_nos=None, race_number=None):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, corner_nos, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, date(2023, 1, 1), race_number or race_id, n_entries, n_starters,
         corner_nos, NOW],
    )


def test_1_n_entries_below_n_starters_rejected(conn):
    with pytest.raises(duckdb.ConstraintException):
        _insert_race(conn, n_entries=10, n_starters=16)


def test_2_duplicate_date_course_race_number_rejected(conn):
    _insert_race(conn, race_id=1, race_number=11)
    with pytest.raises(duckdb.ConstraintException):
        _insert_race(conn, race_id=2, race_number=11)  # 同じ date/course/race_number


def test_3_tied_finish_pos_succeeds(conn):
    _insert_race(conn, race_id=1)
    for hid, num in [(1, 1), (2, 2)]:
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [hid, NOW],
        )
        conn.execute(
            "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, "
            "source, fetched_at) VALUES (1, ?, ?, '出走', 3, 'netkeiba_jra', ?)",
            [hid, num, NOW],
        )
    rows = conn.execute("SELECT finish_pos FROM runners").fetchall()
    assert [r[0] for r in rows] == [3, 3]


def test_4_undefined_status_rejected(conn):
    _insert_race(conn, race_id=1)
    conn.execute("INSERT INTO horses VALUES (1, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW])
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO runners (race_id, horse_id, number, status, source, fetched_at) "
            "VALUES (1, 1, 1, '取消', 'netkeiba_jra', ?)", [NOW],
        )


def test_5_odds_high_below_low_rejected(conn):
    _insert_race(conn, race_id=1)
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO odds VALUES (1, '単勝', '1', [1], 5.0, 3.0, ?, 'netkeiba_jra', ?)",
            [NOW, NOW],
        )


def test_6_empty_array_and_null_are_distinct(conn):
    _insert_race(conn, race_id=1, corner_nos=[])
    _insert_race(conn, race_id=2, corner_nos=None)
    rows = dict(conn.execute("SELECT race_id, corner_nos IS NULL FROM races ORDER BY race_id").fetchall())
    assert rows[1] is False
    assert rows[2] is True
    assert conn.execute("SELECT corner_nos FROM races WHERE race_id=1").fetchone()[0] == []


def test_7_corners_indexed_by_corner_nos(conn):
    _insert_race(conn, race_id=1, corner_nos=[3, 4])
    conn.execute("INSERT INTO horses VALUES (1, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW])
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, corners, "
        "source, fetched_at) VALUES (1, 1, 1, '出走', 1, [3, 2], 'netkeiba_jra', ?)", [NOW],
    )
    row = conn.execute("SELECT corners FROM runners WHERE race_id=1").fetchone()
    assert row[0] == [3, 2]  # corners[1] は corner_nos[1]=4コーナーの値


def test_8_race_corner_nos_empty_vs_null(conn):
    _insert_race(conn, race_id=1, corner_nos=[])
    _insert_race(conn, race_id=2, corner_nos=None)
    r1 = conn.execute("SELECT corner_nos FROM races WHERE race_id=1").fetchone()[0]
    r2 = conn.execute("SELECT corner_nos FROM races WHERE race_id=2").fetchone()[0]
    assert r1 == []
    assert r2 is None
