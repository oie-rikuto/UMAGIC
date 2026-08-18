"""リーク検査用の合成 fixture（`004-leakage-test.md` / `D-053`）。

`004` が要求する5つの性質をすべて満たす。実データ固有の並びを再現しない
fixture では、対応する検査が素通りする（`004` 本文）。

| 性質 | この fixture での実現 |
|---|---|
| 同日・同競馬場に連続するR番号 | 2023-01-07 東京 R1〜R3 |
| 同日・複数競馬場の開催 | 同日の中山 R1 |
| 同一馬が複数レースに出走（時系列） | horse_id=100 が5レースに出走 |
| 少走馬と多走馬の混在 | horse_id=200,201 は1走のみ／horse_id=100 は5走 |
| 期間をまたぐレース（as_of の前後） | `SEALED_CUTOFF=2024-01-01` を挟んで配置 |

`Q-028`（規模）は未決のため、最小構成から始める。
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import duckdb

from umagic.ops_schema import create_ops_schema
from umagic.schema import create_schema

FETCHED_AT = datetime(2026, 8, 18, tzinfo=timezone.utc)

# build_features の第二引数として使う既定の as_of。2023-01-07 と 2024-06-01 の
# 間に置き、「過去」「未来」の両方に実データを持たせる
AS_OF_MID = date(2023, 6, 1)
AS_OF_LATE = date(2024, 6, 1)


def _insert_race(
    conn: duckdb.DuckDBPyConnection, race_id: int, race_date: date, course: str,
    race_number: int, corner_nos: list[int] | None = None, grade: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, corner_nos, grade, source, fetched_at) "
        "VALUES (?, ?, ?, ?, 2000, '芝', ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, course, race_number, 4, 4, corner_nos, grade, FETCHED_AT],
    )


def _insert_horse(conn: duckdb.DuckDBPyConnection, horse_id: int) -> None:
    if conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        return
    conn.execute(
        "INSERT INTO horses VALUES (?, ?, NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
        [horse_id, f"馬{horse_id}", FETCHED_AT],
    )


def _insert_runner(
    conn: duckdb.DuckDBPyConnection, race_id: int, horse_id: int, number: int,
    finish_pos: int, corners: list[int] | None = None, odds_win: float | None = None,
    popularity: int | None = None, time_sec: float | None = None,
) -> None:
    _insert_horse(conn, horse_id)
    if time_sec is None:
        # レースごとに異なる値にする。全行が同一値だと「対象レース自身の
        # 値が紛れ込んだか」を検出できない（値ベースのリーク検査が機能しない）
        time_sec = 100.0 + (race_id % 1000) / 10 + horse_id / 100
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "corners, odds_win, popularity, source, fetched_at) "
        "VALUES (?, ?, ?, '出走', ?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, number, finish_pos, time_sec, corners, odds_win,
         popularity, FETCHED_AT],
    )


def _insert_laps(conn: duckdb.DuckDBPyConnection, race_id: int) -> None:
    for i, lap in enumerate([12.0, 11.5, 12.3, 12.1], start=1):
        conn.execute(
            "INSERT INTO laps VALUES (?, ?, ?, 'netkeiba_jra', ?)",
            [race_id, i, lap, FETCHED_AT],
        )


def build_leakage_fixture_conn() -> duckdb.DuckDBPyConnection:
    """5性質を満たすDBを新規に作って返す。"""
    conn = duckdb.connect()
    create_schema(conn)
    create_ops_schema(conn)

    # --- 同日・同競馬場の連続R番号 + 同日・複数競馬場 ---
    day = date(2023, 1, 7)
    _insert_race(conn, 20230101001, day, "東京", 1, corner_nos=[1, 2, 3, 4])
    _insert_runner(conn, 20230101001, 10, 1, finish_pos=1, corners=[1, 1, 1, 1])
    _insert_runner(conn, 20230101001, 11, 2, finish_pos=2, corners=[2, 2, 2, 2])
    _insert_laps(conn, 20230101001)

    _insert_race(conn, 20230101002, day, "東京", 2, corner_nos=[1, 2, 3, 4])
    _insert_runner(conn, 20230101002, 12, 1, finish_pos=1, corners=[3, 2, 1, 1])
    _insert_runner(conn, 20230101002, 13, 2, finish_pos=2, corners=[1, 1, 2, 2])
    _insert_laps(conn, 20230101002)

    # 対象レース。同日・同競馬場・R番号が対象より小さい2レースのみが
    # F-501 の入力になってよい
    _insert_race(conn, 20230101003, day, "東京", 3, corner_nos=[1, 2, 3, 4], grade="G1")
    _insert_runner(conn, 20230101003, 100, 1, finish_pos=1, corners=[2, 2, 1, 1],
                   odds_win=3.5, popularity=1)
    _insert_runner(conn, 20230101003, 14, 2, finish_pos=2, corners=[1, 1, 2, 2],
                   odds_win=5.0, popularity=2)
    _insert_laps(conn, 20230101003)

    # 同日・別競馬場（中山）。course が一致しないため F-501 の入力に含めてはならない
    _insert_race(conn, 20230106001, day, "中山", 1, corner_nos=[1, 2, 3, 4])
    _insert_runner(conn, 20230106001, 20, 1, finish_pos=1, corners=[1, 1, 1, 1])

    # --- 少走馬（1走のみ） ---
    _insert_race(conn, 20230201001, date(2023, 2, 1), "東京", 1, corner_nos=[1, 2, 3, 4])
    _insert_runner(conn, 20230201001, 200, 1, finish_pos=3, corners=[5, 5, 5, 5])
    _insert_runner(conn, 20230201001, 201, 2, finish_pos=4, corners=[6, 6, 6, 6])

    # --- 多走馬（horse_id=100 が同日の対象レースを含め複数レースに出走） ---
    # 前半3件は対象レース（2023-01-07）より前、後半1件は後。
    # odds_win も付け、principle6（対象レースのオッズ不使用）を値で検出できるようにする
    for i, (d, pos, odds) in enumerate([
        (date(2022, 6, 1), 2, 4.0), (date(2022, 9, 1), 1, 2.0),
        (date(2022, 12, 1), 3, 8.0), (date(2023, 3, 1), 1, 1.5),
    ], start=1):
        rid = 20220000000 + i
        _insert_race(conn, rid, d, "東京", i, corner_nos=[1, 2, 3, 4])
        _insert_runner(conn, rid, 100, 1, finish_pos=pos, corners=[pos, pos, pos, pos],
                       odds_win=odds, popularity=pos)

    # --- 期間をまたぐ（AS_OF_MID の前後） ---
    _insert_race(conn, 20230601001, date(2023, 5, 1), "東京", 1, corner_nos=[1, 2, 3, 4])
    _insert_runner(conn, 20230601001, 100, 1, finish_pos=1, corners=[1, 1, 1, 1])

    _insert_race(conn, 20240601001, date(2024, 5, 1), "東京", 1, corner_nos=[1, 2, 3, 4])
    _insert_runner(conn, 20240601001, 100, 1, finish_pos=1, corners=[1, 1, 1, 1])

    return conn


# 対象レース（同日前後判定・封印セット検査の主眼）
TARGET_RACE_ID = 20230101003
TARGET_DATE = date(2023, 1, 7)
TARGET_COURSE = "東京"
TARGET_RACE_NUMBER = 3
