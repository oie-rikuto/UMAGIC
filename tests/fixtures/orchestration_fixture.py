"""`orchestration.py` のスモークテスト用合成 fixture。

`tests/fixtures/leakage_fixture.py` と同じ挿入パターンを流用しつつ、
Stage 1/Stage 2/校正が最後まで動く程度の量（複数年・複数頭・血統・
騎手/調教師の重複履歴）を持たせる。実データの精度検証ではなく、
**結線が壊れていないことの確認**が目的。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import duckdb

FETCHED_AT = datetime(2026, 8, 22, tzinfo=timezone.utc)

N_HORSES = 16
N_JOCKEYS = 6
N_TRAINERS = 6
N_SIRES = 4
N_DAMSIRES = 4
N_STARTERS = 8
RACES_PER_YEAR = 12
YEARS = [2015, 2016, 2017, 2018]
G1_RACE_INDEXES = {0, 6}  # 各年、この index のレースを G1 にする（検証期間の2018年も含む）

LAPS = [12.0, 11.5, 12.3, 12.1]  # leakage_fixture.py と同じ4本（n>=4を満たす最小構成）


def _insert_horse(conn: duckdb.DuckDBPyConnection, horse_id: int) -> None:
    sire_id = 900 + (horse_id % N_SIRES)
    damsire_id = 910 + (horse_id % N_DAMSIRES)
    conn.execute(
        "INSERT INTO horses (horse_id, name, sire_id, dam_id, damsire_id, source, fetched_at) "
        "VALUES (?, ?, ?, NULL, ?, 'netkeiba_jra', ?)",
        [horse_id, f"馬{horse_id}", sire_id, damsire_id, FETCHED_AT],
    )


def _insert_jockey(conn: duckdb.DuckDBPyConnection, jockey_id: int) -> None:
    conn.execute(
        "INSERT INTO jockeys VALUES (?, ?, 'netkeiba_jra', ?)",
        [jockey_id, f"騎手{jockey_id}", FETCHED_AT],
    )


def _insert_trainer(conn: duckdb.DuckDBPyConnection, trainer_id: int) -> None:
    conn.execute(
        "INSERT INTO trainers VALUES (?, ?, 'netkeiba_jra', ?)",
        [trainer_id, f"調教師{trainer_id}", FETCHED_AT],
    )


def _insert_race(
    conn: duckdb.DuckDBPyConnection, race_id: int, race_date: date, *, grade: str | None,
) -> None:
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "direction, grade, track_condition, weather, n_entries, n_starters, corner_nos, "
        "meeting_no, meeting_day, source, fetched_at) "
        "VALUES (?, ?, '東京', 1, 2000, '芝', '右', ?, '良', '晴', ?, ?, [1,2,3,4], 1, 1, "
        "'netkeiba_jra', ?)",
        [race_id, race_date, grade, N_STARTERS, N_STARTERS, FETCHED_AT],
    )


def _insert_runner(
    conn: duckdb.DuckDBPyConnection, race_id: int, horse_id: int, number: int, *,
    finish_pos: int, jockey_id: int, trainer_id: int,
) -> None:
    margin = "0" if finish_pos == 1 else "1"
    time_sec = 120.0 + finish_pos / 10.0 + (race_id % 97) / 100.0
    last_3f = 35.0 + finish_pos / 10.0
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, frame, jockey_id, trainer_id, "
        "weight_carried, horse_weight, weight_diff, age, sex, odds_win, popularity, "
        "status, finish_pos, margin, time_sec, last_3f, corners, affiliation, "
        "source, fetched_at) VALUES "
        "(?, ?, ?, ?, ?, ?, 55.0, 480, 0, 4, '牡', ?, ?, '出走', ?, ?, ?, ?, "
        "[?,?,?,?], '東', 'netkeiba_jra', ?)",
        [
            race_id, horse_id, number, ((number - 1) % 8) + 1, jockey_id, trainer_id,
            float(finish_pos) * 2.0, finish_pos, finish_pos, margin, time_sec, last_3f,
            finish_pos, finish_pos, finish_pos, finish_pos,
            FETCHED_AT,
        ],
    )


def _insert_laps(conn: duckdb.DuckDBPyConnection, race_id: int) -> None:
    for i, lap in enumerate(LAPS, start=1):
        conn.execute(
            "INSERT INTO laps VALUES (?, ?, ?, 'netkeiba_jra', ?)",
            [race_id, i, lap, FETCHED_AT],
        )


def build_orchestration_fixture_conn() -> duckdb.DuckDBPyConnection:
    from umagic.ops_schema import create_ops_schema
    from umagic.schema import create_schema

    conn = duckdb.connect()
    create_schema(conn)
    create_ops_schema(conn)

    for h in range(1, N_HORSES + 1):
        _insert_horse(conn, h)
    for j in range(1, N_JOCKEYS + 1):
        _insert_jockey(conn, 500 + j)
    for t in range(1, N_TRAINERS + 1):
        _insert_trainer(conn, 600 + t)

    race_id = 1
    horse_cursor = 0
    for year in YEARS:
        for i in range(RACES_PER_YEAR):
            race_date = date(year, 1, 5) + timedelta(weeks=i * 4)
            grade = "G1" if i in G1_RACE_INDEXES else None
            _insert_race(conn, race_id, race_date, grade=grade)
            _insert_laps(conn, race_id)

            for slot in range(N_STARTERS):
                horse_id = ((horse_cursor + slot) % N_HORSES) + 1
                jockey_id = 500 + ((horse_cursor + slot) % N_JOCKEYS) + 1
                trainer_id = 600 + ((horse_cursor + slot) % N_TRAINERS) + 1
                _insert_runner(
                    conn, race_id, horse_id, slot + 1,
                    finish_pos=slot + 1, jockey_id=jockey_id, trainer_id=trainer_id,
                )
            horse_cursor += 3  # レースごとに出走馬の組み合わせをずらす
            race_id += 1

    return conn
