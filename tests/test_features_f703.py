"""`F-703` 厩舎の勝負度（`docs/spec/003-features.md` / `D-069`）。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl

from umagic.features.f703 import compute_f703

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, course):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, ?, ?, 2000, '芝', 1, 1, 'netkeiba_jra', ?)",
        [race_id, race_date, course, race_id, NOW],
    )


def _runner(conn, race_id, horse_id, affiliation, trainer_id, jockey_id):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "affiliation, trainer_id, jockey_id, source, fetched_at) "
        "VALUES (?, ?, ?, '出走', 1, 100.0, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, affiliation, trainer_id, jockey_id, NOW],
    )


def test_travel_flag_home_and_away(conn):
    _race(conn, 1, date(2020, 1, 1), "東京")  # 美浦(東)の地元
    _runner(conn, 1, 10, "東", 500, 100)

    _race(conn, 2, date(2020, 1, 1), "阪神")  # 美浦(東)の遠征
    _runner(conn, 2, 11, "東", 500, 100)

    base = pl.DataFrame({"race_id": [1, 2], "horse_id": [10, 11]})
    out = compute_f703(conn, base, as_of=date(2025, 1, 1)).sort("race_id")
    assert out["f703_travel_flag"].to_list() == [0, 1]
    assert out["f703_travel_flag_unavailable"].to_list() == [0, 0]


def test_local_affiliation_travel_flag_unavailable(conn):
    _race(conn, 3, date(2020, 1, 1), "東京")
    _runner(conn, 3, 12, "地", 600, 100)

    base = pl.DataFrame({"race_id": [3], "horse_id": [12]})
    out = compute_f703(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 12)
    assert row["f703_travel_flag"].to_list()[0] is None
    assert row["f703_travel_flag_unavailable"].to_list()[0] == 1


def test_main_jockey_flag(conn):
    trainer = 700
    # この厩舎の過去走: 騎手100が3回、騎手200が1回 → 主戦騎手は100
    for i, (rid, jid) in enumerate([(10, 100), (11, 100), (12, 100), (13, 200)]):
        _race(conn, rid, date(2019, 1, 1 + i), "東京")
        _runner(conn, rid, 900 + i, "東", trainer, jid)

    _race(conn, 20, date(2020, 1, 1), "東京")
    _runner(conn, 20, 950, "東", trainer, 100)  # 主戦騎手が乗る

    _race(conn, 21, date(2020, 1, 1), "中山")
    _runner(conn, 21, 951, "東", trainer, 200)  # 主戦騎手ではない

    base = pl.DataFrame({"race_id": [20, 21], "horse_id": [950, 951]})
    out = compute_f703(conn, base, as_of=date(2025, 1, 1)).sort("race_id")
    assert out["f703_main_jockey_flag"].to_list() == [1, 0]


def test_integration_with_build_features(conn):
    _race(conn, 30, date(2020, 1, 1), "東京")
    _runner(conn, 30, 40, None, None, None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[30], feature_fns=[compute_f703])
    assert "f703_travel_flag" in df.columns and "f703_main_jockey_flag" in df.columns


def test_main_jockey_same_day_not_counted_and_tie_goes_to_smaller_id(conn):
    trainer = 710
    # 過去: 騎手20と騎手10が1回ずつ（同数）→ 主戦騎手は騎手IDの小さい10
    _race(conn, 1, date(2019, 1, 1), "東京")
    _runner(conn, 1, 960, "東", trainer, 20)
    _race(conn, 2, date(2019, 1, 2), "東京")
    _runner(conn, 2, 961, "東", trainer, 10)

    # 対象日当日に騎手20が2回乗っている。当日の騎乗を数えると騎手20が主戦騎手になってしまう
    _race(conn, 3, date(2020, 1, 1), "東京")
    _runner(conn, 3, 962, "東", trainer, 10)  # 対象
    _race(conn, 4, date(2020, 1, 1), "中山")
    _runner(conn, 4, 963, "東", trainer, 20)
    _race(conn, 5, date(2020, 1, 1), "阪神")
    _runner(conn, 5, 964, "東", trainer, 20)

    base = pl.DataFrame({"race_id": [3], "horse_id": [962]})
    out = compute_f703(conn, base, as_of=date(2025, 1, 1))
    assert out["f703_main_jockey_flag"].to_list() == [1]
    assert out["f703_main_jockey_flag_unavailable"].to_list() == [0]


def test_main_jockey_matches_brute_force_definition(conn):
    """仕様どおり（同じ調教師の、対象日より前の騎乗数が最多の騎手。同数なら騎手IDが小さい方）に
    素朴に数えた結果と、全出走馬で一致する。"""
    import random
    from collections import Counter

    rng = random.Random(0)
    rows = []  # (race_id, horse_id, date, trainer_id, jockey_id)
    horse = 1000
    for race_id in range(1, 81):
        d = date(2019, 1, 1) + timedelta(days=(race_id - 1) // 3 * 2)  # 同じ日に複数レース
        _race(conn, race_id, d, "東京")
        for _ in range(3):
            horse += 1
            trainer = rng.choice([800, 801, 802, None])
            jockey = rng.choice([10, 11, 12, 13, None])
            _runner(conn, race_id, horse, "東", trainer, jockey)
            rows.append((race_id, horse, d, trainer, jockey))

    base = pl.DataFrame({"race_id": [r[0] for r in rows], "horse_id": [r[1] for r in rows]})
    out = compute_f703(conn, base, as_of=date(2025, 1, 1))
    got = {(r["race_id"], r["horse_id"]): (r["f703_main_jockey_flag"], r["f703_main_jockey_flag_unavailable"])
           for r in out.iter_rows(named=True)}

    for race_id, horse_id, d, trainer, jockey in rows:
        counts = Counter(
            j for (_, _, d2, t2, j) in rows if t2 == trainer and t2 is not None and j is not None and d2 < d
        )
        if trainer is None or not counts:
            assert got[(race_id, horse_id)] == (None, 1), (race_id, horse_id)
            continue
        main = min(counts, key=lambda j: (-counts[j], j))
        expected = None if jockey is None else int(main == jockey)
        assert got[(race_id, horse_id)] == (expected, 0), (race_id, horse_id)


def test_main_jockey_leaves_no_temp_tables(conn):
    _race(conn, 40, date(2020, 1, 1), "東京")
    _runner(conn, 40, 970, "東", 720, 10)
    base = pl.DataFrame({"race_id": [40], "horse_id": [970]})
    compute_f703(conn, base, as_of=date(2025, 1, 1))
    left = conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE temporary AND table_name LIKE '_f703_%'"
    ).fetchall()
    assert left == []
