"""`build_features` の骨格（`003-features.md` / `004-leakage-test.md`）。

一意性・決定的な順序・ビット完全一致・`as_of` による対象レースの絞り込みを見る。
個別の `F-xxx` はまだ登録しない（`feature_fns=[]`）。
"""

from __future__ import annotations

from datetime import date

import polars as pl

from tests.conftest import NOW
from umagic.features.build import build_features


def _race(conn, race_id, race_date, race_number=None):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', 2, 2, 'netkeiba_jra', ?)",
        [race_id, race_date, race_number or race_id, NOW],
    )


def _runner(conn, race_id, horse_id, number):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute("INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, "
                    "'netkeiba_jra', ?)", [horse_id, NOW])
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, 'netkeiba_jra', ?)",
        [race_id, horse_id, number, number, NOW],
    )


def _seed(conn):
    _race(conn, 1, date(2022, 1, 1), race_number=1)
    _runner(conn, 1, 1, 1)
    _runner(conn, 1, 2, 2)
    _race(conn, 2, date(2022, 6, 1), race_number=2)
    _runner(conn, 2, 3, 1)
    _race(conn, 3, date(2023, 1, 1), race_number=3)
    _runner(conn, 3, 4, 1)


def test_unique_race_horse_key(conn):
    _seed(conn)
    df = build_features(conn, as_of=date(2024, 1, 1))
    assert df.select(["race_id", "horse_id"]).is_duplicated().sum() == 0


def test_deterministic_sort_order(conn):
    _seed(conn)
    df = build_features(conn, as_of=date(2024, 1, 1))
    pairs = list(zip(df["race_id"].to_list(), df["horse_id"].to_list()))
    assert pairs == sorted(pairs)


def test_bit_exact_across_repeated_calls(conn):
    """同じ入力・同じ as_of なら2回呼んでビット完全一致する（D-055 / R-021）。"""
    _seed(conn)
    df1 = build_features(conn, as_of=date(2024, 1, 1))
    df2 = build_features(conn, as_of=date(2024, 1, 1))
    assert df1.equals(df2)


def test_as_of_bounds_target_population():
    """race_ids 未指定では date < as_of のレースのみが対象になる。"""
    import duckdb
    from umagic.ops_schema import create_ops_schema
    from umagic.schema import create_schema

    c = duckdb.connect()
    create_schema(c)
    create_ops_schema(c)
    _seed(c)

    df_early = build_features(c, as_of=date(2022, 3, 1))
    assert set(df_early["race_id"].to_list()) == {1}  # race 2, 3 はまだ

    df_later = build_features(c, as_of=date(2024, 1, 1))
    assert set(df_later["race_id"].to_list()) == {1, 2, 3}


def test_race_ids_overrides_as_of_population(conn):
    """race_ids を渡すと、その日付が as_of 以降でも対象に含められる（推論用途）。"""
    _seed(conn)
    df = build_features(conn, as_of=date(2022, 1, 1), race_ids=[3])
    assert set(df["race_id"].to_list()) == {3}


def test_empty_feature_fns_returns_key_columns_only(conn):
    _seed(conn)
    df = build_features(conn, as_of=date(2024, 1, 1), feature_fns=[])
    assert df.columns == ["race_id", "horse_id"]


def test_base_population_excludes_scratched_and_withdrawn(conn):
    """`D-109`: 出走取消・競走除外の馬は基底集合に含めない（`build_labels()` と同じ条件）。

    含めると `n_starters`（実出走頭数）より多い頭数が `F-901` の `_rank`
    計算に混入し、比率が1.0を超える（発見の経緯は `D-109`）。
    """
    _race(conn, 10, date(2022, 1, 1), race_number=10)
    conn.execute(
        "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?), "
        "(?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?), "
        "(?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
        [101, NOW, 102, NOW, 103, NOW],
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES "
        "(10, 101, 1, '出走', 1, 100.0, 'netkeiba_jra', ?), "
        "(10, 102, 2, '出走取消', NULL, NULL, 'netkeiba_jra', ?), "
        "(10, 103, 3, '競走除外', NULL, NULL, 'netkeiba_jra', ?)",
        [NOW, NOW, NOW],
    )
    df = build_features(conn, as_of=date(2024, 1, 1), race_ids=[10], feature_fns=[])
    assert set(df.filter(pl.col("race_id") == 10)["horse_id"].to_list()) == {101}
