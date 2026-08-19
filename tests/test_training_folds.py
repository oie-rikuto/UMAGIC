"""fold の生成（`docs/spec/014-training-pipeline.md` 1節 / `D-079` `D-080` `D-085`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from umagic.training import make_folds

NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)
TODAY = date(2026, 8, 19)


def _race(conn, race_id, race_date, grade=None):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "grade, n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, 1, 1, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, grade, NOW],
    )


def _seed_2015_2024(conn):
    """観点1〜3で使う土台: 2015-01-01(最古)〜2024-12-31(最新、非G1)。"""
    _race(conn, 1, date(2015, 1, 1))
    _race(conn, 2, date(2024, 12, 31))  # 封印期間内(2023-08-19〜)だが非G1なので含まれる


def test_seven_folds_with_default_params(conn):
    """観点1: 既定値で fold数7（検証2018〜2024）。各 fold で train_end < valid_start。"""
    _seed_2015_2024(conn)
    folds = make_folds(conn, today=TODAY)
    assert [f.valid_start.year for f in folds] == list(range(2018, 2025))
    for f in folds:
        assert f.train_end < f.valid_start
        assert f.train_end == f.valid_start - __import__("datetime").timedelta(days=1)


def test_sliding_window(conn):
    """観点2: train_years=3 で各foldのtrain_startがvalid_startの3年前。"""
    _seed_2015_2024(conn)
    folds = make_folds(conn, today=TODAY, train_years=3)
    for f in folds:
        expected_start = date(f.valid_start.year - 3, 1, 1)
        assert f.train_start == expected_start
    # expanding にならない: train_start が fold ごとに異なる
    assert len({f.train_start for f in folds}) > 1


def test_expanding_window(conn):
    """観点3: train_years=None で全foldのtrain_startが同じ（対象データの最初の日）。"""
    _seed_2015_2024(conn)
    folds = make_folds(conn, today=TODAY, train_years=None)
    assert all(f.train_start == date(2015, 1, 1) for f in folds)


def test_sealed_g1_excluded_from_population(conn):
    """観点4: 封印期間内のG1は対象母集団から除外され、fold境界に影響しない。"""
    _race(conn, 1, date(2015, 1, 1))
    _race(conn, 2, date(2020, 6, 1))          # 非封印の最新（これが max になるはず）
    _race(conn, 3, date(2026, 8, 1), "G1")    # 封印期間内のG1（除外されるべき）

    folds = make_folds(conn, today=TODAY)
    # G1(2026)が母集団に入っていれば最後の検証年は2026になってしまう
    assert folds[-1].valid_start.year == 2020


def test_sealed_non_g1_included_in_population(conn):
    """観点5: 封印期間内でも非G1は対象母集団に含まれる（D-079）。"""
    _race(conn, 1, date(2015, 1, 1))
    _race(conn, 2, date(2026, 7, 1))  # 封印期間内・非G1 → 含まれるはず

    folds = make_folds(conn, today=TODAY)
    assert folds[-1].valid_start.year == 2026


def test_same_seed_reproduces_fold_seeds(conn):
    """観点6: 同じ seed で2回呼ぶと全foldのseedが一致する（R-021）。"""
    _seed_2015_2024(conn)
    folds1 = make_folds(conn, today=TODAY, seed=42)
    folds2 = make_folds(conn, today=TODAY, seed=42)
    assert [f.seed for f in folds1] == [f.seed for f in folds2]


def test_different_seed_changes_all_fold_seeds(conn):
    """観点7: seedを変えると全foldのseedが変わる。seed+iのような連番にならない（D-085）。"""
    _seed_2015_2024(conn)
    folds1 = make_folds(conn, today=TODAY, seed=1)
    folds2 = make_folds(conn, today=TODAY, seed=2)
    seeds1 = [f.seed for f in folds1]
    seeds2 = [f.seed for f in folds2]
    assert seeds1 != seeds2
    assert set(seeds1).isdisjoint(seeds2)


def test_min_train_years_below_three_raises(conn):
    """観点8: min_train_years=2 は ValueError（D-084）。"""
    _seed_2015_2024(conn)
    with pytest.raises(ValueError):
        make_folds(conn, today=TODAY, min_train_years=2)


def test_no_races_returns_empty(conn):
    assert make_folds(conn, today=TODAY) == []


def test_three_year_data_gives_no_folds(conn):
    """Q-033 の状況（3年分）を再現: fold が0本になる。"""
    _race(conn, 1, date(2022, 1, 5))
    _race(conn, 2, date(2024, 12, 28))
    assert make_folds(conn, today=TODAY) == []


def test_inner_valid_start_is_last_year_of_train(conn):
    _seed_2015_2024(conn)
    folds = make_folds(conn, today=TODAY)
    f = folds[0]  # valid=2018, train=2015-01-01..2017-12-31
    assert f.inner_valid_start == date(2017, 1, 1)
