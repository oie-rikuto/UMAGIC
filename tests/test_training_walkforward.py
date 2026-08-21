"""walk-forward の実行（`docs/spec/014-training-pipeline.md` 6節 / `D-083` / `R-022`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from umagic.training import run_walk_forward

NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)
TODAY = date(2026, 8, 19)


def _race(conn, race_id, race_date, grade=None):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "grade, n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, 1, 1, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, grade, NOW],
    )


def _seed_2015_2024(conn):
    _race(conn, 1, date(2015, 1, 1))
    _race(conn, 2, date(2024, 12, 31))


def _fake_predict_fold(conn, fold):
    """fold.index を race_id の一部に埋め込み、fold ごとに一意な行を返す。"""
    base = fold.index * 1000
    return pl.DataFrame({
        "race_id": [base + 1, base + 2],
        "horse_id": [1, 2],
        "y_true": [1.0, 0.0],
        "y_pred": [0.7, 0.3],
    })


def test_fold_index_present_for_every_fold_and_no_cross_fold_duplicates(conn):
    """観点14: fold_index が全foldぶん含まれ、(race_id, horse_id) が
    fold をまたいで重複しない。
    """
    _seed_2015_2024(conn)
    out = run_walk_forward(conn, predict_fold=_fake_predict_fold, today=TODAY)

    assert sorted(out["fold_index"].unique().to_list()) == list(range(7))  # D-080: 7 fold
    assert out.height == 7 * 2
    keys = list(zip(out["race_id"].to_list(), out["horse_id"].to_list()))
    assert len(keys) == len(set(keys))


def test_columns_match_spec(conn):
    _seed_2015_2024(conn)
    out = run_walk_forward(conn, predict_fold=_fake_predict_fold, today=TODAY)
    assert set(out.columns) == {"race_id", "horse_id", "fold_index", "y_true", "y_pred"}


def test_no_folds_returns_empty_without_error(conn):
    """3年分のデータ（Q-033 の状況）→ fold が0本、空の DataFrame。例外にしない。"""
    _race(conn, 1, date(2022, 1, 5))
    _race(conn, 2, date(2024, 12, 28))
    out = run_walk_forward(conn, predict_fold=_fake_predict_fold, today=TODAY)
    assert out.is_empty()
    assert set(out.columns) == {"race_id", "horse_id", "fold_index", "y_true", "y_pred"}


def test_cross_fold_duplicate_keys_raise(conn):
    """(race_id, horse_id) が fold をまたいで重複したら ValueError。"""
    _seed_2015_2024(conn)

    def bad_predict_fold(conn, fold):
        # fold.index を無視して常に同じキーを返す → 2つ目以降の fold で衝突する
        return pl.DataFrame({
            "race_id": [1], "horse_id": [1], "y_true": [1.0], "y_pred": [0.5],
        })

    with pytest.raises(ValueError):
        run_walk_forward(conn, predict_fold=bad_predict_fold, today=TODAY)


def test_missing_column_in_predict_fold_raises(conn):
    _seed_2015_2024(conn)

    def incomplete_predict_fold(conn, fold):
        return pl.DataFrame({"race_id": [1], "horse_id": [1], "y_true": [1.0]})  # y_pred が無い

    with pytest.raises(ValueError):
        run_walk_forward(conn, predict_fold=incomplete_predict_fold, today=TODAY)


def test_predict_fold_receives_correct_fold_objects(conn):
    """predict_fold に渡される fold が make_folds() の出力と一致する。"""
    _seed_2015_2024(conn)
    seen_folds = []

    def recording_predict_fold(conn, fold):
        seen_folds.append(fold)
        return pl.DataFrame({
            "race_id": [fold.index * 1000 + 1], "horse_id": [1],
            "y_true": [1.0], "y_pred": [0.5],
        })

    run_walk_forward(conn, predict_fold=recording_predict_fold, today=TODAY)
    assert [f.valid_start.year for f in seen_folds] == list(range(2018, 2025))
