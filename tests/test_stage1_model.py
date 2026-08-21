"""Stage 1 のモデル（`docs/spec/006-stage1-pace.md` 3〜6節 / `D-088` `D-090` `D-091`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from umagic.stage1 import LightGBMStage1Model, build_inputs, build_target, predict_f102

NOW = datetime(2026, 8, 21, tzinfo=timezone.utc)


def _race(conn, race_id, race_date=date(2020, 1, 1), n_starters=6, corner_nos=None,
          weather="晴", track_condition="良", distance=2000):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "direction, race_class, n_entries, n_starters, corner_nos, weather, "
        "track_condition, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, ?, '芝', '左', '3勝クラス', ?, ?, ?, ?, ?, "
        "'netkeiba_jra', ?)",
        [race_id, race_date, race_id, distance, n_starters, n_starters, corner_nos,
         weather, track_condition, NOW],
    )


def _runner(conn, race_id, horse_id, number, status="出走", finish_pos=1, corners=None):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "corners, source, fetched_at) VALUES (?, ?, ?, ?, ?, 100.0, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, number, status, finish_pos, corners, NOW],
    )


def _laps(conn, race_id, values):
    for i, v in enumerate(values, start=1):
        conn.execute(
            "INSERT INTO laps (race_id, furlong_no, lap_sec, source, fetched_at) "
            "VALUES (?, ?, ?, 'netkeiba_jra', ?)",
            [race_id, i, v, NOW],
        )


def _seed_training_race(conn, race_id, *, pace, race_date=date(2020, 1, 1)):
    """`pace` の符号に応じたラップ・入力を持つ1レースを仕込む（学習データ用）。"""
    _race(conn, race_id, race_date=race_date, n_starters=5, corner_nos=[1, 2, 3, 4])
    for i, h in enumerate((race_id * 10 + j for j in range(5))):
        _runner(conn, race_id, h, i + 1)
    if pace == "fast":
        _laps(conn, race_id, [10.0] * 7 + [14.0] * 3)
    else:
        _laps(conn, race_id, [14.0] * 7 + [10.0] * 3)


def _fit_dummy_model(conn, race_ids, *, seed=1) -> LightGBMStage1Model:
    x = build_inputs(conn, race_ids, as_of=date(2025, 1, 1))
    y = build_target(conn, race_ids)
    xy = x.join(y, on="race_id", how="inner").sort("race_id")
    model = LightGBMStage1Model()
    model.fit(xy.drop(["race_id", "f102_actual", "n_laps"]), xy["f102_actual"], sample_weight=None, seed=seed)
    return model


def _seed_ten_training_races(conn):
    race_ids = list(range(1, 11))
    for i, rid in enumerate(race_ids):
        _seed_training_race(conn, rid, pace="fast" if i % 2 == 0 else "slow",
                            race_date=date(2020, 1, 1 + i))
    return race_ids


def test_predict_works_without_laps(conn):
    """観点7: laps が無いレースへの predict_f102() は値が出る。例外にしない（D-091）。"""
    race_ids = _seed_ten_training_races(conn)
    model = _fit_dummy_model(conn, race_ids)

    _race(conn, 100, n_starters=4, corner_nos=[1, 2, 3, 4])  # laps は仕込まない
    for i, h in enumerate((1001, 1002, 1003, 1004)):
        _runner(conn, 100, h, i + 1)

    out = predict_f102(model, conn, [100], as_of=date(2025, 1, 1))
    assert out.height == 1
    assert out["f102"].to_list()[0] is not None


def test_predict_works_with_missing_track_condition(conn):
    """観点10: track_condition が欠損（未来のレースなど）でも値が出る。例外にしない（D-090）。"""
    race_ids = _seed_ten_training_races(conn)
    model = _fit_dummy_model(conn, race_ids)

    _race(conn, 101, n_starters=4, corner_nos=[1, 2, 3, 4], track_condition=None)
    for i, h in enumerate((1011, 1012, 1013, 1014)):
        _runner(conn, 101, h, i + 1)

    out = predict_f102(model, conn, [101], as_of=date(2025, 1, 1))
    assert out.height == 1
    assert out["f102"].to_list()[0] is not None


def test_same_seed_gives_bit_exact_predictions(conn):
    """観点11: 同じデータ・同じ seed で2回学習すると予測がビット完全一致（R-021）。"""
    race_ids = _seed_ten_training_races(conn)
    model1 = _fit_dummy_model(conn, race_ids, seed=42)
    model2 = _fit_dummy_model(conn, race_ids, seed=42)

    x = build_inputs(conn, race_ids, as_of=date(2025, 1, 1)).sort("race_id")
    p1 = model1.predict(x.drop("race_id")).to_list()
    p2 = model2.predict(x.drop("race_id")).to_list()
    assert p1 == p2


def test_as_of_excludes_races_after_cutoff(conn):
    """観点14: 対象レースより後の過去走が F-101 の集計に混入しない（R-019）。

    `compute_f101`（`003-features.md`）は `as_of` 引数自体を SQL 内で使わず、
    各行**自身のレース日付**（`hr.date < t.target_date`）を過去走の境界に
    する。`as_of` は `build_features()` が候補レースを絞り込む用途で、
    個々の `FeatureFn` の内部では対象行自身の日付がそのまま境界になる
    （`D-054` の追記事項と同じ構造）。ここでは対象レース自身の日付を
    変えることで、この境界が正しく機能することを確かめる。
    """
    horse = 500
    # 古い過去走（0.2）と、やや新しい過去走（0.9）
    _race(conn, 800, date(2019, 1, 1), n_starters=20, corner_nos=[1, 2, 3, 4])
    _runner(conn, 800, horse, 1, corners=[4, 4, 4, 4])  # 4/20=0.2
    _race(conn, 801, date(2019, 6, 1), n_starters=20, corner_nos=[1, 2, 3, 4])
    _runner(conn, 801, horse, 1, corners=[18, 18, 18, 18])  # 18/20=0.9

    # 対象レースA: 801より前の日付 → 800だけが history（f101=0.2）
    _race(conn, 900, date(2019, 3, 1), n_starters=1, corner_nos=[1, 2, 3, 4])
    _runner(conn, 900, horse, 1)

    # 対象レースB: 801より後の日付 → 800・801の両方が history
    _race(conn, 901, date(2019, 12, 1), n_starters=1, corner_nos=[1, 2, 3, 4])
    _runner(conn, 901, horse, 1)

    out = build_inputs(conn, [900, 901], as_of=date(2025, 1, 1)).sort("race_id")
    mean_a, mean_b = out["f101_mean"].to_list()
    assert mean_a is not None and mean_b is not None
    assert abs(mean_a - 0.2) < 1e-9   # 801（未来）が混入していない
    assert abs(mean_a - mean_b) > 0.01


def test_save_and_load_round_trip_preserves_extra_meta(conn, tmp_path):
    """観点15: save() → load() で meta.json の n_excluded_no_laps が保持される。"""
    race_ids = _seed_ten_training_races(conn)
    model = _fit_dummy_model(conn, race_ids)

    out_dir = tmp_path / "fold_0" / "stage1"
    model.save(out_dir, {"n_excluded_no_laps": 42})

    loaded, meta = LightGBMStage1Model.load(out_dir)
    assert meta["n_excluded_no_laps"] == 42

    x = build_inputs(conn, race_ids, as_of=date(2025, 1, 1)).sort("race_id")
    p_orig = model.predict(x.drop("race_id")).to_list()
    p_loaded = loaded.predict(x.drop("race_id")).to_list()
    for a, b in zip(p_orig, p_loaded):
        assert a == pytest.approx(b)
