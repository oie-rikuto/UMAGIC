"""`orchestration.py`（P-3 の結線）のテスト。

Stage 1 / Stage 2 / 校正の個別テストは `test_stage1_*` / `test_stage2_*` /
`test_calibration_*` が担う。ここでは「結線が壊れていないか」（スモーク
テスト）と、結線層に固有のロジック（特徴量行列の組み立て、母集団ごとの
校正適用）を確認する。
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from tests.fixtures.orchestration_fixture import build_orchestration_fixture_conn
from umagic.orchestration import (
    CATEGORY_COLUMNS,
    RACE_LEVEL_COLUMNS,
    Stage2FoldRunner,
    apply_g1_calibration,
    assemble_stage2_matrix,
    track_variant_fit_all,
)
from umagic.training import make_folds, run_walk_forward


@pytest.fixture(scope="module")
def orch_conn():
    conn = build_orchestration_fixture_conn()
    yield conn
    conn.close()


def test_assemble_stage2_matrix_has_no_raw_leak_columns(orch_conn):
    """観点1: 生の着順・タイム等の禁止列（004の原則1）が混入しない。"""
    race_ids = orch_conn.execute("SELECT race_id FROM races ORDER BY race_id LIMIT 5").pl()[
        "race_id"
    ].to_list()
    f102 = pl.DataFrame(schema={"race_id": pl.Int64, "f102": pl.Float64})
    out = assemble_stage2_matrix(orch_conn, race_ids, as_of=date(2018, 1, 1), f102=f102)

    forbidden = {"finish_pos", "time_sec", "margin", "odds_win", "popularity", "status"}
    assert forbidden.isdisjoint(out.columns)
    assert {"race_id", "horse_id"} <= set(out.columns)


def test_assemble_stage2_matrix_relativizes_non_race_level_columns(orch_conn):
    """観点2: race_level=False の列には `_z`/`_rank` が付き、race_level=True には付かない。"""
    race_ids = orch_conn.execute("SELECT race_id FROM races ORDER BY race_id LIMIT 5").pl()[
        "race_id"
    ].to_list()
    f102 = pl.DataFrame(schema={"race_id": pl.Int64, "f102": pl.Float64})
    out = assemble_stage2_matrix(orch_conn, race_ids, as_of=date(2018, 1, 1), f102=f102)

    # F-601（race_level=False）には _z/_rank が付く
    assert "f601_finish_pos_prev_z" in out.columns
    assert "f601_finish_pos_prev_rank" in out.columns
    # F-501（race_level=True）には付かない
    assert "f501_z" not in out.columns
    assert "f501_rank" not in out.columns
    # F-102/F-104（race_level=True）にも付かない
    assert "f102_z" not in out.columns
    assert "f104_z" not in out.columns
    # カテゴリ列（sire_id 等）にも付かない
    assert "sire_id_z" not in out.columns
    # バグ回帰: f103_z/f103_rank（compute_f104 が内部で作る）が二重に
    # relativize されない（z-score の z-score のような無意味な列を作らない）
    assert "f103_z_z" not in out.columns
    assert "f103_z_rank" not in out.columns
    assert "f103_rank_z" not in out.columns
    assert "f103_rank_rank" not in out.columns


def test_assemble_stage2_matrix_f104_uses_joined_f102(orch_conn):
    """観点3: F-102 を join した後で F-104（F-102×F-103_z）が計算される。"""
    race_ids = orch_conn.execute("SELECT race_id FROM races ORDER BY race_id LIMIT 5").pl()[
        "race_id"
    ].to_list()
    f102 = pl.DataFrame({"race_id": race_ids, "f102": [1.0] * len(race_ids)})
    out = assemble_stage2_matrix(orch_conn, race_ids, as_of=date(2018, 1, 1), f102=f102)
    assert "f104" in out.columns
    # f102 が全行 1.0 のとき、f104 = 1.0 * f103_z（f103 が非nullな行で確認できる）
    non_null = out.filter(pl.col("f103_z").is_not_null())
    if not non_null.is_empty():
        row = non_null.row(0, named=True)
        assert row["f104"] == pytest.approx(row["f103_z"])


def _make_calibration_fixture_conn():
    import duckdb

    from umagic.ops_schema import create_ops_schema
    from umagic.schema import create_schema

    conn = duckdb.connect()
    create_schema(conn)
    create_ops_schema(conn)
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "grade, n_entries, n_starters, source, fetched_at) VALUES "
        "(1, '2020-01-01', '東京', 1, 2000, '芝', 'G1', 2, 2, 'netkeiba_jra', now()), "
        "(2, '2020-01-01', '中山', 1, 2000, '芝', NULL, 2, 2, 'netkeiba_jra', now())"
    )
    return conn


def test_apply_g1_calibration_leaves_non_g1_untouched():
    """観点4: `apply_g1_calibration` は G1以外の行を返さない（呼び出し側が
    `all` 母集団には元の`walk_forward_out`をそのまま使う設計を裏付ける）。
    """
    from umagic.calibration import Calibrator

    class _FakeRunner:
        fold_calibrators = {
            0: Calibrator(
                temperature=2.0, n_races_fit=10, n_runners_fit=80,
                logloss_before=1.0, logloss_after=0.9, at_bound=False,
            )
        }
        fold_valid_scores = {
            0: pl.DataFrame({"race_id": [1, 1], "horse_id": [10, 11], "score": [2.0, 0.0]}),
        }

    conn = _make_calibration_fixture_conn()
    walk_forward_out = pl.DataFrame({
        "race_id": [1, 1, 2, 2], "horse_id": [10, 11, 20, 21], "fold_index": [0, 0, 0, 0],
        "y_true": [1.0, 0.0, 1.0, 0.0], "y_pred": [0.6, 0.4, 0.6, 0.4],
    })
    out = apply_g1_calibration(conn, walk_forward_out, _FakeRunner())
    conn.close()

    assert set(out["race_id"].unique().to_list()) == {1}
    assert set(out.columns) == set(walk_forward_out.columns)


def test_apply_g1_calibration_uses_raw_score_not_y_pred():
    """観点4b（バグ回帰）: `apply_g1_calibration` は `predict_fold` が保持した
    校正前の生スコア（`fold_valid_scores`）に温度を適用する。`y_pred`
    （レース内 softmax 済みの `win_prob`）をそのまま `Calibrator.apply()`
    に渡すと二重に softmax を取ってしまい、下のアサーションで検証する
    正しい値（`softmax(score/T)`）と一致しなくなる。
    """
    import math

    from umagic.calibration import Calibrator

    T = 2.0
    raw_scores = {10: 2.0, 11: 0.0}
    # y_pred（win_prob）は T=1 の softmax なので raw_scores とは異なる値にしておく
    y_pred = {10: 0.7, 11: 0.3}

    class _FakeRunner:
        fold_calibrators = {
            0: Calibrator(
                temperature=T, n_races_fit=10, n_runners_fit=80,
                logloss_before=1.0, logloss_after=0.9, at_bound=False,
            )
        }
        fold_valid_scores = {
            0: pl.DataFrame({
                "race_id": [1, 1], "horse_id": [10, 11],
                "score": [raw_scores[10], raw_scores[11]],
            }),
        }

    conn = _make_calibration_fixture_conn()
    walk_forward_out = pl.DataFrame({
        "race_id": [1, 1], "horse_id": [10, 11], "fold_index": [0, 0],
        "y_true": [1.0, 0.0], "y_pred": [y_pred[10], y_pred[11]],
    })
    out = apply_g1_calibration(conn, walk_forward_out, _FakeRunner())
    conn.close()

    expected_10 = math.exp(raw_scores[10] / T) / (math.exp(raw_scores[10] / T) + math.exp(raw_scores[11] / T))
    got_10 = out.filter(pl.col("horse_id") == 10)["y_pred"].item()
    assert got_10 == pytest.approx(expected_10, abs=1e-9)


def test_race_ids_in_range_excludes_sealed_g1():
    """観点7（バグ回帰）: 封印期間内のG1レースは、fold の学習・検証どちらの
    `race_id` 一覧にも含まれない（`D-017`/`D-079`）。非G1は同じ期間内でも
    封印されない（`D-003`: 学習データ全体は封印対象ではない）。
    """
    from umagic.orchestration import _race_ids_in_range

    conn = _make_calibration_fixture_conn()  # race_id=1: G1・2020-01-01, race_id=2: 非G1・同日
    today = date(2022, 6, 1)  # sealed_years=3 → 封印窓 [2019-06-01, 2022-06-01]（2020-01-01を含む）
    ids = _race_ids_in_range(conn, date(2019, 1, 1), date(2021, 1, 1), today=today, sealed_years=3)
    conn.close()

    assert 1 not in ids  # G1・封印期間内 → 除外
    assert 2 in ids  # 非G1 → 封印されない


def test_extra_lgb_params_merge_and_categorical_feature_wins(orch_conn):
    """`Stage2FoldRunner.extra_lgb_params`（`Q-039` 検証用フック）:
    LightGBM の `params` にそのまま合流するが、`categorical_feature` は
    `predict_fold()` が自前で組み立てた値が常に勝つ（後書き優先）。
    """
    runner = Stage2FoldRunner(
        today=date(2026, 1, 1), n_blocks=2, min_category_count=2,
        num_boost_round=5, early_stopping_rounds=2,
        extra_lgb_params={"cat_smooth": 200.0, "cat_l2": 200.0, "categorical_feature": "should_be_overridden"},
    )
    fold = make_folds(orch_conn, today=date(2026, 1, 1))[0]
    out = runner.predict_fold(orch_conn, fold)
    assert out.height > 0  # 例外なく最後まで走る（LightGBMがcat_smooth/cat_l2を受理する）


def test_include_track_variant_toggle_controls_f302(orch_conn):
    """`Stage2FoldRunner.include_track_variant`（`D-110` の比較実験用）:
    **既定は `False`**（`D-110`: `F-302` が7fold全てで `all` 母集団を悪化
    させたため）。`False` なら `F-301` を呼ばず `f302` が全欠損のまま、
    `True` を明示すれば `track_variant_fit_all()` の結果が使われ `f302`
    に実値が入る。
    """
    fold = make_folds(orch_conn, today=date(2026, 1, 1))[0]

    off = Stage2FoldRunner(today=date(2026, 1, 1))
    assert off.include_track_variant is False  # 既定値（D-110）
    out_off = off.predict_fold(orch_conn, fold)
    assert out_off.height > 0  # 結線自体は例外なく走る

    on = Stage2FoldRunner(today=date(2026, 1, 1), include_track_variant=True)
    train_ids = on._race_ids(orch_conn, fold.train_start, fold.train_end)
    valid_ids = on._race_ids(orch_conn, fold.valid_start, fold.valid_end)
    horse_effects_on = track_variant_fit_all(orch_conn, fold, train_ids, n_blocks=on.n_blocks)
    horse_effects_off = None  # include_track_variant=False のときの assemble 呼び出しと同じ

    f102 = pl.DataFrame(schema={"race_id": pl.Int64, "f102": pl.Float64})
    all_ids = train_ids + valid_ids
    mat_on = assemble_stage2_matrix(
        orch_conn, all_ids, as_of=fold.valid_start, f102=f102, horse_effects=horse_effects_on,
    )
    mat_off = assemble_stage2_matrix(
        orch_conn, all_ids, as_of=fold.valid_start, f102=f102, horse_effects=horse_effects_off,
    )
    assert mat_off["f302"].is_null().all()
    assert not mat_on["f302"].is_null().all()


def test_walk_forward_end_to_end_smoke(orch_conn):
    """観点5（スモークテスト）: Stage 1 → Stage 2 → 校正 の結線が最後まで
    例外なく走り、`run_walk_forward()` の契約どおりの出力になる。
    実データでの精度は検証しない（`006`/`007`/`015` 個別のテストの役割）。
    """
    runner = Stage2FoldRunner(
        today=date(2026, 1, 1), n_blocks=2, min_category_count=2,
        num_boost_round=10, early_stopping_rounds=3,
    )
    out = run_walk_forward(orch_conn, predict_fold=runner.predict_fold, today=date(2026, 1, 1))

    assert set(out.columns) == {"race_id", "horse_id", "fold_index", "y_true", "y_pred"}
    assert out.height > 0
    # 検証期間は 2018 年のみ（min_train_years=3、データが2015-2018）→ fold 1本
    assert sorted(out["fold_index"].unique().to_list()) == [0]
    # y_pred はレース内 softmax なので (0,1) に収まる
    assert out["y_pred"].min() > 0.0
    assert out["y_pred"].max() < 1.0
    # レース単位で win_prob の合計が 1.0 に近い（R-002）
    sums = out.group_by("race_id").agg(pl.col("y_pred").sum().alias("s"))["s"].to_list()
    assert all(abs(s - 1.0) < 1e-6 for s in sums)

    # 校正（G1のみ）を後段で適用しても例外にならない
    assert len(runner.fold_calibrators) == 1
    from umagic.orchestration import apply_g1_calibration
    calibrated = apply_g1_calibration(orch_conn, out, runner)
    assert set(calibrated.columns) == set(out.columns)

    # Q-037診断用: G1 の out-of-fold スコアが n_starters 付きで保持される
    assert 0 in runner.fold_oof_g1
    assert {"race_id", "horse_id", "score", "is_winner", "n_starters"} <= set(
        runner.fold_oof_g1[0].columns
    )

    # y_true は同着頭数で等分済み（D-074）。今回の fixture に同着は無いので
    # レースごとの合計は常に1.0（勝者1頭）になる
    y_true_sums = out.group_by("race_id").agg(pl.col("y_true").sum().alias("s"))["s"].to_list()
    assert all(abs(s - 1.0) < 1e-9 for s in y_true_sums)


def test_run_p3_completion_check_smoke(orch_conn):
    """観点6（スモークテスト）: `R-023` 判定の一連の計算が例外なく走る。

    実データではないため `passes_r023` の真偽自体は検証しない。
    """
    from umagic.orchestration import P3CompletionResult, run_p3_completion_check

    runner = Stage2FoldRunner(
        today=date(2026, 1, 1), n_blocks=2, min_category_count=2,
        num_boost_round=10, early_stopping_rounds=3,
    )
    result = run_p3_completion_check(orch_conn, today=date(2026, 1, 1), runner=runner)

    assert isinstance(result, P3CompletionResult)
    assert result.all_n_races > 0
    assert result.g1_n_races > 0
    assert result.all_model_logloss == result.all_model_logloss  # NaN でない
    assert result.g1_model_logloss == result.g1_model_logloss


def test_production_cache_build_and_predict_smoke(orch_conn, tmp_path):
    """観点7（スモークテスト）: `D-183`の推論キャッシュが例外なく作れて、
    キャッシュ済みモデルで（対象レース1件だけ特徴量計算して）予測できる。

    真の未来レースではなく既存レースの1件をそのまま対象に使う——
    `predict_with_cache()` の呼び出し経路（Stage1予測→対象1件だけの
    特徴量計算→カテゴリ変換→予測）が壊れていないかのスモークテストで、
    リーク安全性や精度は検証しない（それは`D-054`のリークテスト・
    walk-forward検証の役割）。
    """
    from umagic.production_model import build_production_cache_from_conn, predict_with_cache

    runner = Stage2FoldRunner(
        today=date(2026, 1, 1), n_blocks=2, min_category_count=2,
        num_boost_round=10, early_stopping_rounds=3,
    )
    cache_dir = tmp_path / "prediction_cache"
    meta = build_production_cache_from_conn(orch_conn, cache_dir, runner=runner)
    assert meta["n_train_races"] > 0
    assert (cache_dir / "stage1").exists()
    assert (cache_dir / "stage2").exists()
    assert (cache_dir / "category_mappings.json").exists()

    target = orch_conn.execute(
        "SELECT race_id, date FROM races ORDER BY race_id DESC LIMIT 1"
    ).fetchone()
    race_id, race_date = target

    out = predict_with_cache(orch_conn, race_id, race_date, cache_dir)
    assert set(out.columns) == {"race_id", "horse_id", "win_prob"}
    assert out.height > 0
    assert out["win_prob"].min() > 0.0
    assert out["win_prob"].max() < 1.0
    assert abs(out["win_prob"].sum() - 1.0) < 1e-6
