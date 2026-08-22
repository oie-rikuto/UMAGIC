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
)
from umagic.training import run_walk_forward


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
    walk_forward_out = pl.DataFrame({
        "race_id": [1, 1, 2, 2], "horse_id": [10, 11, 20, 21], "fold_index": [0, 0, 0, 0],
        "y_true": [1.0, 0.0, 1.0, 0.0], "y_pred": [0.6, 0.4, 0.6, 0.4],
    })
    out = apply_g1_calibration(conn, walk_forward_out, _FakeRunner())
    conn.close()

    assert set(out["race_id"].unique().to_list()) == {1}
    assert set(out.columns) == set(walk_forward_out.columns)


def test_walk_forward_end_to_end_smoke(orch_conn):
    """観点5（スモークテスト）: Stage 1 → Stage 2 → 校正 の結線が最後まで
    例外なく走り、`run_walk_forward()` の契約どおりの出力になる。
    実データでの精度は検証しない（`006`/`007`/`015` 個別のテストの役割）。
    """
    runner = Stage2FoldRunner(
        n_blocks=2, min_category_count=2, num_boost_round=10, early_stopping_rounds=3,
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
        n_blocks=2, min_category_count=2, num_boost_round=10, early_stopping_rounds=3,
    )
    result = run_p3_completion_check(orch_conn, today=date(2026, 1, 1), runner=runner)

    assert isinstance(result, P3CompletionResult)
    assert result.all_n_races > 0
    assert result.g1_n_races > 0
    assert result.all_model_logloss == result.all_model_logloss  # NaN でない
    assert result.g1_model_logloss == result.g1_model_logloss
