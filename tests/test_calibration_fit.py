"""温度の推定（`docs/spec/015-calibration.md` 2節 / `D-097` `D-098`）。"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from umagic.calibration import T_MAX, T_MIN, fit_calibrator


def _make_oof(seed, n_races, n_per_race, score_scale):
    """真の確率どおりに勝者を抽選した合成データ（`score_scale=1.0` で完全校正）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        true_logits = rng.normal(size=n_per_race)
        true_p = np.exp(true_logits) / np.exp(true_logits).sum()
        winner = rng.choice(n_per_race, p=true_p)
        score = true_logits * score_scale
        for h in range(n_per_race):
            rows.append((r, h, float(score[h]), bool(h == winner)))
    return pl.DataFrame(
        rows, schema=["race_id", "horse_id", "score", "is_winner"], orient="row",
    )


def test_calibrated_scores_converge_near_one():
    """観点6: 完全に校正済みのスコア → T* が1.0付近に収束する。"""
    oof = _make_oof(seed=3, n_races=2000, n_per_race=8, score_scale=1.0)
    cal = fit_calibrator(oof)
    assert abs(cal.temperature - 1.0) < 0.2
    assert not cal.at_bound


def test_overconfident_scores_need_higher_temperature():
    """観点7: 過度に尖ったスコア → T*>1.0 になり logloss が改善する。"""
    oof = _make_oof(seed=1, n_races=300, n_per_race=8, score_scale=5.0)
    cal = fit_calibrator(oof)
    assert cal.temperature > 1.0
    assert cal.logloss_after < cal.logloss_before


def test_dead_heat_label_split_and_prob_sums_to_one(conn):
    """観点8: 1着同着2頭を含むレース → 正解ラベルが各0.5、Σy=1（D-074と同じ）。

    `fit_calibrator` 自体は `y` を外に出さないため、間接的に確認する:
    同着2頭の win_prob が等しいスコアなら、T=1.0 での LogLoss が
    「1/2ずつの正解ラベル」を仮定した手計算と一致することで検証する。
    """
    import math
    # 1レースだけ、3頭。1,2番が同着1着(is_winner=True)、3番は3着
    oof = pl.DataFrame({
        "race_id": [1, 1, 1], "horse_id": [1, 2, 3],
        "score": [1.0, 1.0, 0.0], "is_winner": [True, True, False],
    })
    cal = fit_calibrator(oof)
    # T=1.0でのp: exp(1)/(2exp(1)+exp(0)), exp(1)/(...), exp(0)/(...)
    e1, e0 = math.exp(1.0), math.exp(0.0)
    denom = 2 * e1 + e0
    p1 = p2 = e1 / denom
    # y1=y2=0.5, y3=0
    expected_logloss_at_1 = -(0.5 * math.log(p1) + 0.5 * math.log(p2))
    assert abs(cal.logloss_before - expected_logloss_at_1) < 1e-9


def test_no_g1_in_training_period_falls_back(conn):
    """観点9: G1が0件の学習期間 → T=1.0、n_races_fit=0。例外にしない（D-098）。"""
    empty = pl.DataFrame(schema={
        "race_id": pl.Int64, "horse_id": pl.Int64, "score": pl.Float64, "is_winner": pl.Boolean,
    })
    cal = fit_calibrator(empty)
    assert cal.temperature == 1.0
    assert cal.n_races_fit == 0
    assert cal.n_runners_fit == 0


def test_extreme_scores_hit_boundary():
    """観点10: 極端なスコア（境界に張り付く入力）→ at_bound=True。例外にしない。"""
    # 非常に尖ったスコアだが勝者はランダム → 大きなTが最適になり境界に張り付く
    rng = np.random.default_rng(5)
    rows = []
    for r in range(50):
        n = 6
        score = rng.normal(scale=1000.0, size=n)  # 極端に尖らせる
        winner = rng.integers(0, n)  # スコアと無関係にランダムな勝者
        for h in range(n):
            rows.append((r, h, float(score[h]), bool(h == winner)))
    oof = pl.DataFrame(rows, schema=["race_id", "horse_id", "score", "is_winner"], orient="row")
    cal = fit_calibrator(oof)
    assert cal.at_bound
    assert cal.temperature == pytest.approx(T_MAX, abs=1e-2) or cal.temperature == pytest.approx(T_MIN, abs=1e-2)


def test_same_oof_gives_bit_exact_temperature():
    """観点11: 同じ oof で2回 fit_calibrator() → T がビット完全一致（R-021）。"""
    oof = _make_oof(seed=2, n_races=100, n_per_race=6, score_scale=2.0)
    cal1 = fit_calibrator(oof)
    cal2 = fit_calibrator(oof)
    assert cal1.temperature == cal2.temperature
    assert cal1.logloss_after == cal2.logloss_after


def test_mixing_non_g1_rows_changes_result_no_internal_filtering():
    """観点12: `oof` に非G1レースを混ぜて渡すと結果が変わる
    → `fit_calibrator` 自身はレースの種類を判定・除外しない契約であること
    をテストで固定する（呼び出し側でG1に絞る、`D-098`）。
    """
    g1_only = _make_oof(seed=9, n_races=200, n_per_race=8, score_scale=1.0)
    extra = _make_oof(seed=99, n_races=50, n_per_race=8, score_scale=6.0)
    extra = extra.with_columns((pl.col("race_id") + 100000).alias("race_id"))
    mixed = pl.concat([g1_only, extra])

    cal_g1 = fit_calibrator(g1_only)
    cal_mixed = fit_calibrator(mixed)

    assert cal_mixed.n_races_fit == g1_only["race_id"].n_unique() + 50
    assert cal_mixed.temperature != cal_g1.temperature
