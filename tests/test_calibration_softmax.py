"""温度スケーリングの softmax（`docs/spec/015-calibration.md` 4節 / `D-097`）。"""

from __future__ import annotations

import polars as pl

from umagic.calibration import softmax_by_race


def _scores(race_id, values):
    return pl.DataFrame({
        "race_id": [race_id] * len(values),
        "horse_id": list(range(len(values))),
        "score": values,
    })


def test_win_prob_sums_to_one():
    """観点1: 任意の T>0 で win_prob のレース内合計が 1.0 ± 1e-6（R-002）。"""
    for t in (0.2, 1.0, 3.5, 10.0):
        out = softmax_by_race(_scores(1, [0.1, 2.0, -1.5, 0.7]), temperature=t)
        assert abs(out["win_prob"].sum() - 1.0) < 1e-6


def test_temperature_one_matches_plain_softmax():
    """観点2: T=1.0 は007の素の softmax と一致する。"""
    import math
    values = [0.5, 1.5, -0.5]
    out = softmax_by_race(_scores(1, values), temperature=1.0)
    exp_vals = [math.exp(v) for v in values]
    expected = [e / sum(exp_vals) for e in exp_vals]
    for got, exp in zip(out["win_prob"].to_list(), expected):
        assert abs(got - exp) < 1e-9


def test_temperature_above_one_flattens():
    """観点3: T>1.0 は確率が均一方向に寄る（最大値が下がる）。"""
    values = [0.1, 3.0, -1.0, 0.5]
    p1 = softmax_by_race(_scores(1, values), temperature=1.0)["win_prob"].max()
    p3 = softmax_by_race(_scores(1, values), temperature=3.0)["win_prob"].max()
    assert p3 < p1


def test_temperature_below_one_sharpens():
    """観点4: T<1.0 は確率が尖る方向に寄る（最大値が上がる）。"""
    values = [0.1, 3.0, -1.0, 0.5]
    p1 = softmax_by_race(_scores(1, values), temperature=1.0)["win_prob"].max()
    p05 = softmax_by_race(_scores(1, values), temperature=0.5)["win_prob"].max()
    assert p05 > p1


def test_rank_unchanged_by_temperature():
    """観点5: 任意の T>0 で win_prob の順位が score の順位と一致する。"""
    values = [0.1, 3.0, -1.0, 0.5, 2.2]
    for t in (0.1, 0.5, 1.0, 2.0, 9.0):
        out = softmax_by_race(_scores(1, values), temperature=t)
        score_order = out.sort("score", descending=True)["horse_id"].to_list()
        prob_order = out.sort("win_prob", descending=True)["horse_id"].to_list()
        assert score_order == prob_order
