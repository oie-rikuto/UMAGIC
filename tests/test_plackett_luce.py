"""Plackett-Luce top-K の目的関数（`src/umagic/plackett_luce.py` / `D-130`）。"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from umagic.plackett_luce import _placed_order, make_pl_objective, race_bounds
from umagic.stage2 import _objective_params


def _pl_loss(z: np.ndarray, group: list[int], label: np.ndarray, top_k: int) -> float:
    """素朴な参照実装。ベクトル化版の検算に使う。"""
    total = 0.0
    for s, e in race_bounds(group):
        alive = np.ones(e - s, dtype=bool)
        for winner in _placed_order(label, s, e, top_k):
            zz = z[s:e].copy()
            zz[~alive] = -np.inf
            mx = zz[alive].max()
            total -= (z[winner] - mx) - np.log(np.exp(zz[alive] - mx).sum())
            alive[winner - s] = False
    return total


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_gradient_and_hessian_match_numeric_derivative(top_k):
    """解析解が数値微分と一致する。"""
    group = [5, 4, 6]
    label = np.array([3, 0, 2, 1, 0, 0, 3, 0, 2, 3, 2, 1, 0, 0, 0])
    rng = np.random.default_rng(11)
    z = rng.normal(size=sum(group))

    fobj = make_pl_objective(group, label, None, top_k=top_k)
    grad, hess = fobj(label, z)

    eps = 1e-6
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        num_g = (_pl_loss(zp, group, label, top_k) - _pl_loss(zm, group, label, top_k)) / (2 * eps)
        assert grad[i] == pytest.approx(num_g, abs=1e-6)

        gp, _ = fobj(label, zp)
        gm, _ = fobj(label, zm)
        assert hess[i] == pytest.approx((gp[i] - gm[i]) / (2 * eps), abs=1e-6)


def test_weight_is_applied_by_hand():
    """**カスタム目的関数では LightGBM が `weight` を適用しない**（`D-130`）。

    自前で掛けていることを固定する。怠ると `class_weights`（`D-081`）が
    黙って無効化される。
    """
    group = [4, 3]
    label = np.array([3, 2, 1, 0, 3, 2, 0])
    rng = np.random.default_rng(12)
    z = rng.normal(size=7)
    w = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0])

    g0, h0 = make_pl_objective(group, label, None, top_k=3)(label, z)
    gw, hw = make_pl_objective(group, label, w, top_k=3)(label, z)

    assert gw == pytest.approx(g0 * w)
    # ヘシアンは下限（MIN_HESS）を掛ける前に適用されるため、重みぶんだけ増える
    assert hw == pytest.approx(h0 * w)


def test_top_k_1_is_conditional_logit():
    """`K=1` はレース内 softmax の交差エントロピーに一致する。"""
    group = [5]
    label = np.array([3, 0, 0, 0, 0])
    rng = np.random.default_rng(13)
    z = rng.normal(size=5)

    grad, _ = make_pl_objective(group, label, None, top_k=1)(label, z)
    p = np.exp(z - z.max())
    p = p / p.sum()
    expected = p.copy()
    expected[0] -= 1.0
    assert grad == pytest.approx(expected)


def test_dead_heat_for_first_is_ordered_by_row():
    """1着同着は行順で並べる（PL分解が同着を表現できないため）。"""
    label = np.array([3, 3, 0])
    assert _placed_order(label, 0, 3, 3) == [0, 1]


def test_objective_params_dispatch():
    group, label = [3], np.array([3, 2, 1])
    assert _objective_params("lambdarank", group, label, None)["objective"] == "lambdarank"
    assert callable(_objective_params("pl3", group, label, None)["objective"])
    with pytest.raises(ValueError, match="未知の objective"):
        _objective_params("softmax", group, label, None)


def test_fit_stage2_accepts_pl_objective():
    """`fit_stage2()` が `objective='pl3'` を受け付け、学習が通る。"""
    from umagic.stage2 import fit_stage2

    rng = np.random.default_rng(14)
    n_races, k = 40, 6
    n = n_races * k
    x = pl.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    label = np.zeros(n, dtype=np.int64)
    for r in range(n_races):
        label[r * k] = 3
        label[r * k + 1] = 2
        label[r * k + 2] = 1
    group = pl.Series([k] * n_races)

    booster, metrics = fit_stage2(
        x=x, label=pl.Series(label), group=group,
        sample_weight=pl.Series(np.ones(n)), seed=0, params={},
        inner_x=x, inner_label=pl.Series(label), inner_group=group,
        num_boost_round=5, early_stopping_rounds=5, objective="pl3",
    )
    assert booster is not None
    assert "inner_logloss" in metrics
