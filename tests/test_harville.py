"""`D-202`: Harville近似による組み合わせ馬券確率のテスト。

数学的な整合性（確率の合計）を主に検証する。近似の精度そのもの
（実際の的中率との一致）は検証できない・しない——`D-153`が複勝で
実測した偏り（最大10.6ポイント過大評価）と同種の偏りが他の券種にも
ある可能性を、このモジュール自体は検出できないため。
"""
from __future__ import annotations

import pytest

from umagic.harville import compute_combos, top_combos


def _uniform(n: int) -> dict[int, float]:
    return {i: 1.0 / n for i in range(1, n + 1)}


def _skewed(n: int) -> dict[int, float]:
    """1頭が突出して強い、より現実的な分布。"""
    raw = [0.4] + [0.6 / (n - 1)] * (n - 1)
    return {i: p for i, p in enumerate(raw, start=1)}


@pytest.mark.parametrize("bet_type", ["馬単", "馬連", "3連単", "3連複"])
@pytest.mark.parametrize("dist_fn", [_uniform, _skewed])
def test_probabilities_sum_to_one(bet_type, dist_fn):
    """馬単・馬連・3連単・3連複は、可能な組み合わせを尽くせば
    確率の合計が1.0になる（同じ着順空間を単に集約しているだけのため）。
    """
    p = dist_fn(8)
    combos = compute_combos(p, bet_type)
    total = sum(c.prob for c in combos)
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("dist_fn", [_uniform, _skewed])
def test_wide_probabilities_sum_to_three(dist_fn):
    """ワイドは1回の着順確定につき3組（上位3頭からの2頭選び）が
    同時に的中するため、全組み合わせの確率合計は3.0になる。
    """
    p = dist_fn(8)
    combos = compute_combos(p, "ワイド")
    total = sum(c.prob for c in combos)
    assert total == pytest.approx(3.0, abs=1e-9)


def test_all_probabilities_are_valid():
    p = _skewed(10)
    for bet_type in ["馬連", "馬単", "ワイド", "3連複", "3連単"]:
        for c in compute_combos(p, bet_type):
            assert 0.0 <= c.prob <= 1.0 + 1e-9


def test_favorite_ranks_first_in_exacta():
    """突出して強い馬が絡む組み合わせが上位に来ることの健全性チェック。"""
    p = _skewed(6)  # 馬番1が0.4、他は均等
    top = top_combos(p, "馬単", top_n=1)[0]
    assert 1 in top.numbers


def test_top_combos_respects_top_n():
    p = _uniform(6)
    combos = top_combos(p, "3連単", top_n=5)
    assert len(combos) == 5
    probs = [c.prob for c in combos]
    assert probs == sorted(probs, reverse=True)


def test_unsupported_bet_type_raises():
    with pytest.raises(ValueError, match="対象外"):
        compute_combos(_uniform(5), "単勝")  # type: ignore[arg-type]


def test_small_field_does_not_crash():
    """3連系は頭数2ではそもそも組み合わせが作れない——例外にせず空で返す。"""
    p = _uniform(2)
    assert compute_combos(p, "3連単") == []
    assert compute_combos(p, "3連複") == []
    assert all(c.prob == 0.0 for c in compute_combos(p, "ワイド"))  # 3着以内3頭が作れないため確率0
    # 馬単・馬連は2頭でも1通り(逆順含め2通り)成立する
    assert len(compute_combos(p, "馬単")) == 2
    assert len(compute_combos(p, "馬連")) == 1
