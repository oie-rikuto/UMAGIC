"""単勝勝率から組み合わせ馬券の確率を近似する（Harville、`D-202`）。

UMAGICが直接出力するのは単勝勝率のみ（`docs/mcp-server.md`）。馬連・馬単・
ワイド・3連複・3連単の確率は、Harville(1973)の近似——「1着馬を確率どおり
選んだ後、残りの馬の間で再正規化した勝率をそのまま2着の確率として使う」
を再帰適用して概算する。

**この近似には既知の偏りがある。** `D-153`は同じ近似を複勝市場に適用した
結果、人気馬の複勝確率を最大10.6ポイント過大評価すると実測した
（予測0.825→実測0.719）。他の組み合わせ馬券でも同方向の偏りがあると
見るのが自然だが、複勝以外では個別に測定していない。

**したがってこのモジュールの出力は「参考値」であり、的中率・回収率の
検証はできない。** `D-008`の規律により、3連単（控除率27.5%・超高分散）
は現実的な標本数では有意差を検出できず、`prediction_log`の対象にも
含めていない（`Pick.validate()`が拒否する）。ここで返す確率・組み合わせ
は、エージェントが**参考として提示する**ための計算補助であって、
選択の正しさを保証するものではない。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Literal

BetType = Literal["馬連", "馬単", "ワイド", "3連複", "3連単"]

# `D-008`: 3連単は控除率27.5%・超高分散で検出力が無く、複勝・ワイドが
# 主評価券種。ここでは提案の対象そのものは制限しない（記録側で制限する）
SUPPORTED_BET_TYPES: frozenset[str] = frozenset({"馬連", "馬単", "ワイド", "3連複", "3連単"})


@dataclass(frozen=True)
class Combo:
    """1つの組み合わせとその近似確率。"""

    numbers: tuple[int, ...]  # 馬単・3連単は着順どおり、他は昇順に正規化
    prob: float


def _exacta_prob(p: dict[int, float], i: int, j: int) -> float:
    """馬単: `i`が1着・`j`が2着になる近似確率。"""
    denom = 1.0 - p[i]
    if denom <= 1e-12:
        return 0.0
    return p[i] * p[j] / denom


def _trifecta_prob(p: dict[int, float], i: int, j: int, k: int) -> float:
    """3連単: `i`1着・`j`2着・`k`3着になる近似確率。"""
    d1 = 1.0 - p[i]
    if d1 <= 1e-12:
        return 0.0
    d2 = 1.0 - p[i] - p[j]
    if d2 <= 1e-12:
        return 0.0
    return p[i] * (p[j] / d1) * (p[k] / d2)


def compute_combos(win_prob: dict[int, float], bet_type: BetType) -> list[Combo]:
    """全馬の単勝勝率（馬番→勝率、レース内合計1.0）から、指定券種の
    組み合わせを全列挙し、近似確率の降順で返す。

    頭数`n`に対し馬単・3連単は最大`n×(n-1)`・`n×(n-1)×(n-2)`通り、
    馬連・ワイド・3連複はその半分/1/6——JRAの最大出走頭数（18頭）でも
    計算量は問題にならない。
    """
    if bet_type not in SUPPORTED_BET_TYPES:
        raise ValueError(f"bet_type={bet_type!r} は対象外です（{sorted(SUPPORTED_BET_TYPES)}）")

    numbers = list(win_prob.keys())
    out: dict[tuple[int, ...], float] = {}

    if bet_type == "馬単":
        for i, j in permutations(numbers, 2):
            out[(i, j)] = _exacta_prob(win_prob, i, j)

    elif bet_type == "馬連":
        for i, j in permutations(numbers, 2):
            key = tuple(sorted((i, j)))
            out[key] = out.get(key, 0.0) + _exacta_prob(win_prob, i, j)

    elif bet_type == "3連単":
        for i, j, k in permutations(numbers, 3):
            out[(i, j, k)] = _trifecta_prob(win_prob, i, j, k)

    elif bet_type == "3連複":
        for i, j, k in permutations(numbers, 3):
            key = tuple(sorted((i, j, k)))
            out[key] = out.get(key, 0.0) + _trifecta_prob(win_prob, i, j, k)

    elif bet_type == "ワイド":
        # 「i, j がともに3着以内」＝ i, j と第三の馬 k の3頭が
        # 上位3着を占める確率を、あり得る全ての k について足し合わせる
        for i, j in permutations(numbers, 2):
            if i > j:
                continue  # 無向対を1回だけ数える
            total = 0.0
            for k in numbers:
                if k in (i, j):
                    continue
                total += (
                    _trifecta_prob(win_prob, i, j, k)
                    + _trifecta_prob(win_prob, i, k, j)
                    + _trifecta_prob(win_prob, j, i, k)
                    + _trifecta_prob(win_prob, j, k, i)
                    + _trifecta_prob(win_prob, k, i, j)
                    + _trifecta_prob(win_prob, k, j, i)
                )
            out[(i, j)] = total

    combos = [Combo(numbers=key, prob=prob) for key, prob in out.items()]
    combos.sort(key=lambda c: c.prob, reverse=True)
    return combos


def top_combos(win_prob: dict[int, float], bet_type: BetType, top_n: int = 5) -> list[Combo]:
    """`compute_combos`の上位`top_n`件だけを返す（表示用）。"""
    return compute_combos(win_prob, bet_type)[:top_n]
