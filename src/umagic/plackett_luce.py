"""Plackett-Luce top-K の目的関数（`D-130`）。

`R-029` が測るのはレース単位の勝利確率の負対数尤度そのものだが、`D-095`
以来の既定である `lambdarank` は NDCG（順位）を最適化しており、**学習と
評価の目的が一致していない。** この目的関数はその不一致を解消する。

    log L = Σ_k [ z_(k) − log Σ_{j∈R_k} exp(z_j) ]      R_k は第k段の残存集合
    ∂L/∂z_i   = Σ_k [ 1{i∈R_k}·p_i^(k) − 1{i=(k)} ]
    ∂²L/∂z_i² = Σ_k p_i^(k)(1 − p_i^(k))

`K=1` はレース内 softmax の交差エントロピー（条件付きlogit）に一致する。
**`K=1` では `lambdarank` に負ける**（`D-130`）——ラベル 3/2/1/0（`D-093`）が
持つ2着・3着の情報を捨てるため。`K` を2以上にすると符号が反転する。

**重みは自前で掛ける。** LightGBM はカスタム目的関数のとき `Dataset` の
`weight` を勾配に自動適用しないため、これを怠ると `class_weights`
（`D-081`）が黙って無効化される。

**ヘシアンに下限を置く。** `p(1−p)` は `p` が0/1に寄ると0に近づき、葉の値
`−Σgrad/Σhess` が発散しうる。`min_data_in_leaf=1` を使う本プロジェクトでは
`min_sum_hessian_in_leaf` も併せて上げる必要がある（`D-130` の実測では
`1.0` と `learning_rate=0.02` の併用が最良だった）。
"""

from __future__ import annotations

import numpy as np

MIN_HESS = 1e-6
WINNER_LABEL = 3  # D-093: 1着のラベル


def race_bounds(group: list[int]) -> np.ndarray:
    """`group`（レースごとの行数）から `[start, end)` の境界配列を作る。"""
    ends = np.cumsum(np.asarray(group, dtype=np.int64))
    starts = np.concatenate([[0], ends[:-1]])
    return np.stack([starts, ends], axis=1)


def _placed_order(label: np.ndarray, s: int, e: int, k: int) -> list[int]:
    """レース `[s,e)` の上位K頭を着順に並べた行インデックスを返す。

    `D-093` のラベル（1着3 / 2着2 / 3着1 / それ以外0）を降順に見る。
    **同着は行順で決める**（PL分解は同着を表現できないため。実データでは稀）。
    """
    idx = [i for i in range(s, e) if label[i] > 0]
    idx.sort(key=lambda i: (-label[i], i))
    return idx[:k]


def make_pl_objective(
    group: list[int], label: np.ndarray, weight: np.ndarray | None, *, top_k: int = 3,
):
    """LightGBM の `params["objective"]` に渡すカスタム目的関数を作る。

    `(レース数 × 最大頭数)` のパディング行列で計算し、ループは `top_k` 回
    だけにする。JRAの最大出走頭数は18なので行列は小さい。
    """
    bounds = race_bounds(group)
    label = np.asarray(label)
    n_races = len(bounds)
    if n_races == 0:
        raise ValueError("group が空。レース境界を作れない")
    m = int(max(e - s for s, e in bounds))

    idx = np.zeros((n_races, m), dtype=np.int64)
    valid = np.zeros((n_races, m), dtype=bool)
    for r, (s, e) in enumerate(bounds):
        k = e - s
        idx[r, :k] = np.arange(s, e)
        valid[r, :k] = True

    # 各段の着順（レース内の列位置）。その段が存在しないレースは -1
    placed = np.full((n_races, top_k), -1, dtype=np.int64)
    for r, (s, e) in enumerate(bounds):
        for j, row in enumerate(_placed_order(label, s, e, top_k)):
            placed[r, j] = row - s

    w = None if weight is None else np.asarray(weight, dtype=np.float64)
    rows = np.arange(n_races)
    neg_inf = np.float64(-np.inf)

    def fobj(a, b):
        # LightGBM 4.x は `(preds, Dataset)` で呼ぶ。旧版の `(y_true, y_pred)` にも備える
        z = np.asarray(b, dtype=np.float64) if isinstance(b, np.ndarray) else np.asarray(a, dtype=np.float64)
        z2 = np.where(valid, z[idx], neg_inf)
        alive = valid.copy()
        g2 = np.zeros((n_races, m), dtype=np.float64)
        h2 = np.zeros((n_races, m), dtype=np.float64)
        for k in range(top_k):
            has = placed[:, k] >= 0
            if not has.any():
                break
            zz = np.where(alive, z2, neg_inf)
            mx = np.max(zz, axis=1, keepdims=True)
            ez = np.where(alive, np.exp(zz - mx), 0.0)
            p = ez / ez.sum(axis=1, keepdims=True)
            g2[has] += p[has]
            h2[has] += (p * (1.0 - p))[has]
            hr, hc = rows[has], placed[has, k]
            g2[hr, hc] -= 1.0
            alive[hr, hc] = False

        grad = np.zeros_like(z)
        hess = np.zeros_like(z)
        grad[idx[valid]] = g2[valid]
        hess[idx[valid]] = h2[valid]
        hess = np.maximum(hess, MIN_HESS)
        if w is not None:
            grad = grad * w
            hess = hess * w
        return grad, hess

    return fobj
