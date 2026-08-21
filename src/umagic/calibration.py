"""`P-3` 確率校正（`docs/spec/015-calibration.md`）。

`007-stage2-ranker.md` が出す未校正のレース内 softmax を、G1での確率
として正しい尺度に直す。`D-003` のG1特化3手段のうち3番目。温度スケー
リング（`D-097`）のみを行い、**順位は変えない**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

T_MIN = 0.1
T_MAX = 10.0
T_NO_CALIBRATION = 1.0

_AT_BOUND_EPS = 1e-3
_GOLDEN_TOL = 1e-4


def softmax_by_race(scores: pl.DataFrame, *, temperature: float) -> pl.DataFrame:
    """`softmax(score / temperature)` をレース内で取る（`D-097`）。

    `scores` は `race_id, horse_id, score` を持つ。`win_prob` を追加して返す。
    `temperature = 1.0` のとき `007-stage2-ranker.md` の素の softmax と一致する。
    """
    out = scores.with_columns((pl.col("score") / temperature).alias("_scaled"))
    out = out.with_columns(
        (pl.col("_scaled") - pl.col("_scaled").max().over("race_id")).alias("_shifted")
    )
    out = out.with_columns(pl.col("_shifted").exp().alias("_exp"))
    out = out.with_columns(
        (pl.col("_exp") / pl.col("_exp").sum().over("race_id")).alias("win_prob")
    )
    return out.drop(["_scaled", "_shifted", "_exp"])


def _race_logloss(oof: pl.DataFrame, *, temperature: float) -> float:
    """`oof`（`race_id, horse_id, score, is_winner`）に温度 `temperature` を当てた
    ときのレース単位平均 LogLoss（2節）。1着同着は `D-074` と同じく等分する。
    """
    probs = softmax_by_race(oof.select(["race_id", "horse_id", "score"]), temperature=temperature)
    df = probs.join(oof.select(["race_id", "horse_id", "is_winner"]), on=["race_id", "horse_id"])

    n_winners = (
        df.filter(pl.col("is_winner"))
        .group_by("race_id")
        .agg(pl.len().alias("n_winners"))
    )
    df = df.join(n_winners, on="race_id", how="left")
    df = df.with_columns(
        pl.when(pl.col("is_winner"))
        .then(1.0 / pl.col("n_winners"))
        .otherwise(0.0)
        .alias("y")
    )

    n_races = df["race_id"].n_unique()
    if n_races == 0:
        return float("nan")
    eps = 1e-15
    total = df.select(
        (pl.col("y") * pl.col("win_prob").clip(eps, 1.0).log()).sum()
    ).item()
    return -total / n_races


def _golden_section_minimize(f, lo: float, hi: float, *, tol: float = _GOLDEN_TOL) -> float:
    """決定的な1次元・有界の最小化（`scipy` を使わない、`D-097`）。

    黄金分割探索。反復回数は区間幅と `tol` から計算されるため、同じ
    `(f, lo, hi, tol)` からは常に同じ手順で同じ結果に収束する（`R-021`）。
    """
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0

    a, b = lo, hi
    h = b - a
    if h <= tol:
        return (a + b) / 2.0

    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    fc, fd = f(c), f(d)

    for _ in range(max(n, 1)):
        if fc < fd:
            b, d, fd = d, c, fc
            h = invphi * h
            c = a + invphi2 * h
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            h = invphi * h
            d = a + invphi * h
            fd = f(d)

    return (a + b) / 2.0


@dataclass(frozen=True)
class Calibrator:
    """温度スケーリング（`D-097`）。fold ごとに1つ持つ（`D-098`）。"""

    temperature: float
    n_races_fit: int
    n_runners_fit: int
    logloss_before: float
    logloss_after: float
    at_bound: bool

    def apply(self, scores: pl.DataFrame) -> pl.DataFrame:
        """列 `race_id, horse_id, score` を受け取り `win_prob` を付けて返す。"""
        return softmax_by_race(scores, temperature=self.temperature)


def fit_calibrator(oof: pl.DataFrame) -> Calibrator:
    """`oof`（`race_id, horse_id, score, is_winner`）から温度を推定する（`D-097` / `D-098`）。

    **呼び出し側で `races.grade = 'G1'` に絞ったうえで渡すこと。** `oof` は
    `grade` を持たず、本関数はレースの種類を判定・除外しない。

    `oof` が空（学習期間にG1が0件）のときは `T = T_NO_CALIBRATION` に
    フォールバックする（`D-098`）。
    """
    if oof.is_empty():
        nan = float("nan")
        return Calibrator(
            temperature=T_NO_CALIBRATION, n_races_fit=0, n_runners_fit=0,
            logloss_before=nan, logloss_after=nan, at_bound=False,
        )

    n_races = oof["race_id"].n_unique()
    n_runners = oof.height

    logloss_before = _race_logloss(oof, temperature=T_NO_CALIBRATION)

    def objective(t: float) -> float:
        return _race_logloss(oof, temperature=t)

    t_star = _golden_section_minimize(objective, T_MIN, T_MAX)
    logloss_after = objective(t_star)
    at_bound = (t_star - T_MIN) < _AT_BOUND_EPS or (T_MAX - t_star) < _AT_BOUND_EPS

    return Calibrator(
        temperature=t_star, n_races_fit=n_races, n_runners_fit=n_runners,
        logloss_before=logloss_before, logloss_after=logloss_after, at_bound=at_bound,
    )
