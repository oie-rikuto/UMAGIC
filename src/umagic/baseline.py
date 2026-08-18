"""`P-2` ベースライン（`docs/spec/005-baseline.md`）。

市場確率 `normalize(1/単勝オッズ)` の確率指標と、ベタ買い戦略の回収率を
算出する。**学習を伴わない**ため `003-features.md` にも
`014-training-pipeline.md` にも依存しない（`D-075`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import duckdb
import polars as pl

from umagic.sealed import is_sealed

Population = Literal["all", "g1"]
BetType = Literal["単勝", "複勝", "ワイド"]

DEFAULT_SEALED_YEARS = 3


@dataclass(frozen=True)
class TargetRaces:
    """対象母集団の抽出結果（`005-baseline.md` 1節）。"""

    races: pl.DataFrame  # 列: race_id, date, grade, n_starters
    n_sealed_g1_excluded: int  # 封印により除外したG1レース数（D-076: 開封回数には計上しない）


@dataclass(frozen=True)
class ProbabilityMetrics:
    """市場確率の確率指標（`005-baseline.md` 2節）。"""

    population: Population
    n_races: int
    n_runners: int
    log_loss: float
    brier: float
    top1_hit_rate: float  # 最大確率の馬が1着だった割合
    top3_hit_rate: float  # 最大確率の馬が3着以内だった割合


_PROBABILITY_RUNNERS_SQL = """
SELECT race_id, number, odds_win, finish_pos, popularity
FROM runners
WHERE race_id = ANY(?)
  AND status NOT IN ('出走取消', '競走除外')
  AND odds_win IS NOT NULL
ORDER BY race_id, number
"""


def probability_metrics(
    conn: duckdb.DuckDBPyConnection, race_ids: list[int], *, population: Population,
) -> ProbabilityMetrics:
    """市場確率 `normalize(1/odds_win)` の確率指標を計算する。

    `出走取消`/`競走除外`（`D-073`）は正規化の対象に含めない。1着同着は
    正解ラベルを同着頭数で等分する（`D-074`）。
    """
    df = conn.execute(_PROBABILITY_RUNNERS_SQL, [race_ids]).pl()
    if df.is_empty():
        nan = float("nan")
        return ProbabilityMetrics(
            population=population, n_races=0, n_runners=0,
            log_loss=nan, brier=nan, top1_hit_rate=nan, top3_hit_rate=nan,
        )

    df = df.with_columns((1.0 / pl.col("odds_win")).alias("inv_odds"))
    df = df.with_columns(
        (pl.col("inv_odds") / pl.col("inv_odds").sum().over("race_id")).alias("p")
    )

    n_winners = (
        df.filter(pl.col("finish_pos") == 1)
        .group_by("race_id")
        .agg(pl.len().alias("n_winners"))
    )
    df = df.join(n_winners, on="race_id", how="left")
    df = df.with_columns(
        pl.when(pl.col("finish_pos") == 1)
        .then(1.0 / pl.col("n_winners"))
        .otherwise(0.0)
        .alias("y")
    )

    n_races = df["race_id"].n_unique()
    log_loss = -df.select((pl.col("y") * pl.col("p").log()).sum()).item() / n_races
    brier = df.select(((pl.col("p") - pl.col("y")) ** 2).sum()).item() / n_races

    # Top-1/Top-3: レースごとに p が最大の馬を選ぶ。同値は popularity が小さい方（D-077）
    picks = (
        df.sort(["race_id", "p", "popularity"], descending=[False, True, False])
        .group_by("race_id", maintain_order=True)
        .agg(pl.col("finish_pos").first().alias("picked_finish_pos"))
    )
    top1 = picks.select((pl.col("picked_finish_pos") == 1).mean()).item()
    top3 = picks.select((pl.col("picked_finish_pos") <= 3).mean()).item()

    return ProbabilityMetrics(
        population=population, n_races=n_races, n_runners=df.height,
        log_loss=log_loss, brier=brier, top1_hit_rate=top1, top3_hit_rate=top3,
    )


Strategy = Literal["favorite", "uniform"]

_PURCHASE_ELIGIBLE_SQL = """
SELECT race_id, number, popularity
FROM runners
WHERE race_id = ANY(?) AND status NOT IN ('出走取消', '競走除外')
ORDER BY race_id, number
"""

_PAYOUTS_SQL = """
SELECT race_id, comb_key, payout
FROM payouts
WHERE race_id = ANY(?) AND bet_type = ?
"""

_LEDGER_SCHEMA = {
    "race_id": pl.Int64, "n_bets": pl.Int64, "n_hits": pl.Int64,
    "stake_yen": pl.Int64, "payout_yen": pl.Int64,
}


def race_ledger(
    conn: duckdb.DuckDBPyConnection, race_ids: list[int], *, strategy: Strategy, bet_type: BetType,
) -> pl.DataFrame:
    """`race_id` ごとの購入点数・的中数・投資額・払戻額を返す（`005-baseline.md` 4節）。

    的中判定は `payouts` の `comb_key` に行が存在するかで行う（`D-072`）。
    着順から複勝・ワイドの成立条件を再実装しない。取消・除外（`D-073`）は
    購入対象に含めない。
    """
    if not race_ids:
        return pl.DataFrame(schema=_LEDGER_SCHEMA)

    runners = conn.execute(_PURCHASE_ELIGIBLE_SQL, [race_ids]).pl()
    payouts = conn.execute(_PAYOUTS_SQL, [race_ids, bet_type]).pl()
    payout_map: dict[tuple[int, str], int] = {
        (row["race_id"], row["comb_key"]): row["payout"] for row in payouts.iter_rows(named=True)
    }

    rows: list[tuple] = []
    if not runners.is_empty():
        for key, group in runners.group_by("race_id", maintain_order=True):
            race_id = key[0] if isinstance(key, tuple) else key
            group = group.sort("number")
            numbers = group["number"].to_list()
            popularities = dict(zip(numbers, group["popularity"].to_list()))

            if strategy == "favorite":
                favorite = next((n for n, p in popularities.items() if p == 1), None)
                targets = [favorite] if favorite is not None else []
            else:
                targets = numbers

            if bet_type in ("単勝", "複勝"):
                comb_keys = [str(n) for n in targets]
            else:  # ワイド
                if strategy == "favorite":
                    comb_keys = (
                        [f"{min(targets[0], o)}-{max(targets[0], o)}"
                         for o in numbers if o != targets[0]]
                        if targets else []
                    )
                else:
                    comb_keys = [
                        f"{min(a, b)}-{max(a, b)}"
                        for i, a in enumerate(numbers) for b in numbers[i + 1:]
                    ]

            n_bets = len(comb_keys)
            hits = [payout_map[(race_id, k)] for k in comb_keys if (race_id, k) in payout_map]
            rows.append((race_id, n_bets, len(hits), n_bets * 100, sum(hits)))

    return pl.DataFrame(rows, schema=_LEDGER_SCHEMA, orient="row")


def target_races(
    conn: duckdb.DuckDBPyConnection,
    *,
    population: Population,
    today: date,
    sealed_years: int = DEFAULT_SEALED_YEARS,
) -> TargetRaces:
    """`population` に応じた対象レースを抽出する。

    封印セット（`D-017`）はいずれの母集団からも除外する（`D-076`）。
    `is_sealed` は `grade != 'G1'` のレースを常に非封印として扱うため、
    封印期間内の**非G1**レースは除外されない。
    """
    all_races = conn.execute(
        "SELECT race_id, date, grade, n_starters FROM races ORDER BY race_id"
    ).pl()

    sealed_mask = [
        is_sealed(d, g, today=today, n_years=sealed_years)
        for d, g in zip(all_races["date"].to_list(), all_races["grade"].to_list())
    ]
    n_sealed = sum(sealed_mask)
    kept = all_races.filter(~pl.Series(sealed_mask, dtype=pl.Boolean))

    if population == "g1":
        kept = kept.filter(pl.col("grade") == "G1")

    return TargetRaces(races=kept.sort("race_id"), n_sealed_g1_excluded=n_sealed)
