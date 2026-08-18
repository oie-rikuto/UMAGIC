"""`P-2` ベースライン（`docs/spec/005-baseline.md`）。

市場確率 `normalize(1/単勝オッズ)` の確率指標と、ベタ買い戦略の回収率を
算出する。**学習を伴わない**ため `003-features.md` にも
`014-training-pipeline.md` にも依存しない（`D-075`）。
"""

from __future__ import annotations

import random
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


@dataclass(frozen=True)
class ReturnMetrics:
    """ベタ買い戦略の回収率（`005-baseline.md` 4節・5節）。"""

    population: Population
    strategy: Strategy
    bet_type: BetType
    n_races: int
    n_bets: int
    n_hits: int
    stake_yen: int
    payout_yen: int
    roi: float
    roi_ci_low: float
    roi_ci_high: float


DEFAULT_BOOTSTRAP_N = 10_000
DEFAULT_SEED = 20260819


def bootstrap_roi_ci(
    ledger: pl.DataFrame, *, bootstrap_n: int = DEFAULT_BOOTSTRAP_N, seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """`race_ledger()` の出力をレース単位で復元抽出し、回収率の95%CIを返す（`D-078`）。

    パーセンタイル法。`seed` は `random.Random(seed)` を経由して各回の
    `ledger.sample()` 呼び出し用の副シードを生成する。**`seed` を単純に
    `seed + i` として毎回インクリメントしない** — 隣接するシード値
    （`seed=1` と `seed=2` など）は `polars` の実装上ほぼ同じ抽出列を
    生成し、パーセンタイル境界が偶然一致してしまう。`Random(seed)` を
    経由すると、近い `seed` 同士でも副シード列は無相関になる。
    同じ `(ledger, bootstrap_n, seed)` からは常に同じ結果が再現する（`R-021`）。
    """
    n = ledger.height
    if n == 0:
        return (float("nan"), float("nan"))

    total_stake = ledger["stake_yen"].sum()
    if n == 1:
        roi = ledger["payout_yen"][0] / total_stake if total_stake else float("nan")
        return (roi, roi)

    rng = random.Random(seed)
    rois: list[float] = []
    for _ in range(bootstrap_n):
        sub_seed = rng.randrange(2**32)
        resampled = ledger.sample(n=n, with_replacement=True, seed=sub_seed)
        stake = resampled["stake_yen"].sum()
        payout = resampled["payout_yen"].sum()
        rois.append(payout / stake if stake else 0.0)

    rois.sort()
    lo_idx = int(0.025 * bootstrap_n)
    hi_idx = min(int(0.975 * bootstrap_n), bootstrap_n - 1)
    return (rois[lo_idx], rois[hi_idx])


def return_metrics(
    conn: duckdb.DuckDBPyConnection,
    race_ids: list[int],
    *,
    population: Population,
    strategy: Strategy,
    bet_type: BetType,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> ReturnMetrics:
    """`race_ledger()` を集計し、ブートストラップCIを付けた `ReturnMetrics` を返す。"""
    ledger = race_ledger(conn, race_ids, strategy=strategy, bet_type=bet_type)
    n_races = ledger.height
    if n_races == 0:
        nan = float("nan")
        return ReturnMetrics(
            population=population, strategy=strategy, bet_type=bet_type,
            n_races=0, n_bets=0, n_hits=0, stake_yen=0, payout_yen=0,
            roi=nan, roi_ci_low=nan, roi_ci_high=nan,
        )

    n_bets = ledger["n_bets"].sum()
    n_hits = ledger["n_hits"].sum()
    stake = ledger["stake_yen"].sum()
    payout = ledger["payout_yen"].sum()
    roi = payout / stake if stake else float("nan")
    ci_low, ci_high = bootstrap_roi_ci(ledger, bootstrap_n=bootstrap_n, seed=seed)

    return ReturnMetrics(
        population=population, strategy=strategy, bet_type=bet_type,
        n_races=n_races, n_bets=n_bets, n_hits=n_hits,
        stake_yen=stake, payout_yen=payout, roi=roi,
        roi_ci_low=ci_low, roi_ci_high=ci_high,
    )


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


_STRATEGIES: tuple[Strategy, ...] = ("favorite", "uniform")
_BET_TYPES: tuple[BetType, ...] = ("単勝", "複勝", "ワイド")
_POPULATIONS: tuple[Population, ...] = ("all", "g1")


def _point_roi(ledger: pl.DataFrame) -> float:
    """`race_ledger()` の出力から、ブートストラップを伴わない点推定の回収率を返す。"""
    if ledger.is_empty():
        return float("nan")
    stake = ledger["stake_yen"].sum()
    payout = ledger["payout_yen"].sum()
    return payout / stake if stake else float("nan")


def by_era_breakdown(
    conn: duckdb.DuckDBPyConnection,
    *,
    today: date,
    sealed_years: int = DEFAULT_SEALED_YEARS,
) -> pl.DataFrame:
    """年別（`YEAR(races.date)`）の内訳（`005-baseline.md` 3節 / `D-018` / `R-027`）。

    **点推定のみを出す（ブートストラップCIは含まない）。** 年×母集団×戦略×
    券種の全組み合わせで10,000回のリサンプリングを回すと実行時間が過大になる。
    信頼区間はトップレベルの `ReturnMetrics`（`run_baseline` の `returns`）が持つ。
    """
    rows: list[tuple] = []
    schema = [
        "year", "population", "metric_kind", "metric_name", "strategy", "bet_type",
        "n_races", "value",
    ]

    for population in _POPULATIONS:
        tr = target_races(conn, population=population, today=today, sealed_years=sealed_years)
        races = tr.races
        if races.is_empty():
            continue
        races = races.with_columns(pl.col("date").dt.year().alias("year"))
        for year in races["year"].unique().sort().to_list():
            race_ids = races.filter(pl.col("year") == year)["race_id"].to_list()

            pm = probability_metrics(conn, race_ids, population=population)
            for metric_name, value in [
                ("log_loss", pm.log_loss),
                ("brier", pm.brier),
                ("top1_hit_rate", pm.top1_hit_rate),
                ("top3_hit_rate", pm.top3_hit_rate),
            ]:
                rows.append((year, population, "probability", metric_name, None, None, pm.n_races, value))

            for strategy in _STRATEGIES:
                for bet_type in _BET_TYPES:
                    ledger = race_ledger(conn, race_ids, strategy=strategy, bet_type=bet_type)
                    rows.append(
                        (year, population, "return", "roi", strategy, bet_type, ledger.height, _point_roi(ledger))
                    )

    return pl.DataFrame(rows, schema=schema, orient="row")


@dataclass(frozen=True)
class BaselineReport:
    """`run_baseline()` の出力（`005-baseline.md` 6節）。"""

    generated_at: date
    n_sealed_g1_excluded: int
    probability: list[ProbabilityMetrics]
    returns: list[ReturnMetrics]
    by_era: pl.DataFrame

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# ベースラインレポート（{self.generated_at.isoformat()}）")
        lines.append("")
        lines.append(f"封印セットにより除外したG1: {self.n_sealed_g1_excluded}レース（`D-076`。開封回数には計上しない）")
        lines.append("")

        lines.append("## 確率指標")
        lines.append("")
        lines.append("| 母集団 | n_races | LogLoss | Brier | Top-1 | Top-3 |")
        lines.append("|---|---|---|---|---|---|")
        for pm in self.probability:
            lines.append(
                f"| {pm.population} | {pm.n_races} | {pm.log_loss:.4f} | {pm.brier:.4f} "
                f"| {pm.top1_hit_rate:.3f} | {pm.top3_hit_rate:.3f} |"
            )
        lines.append("")

        lines.append("## 回収率（信頼区間付き）")
        lines.append("")
        lines.append("| 母集団 | 戦略 | 券種 | n_races | roi | 95%CI |")
        lines.append("|---|---|---|---|---|---|")
        for rm in self.returns:
            lines.append(
                f"| {rm.population} | {rm.strategy} | {rm.bet_type} | {rm.n_races} "
                f"| {rm.roi:.3f} | [{rm.roi_ci_low:.3f}, {rm.roi_ci_high:.3f}] |"
            )
        lines.append("")

        for rm in self.returns:
            if rm.population == "g1" and rm.n_races > 0:
                se = 4 / (rm.n_races**0.5)
                lines.append(
                    f"**`g1` 母集団は開発用{rm.n_races}レースで検出力が限定的**"
                    f"（単勝換算の標準誤差 ≈ {se:.0%}、`D-008` / `Q-033`）。参考値として扱う。"
                )
                break
        lines.append("")

        lines.append("## 年別内訳")
        lines.append("")
        lines.append("| 年 | 母集団 | 種別 | 指標 | 戦略 | 券種 | n_races | 値 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in self.by_era.iter_rows(named=True):
            strategy = row["strategy"] or "-"
            bet_type = row["bet_type"] or "-"
            value = row["value"]
            value_str = f"{value:.4f}" if value is not None else "NaN"
            lines.append(
                f"| {row['year']} | {row['population']} | {row['metric_kind']} "
                f"| {row['metric_name']} | {strategy} | {bet_type} | {row['n_races']} | {value_str} |"
            )

        return "\n".join(lines) + "\n"


def run_baseline(
    conn: duckdb.DuckDBPyConnection,
    *,
    today: date,
    sealed_years: int = DEFAULT_SEALED_YEARS,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> BaselineReport:
    """ベースラインレポートを組み立てる（`005-baseline.md`）。"""
    probability: list[ProbabilityMetrics] = []
    returns: list[ReturnMetrics] = []
    n_sealed_g1_excluded = 0

    for population in _POPULATIONS:
        tr = target_races(conn, population=population, today=today, sealed_years=sealed_years)
        race_ids = tr.races["race_id"].to_list()
        n_sealed_g1_excluded = tr.n_sealed_g1_excluded

        probability.append(probability_metrics(conn, race_ids, population=population))

        for strategy in _STRATEGIES:
            for bet_type in _BET_TYPES:
                returns.append(
                    return_metrics(
                        conn, race_ids, population=population, strategy=strategy,
                        bet_type=bet_type, bootstrap_n=bootstrap_n, seed=seed,
                    )
                )

    era = by_era_breakdown(conn, today=today, sealed_years=sealed_years)

    return BaselineReport(
        generated_at=today,
        n_sealed_g1_excluded=n_sealed_g1_excluded,
        probability=probability,
        returns=returns,
        by_era=era,
    )
