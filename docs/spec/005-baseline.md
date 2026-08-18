# 005 ベースライン

| | |
|---|---|
| Phase | P-2 |
| 関連決定 | `D-002` `D-003` `D-008` `D-011` `D-012` `D-015` `D-017` `D-018` `D-034` `D-071` `D-072` `D-073` `D-074` `D-075` `D-076` `D-077` `D-078` |
| 関連要件 | `R-022` `R-023` `R-025` `R-026` `R-027` |
| 先行仕様 | `001-schema.md` |
| 状態 | Draft |

## 目的

`architecture.md` 2節が定める基準線 `市場確率 = normalize(1/単勝オッズ)` の確率指標と、ベタ買い戦略の回収率を算出する。以降すべての評価がこの数値との比較で行われる。

**このモジュールは学習を伴わない。** 推定するパラメータを持たず、`003-features.md` にも依存しない。入力は中間スキーマ（`001-schema.md`）だけである。

## 入出力

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal

import duckdb
import polars as pl

Population = Literal["all", "g1"]
BetType = Literal["単勝", "複勝", "ワイド"]


@dataclass(frozen=True)
class ProbabilityMetrics:
    population: Population
    n_races: int
    n_runners: int
    log_loss: float
    brier: float
    top1_hit_rate: float          # 最大確率の馬が1着だった割合
    top3_hit_rate: float          # 最大確率の馬が3着以内だった割合


@dataclass(frozen=True)
class ReturnMetrics:
    population: Population
    strategy: str                 # 'favorite' / 'uniform'
    bet_type: BetType
    n_races: int
    n_bets: int                   # 購入点数（100円 × この数が総投資額）
    n_hits: int
    stake_yen: int
    payout_yen: int
    roi: float                    # payout_yen / stake_yen
    roi_ci_low: float             # ブートストラップ95%CI 下限
    roi_ci_high: float            # 同 上限


@dataclass(frozen=True)
class BaselineReport:
    generated_at: date
    n_sealed_g1_excluded: int
    probability: list[ProbabilityMetrics]
    returns: list[ReturnMetrics]
    by_era: pl.DataFrame          # 年別の内訳（D-018 / R-027）

    def to_markdown(self) -> str: ...


def run_baseline(
    conn: duckdb.DuckDBPyConnection,
    *,
    today: date,
    sealed_years: int = 3,
    bootstrap_n: int = 10_000,
    seed: int = 20260819,
) -> BaselineReport: ...
```

## 仕様

### 1. 対象母集団

`D-071` により2つの母集団それぞれについて全指標を算出する。

| `population` | 対象 |
|---|---|
| `all` | 全レース |
| `g1` | `races.grade = 'G1'` のみ |

**いずれの母集団からも封印セットを除外する（`D-076`）。** 判定は `umagic.sealed.is_sealed(race_date, grade, today=today, n_years=sealed_years)`。除外件数を `BaselineReport.n_sealed_g1_excluded` に記録する。**開封回数（`R-026`）には計上しない。**

```sql
-- 対象レースの抽出（population='all' の場合）
SELECT r.race_id, r.date, r.grade, r.n_starters
FROM races r
WHERE NOT (r.grade = 'G1' AND r.date BETWEEN :sealed_start AND :sealed_end)
ORDER BY r.race_id
```

**fold には分けない（`D-075`）。** 全期間一括で集計し、時系列の変化は3節の年別内訳で見る。

### 2. 市場確率

#### 2.1 購入・評価の対象となる馬

| `status`（`D-011`） | 確率の正規化に含めるか | 購入対象か（`D-073`） |
|---|---|---|
| `出走` | 含める | 対象 |
| `競走中止` | 含める | 対象（返還なし。的中しない） |
| `降着` | 含める | 対象（`finish_pos` は公式着順、`D-034`） |
| `出走取消` | **含めない** | **対象外** |
| `競走除外` | **含めない** | **対象外** |

取消・除外馬は `odds_win` と `popularity` がいずれも `NULL` である（取り込み済み3年分で確認、2026-08-19）。実装上は `odds_win IS NOT NULL` で上表の分割と一致する。

#### 2.2 確率の定義

レース `r` の対象馬集合を `S_r` とする。

```
p_i = (1 / odds_i) / Σ_{j∈S_r} (1 / odds_j)
```

`Σ_{i∈S_r} p_i = 1` が常に成り立つ。`odds_i` は `runners.odds_win`（確定単勝オッズ）。

#### 2.3 正解ラベル

`finish_pos = 1` の馬に `y = 1`。**1着が同着のときは同着頭数で等分する（`D-074`）。**

```
y_i = 1 / |{j ∈ S_r : finish_pos_j = 1}|   （finish_pos_i = 1 のとき）
y_i = 0                                     （それ以外）
```

これにより `Σ_{i∈S_r} y_i = 1` が保たれる。

#### 2.4 確率指標

| 指標 | 定義 |
|---|---|
| LogLoss | `−(1/N) Σ_r Σ_{i∈S_r} y_i · log(p_i)`。`N` はレース数 |
| Brier | `(1/N) Σ_r Σ_{i∈S_r} (p_i − y_i)²` |
| Top-1 的中率 | `p_i` が最大の馬の `finish_pos = 1` だったレースの割合 |
| Top-3 的中率 | `p_i` が最大の馬の `finish_pos <= 3` だったレースの割合 |

**`N` の単位はレースであって出走行ではない。** Top-1 / Top-3 で `p_i` 最大の馬が複数いる場合（オッズ同値、1,061レースで発生）は `popularity` が小さい方を採る（`D-077`）。

### 3. 年別内訳（`D-018` / `R-027`）

`by_era` は上記すべての指標を `YEAR(races.date)` で分割したもの。列は以下。

```
year, population, metric_kind, metric_name, strategy, bet_type, n_races, value
```

`metric_kind` は `'probability'` または `'return'`。

### 4. ベタ買い戦略

#### 4.1 戦略の定義

| `strategy` | 購入対象 |
|---|---|
| `favorite` | `runners.popularity = 1` の馬（`D-077`） |
| `uniform` | 2.1節の購入対象馬すべて |

**「1番人気」は `popularity` で定義し、`odds_win` の順位では定義しない（`D-077`）。**

市場確率については確率指標のみを出し、回収率は出さない。最大確率の馬を買う戦略は `favorite` と一致する（`D-077` により同値オッズの解きほぐしも `popularity` に従う）。

#### 4.2 券種ごとの購入点数

購入単位は **100円/点**。

| `bet_type` | `favorite` の点数 | `uniform` の点数 |
|---|---|---|
| `単勝` | 1（1番人気） | `n`（対象馬全頭） |
| `複勝` | 1（1番人気） | `n` |
| `ワイド` | `n − 1`（1番人気を含む全ペア） | `C(n, 2)`（全ペア） |

`n` は2.1節の購入対象馬の数。

#### 4.3 的中判定と払戻

**`payouts` テーブルに行が存在するかで判定する（`D-072`）。着順からルールを再構成しない。**

| `bet_type` | 的中条件 | 払戻額 |
|---|---|---|
| `単勝` | `payouts(race_id, '単勝', combination = [number])` が存在 | その行の `payout` 円 |
| `複勝` | `payouts(race_id, '複勝', combination = [number])` が存在 | 同上 |
| `ワイド` | `payouts(race_id, 'ワイド', combination = [min(a,b), max(a,b)])` が存在 | 同上 |

`payouts.payout` は100円あたりの払戻額である。`combination` は昇順に格納されている（`001-schema.md`）。

該当行が無ければ払戻0円。

#### 4.4 回収率

```
roi = Σ payout_yen / Σ stake_yen
stake_yen = 100 × n_bets
```

### 5. ブートストラップ信頼区間（`R-025`）

**リサンプリング単位はレースとする（`D-078`）。** 購入点単位でリサンプリングしない。

```python
# 対象レースの (stake_yen, payout_yen) の組を N 個用意する
# B 回、N レースを復元抽出し、各回の roi を計算する
# B 個の roi の 2.5 パーセンタイルと 97.5 パーセンタイルを CI とする
```

| パラメータ | 既定値 |
|---|---|
| `bootstrap_n`（B） | `10_000` |
| `seed` | `20260819` |
| 信頼水準 | 95%（パーセンタイル法） |

`seed` は固定する（`R-021`）。

### 6. レポート

`BaselineReport.to_markdown()` が出力する内容。

1. 生成日と、除外した封印G1の件数（`D-076`）
2. 母集団ごとの確率指標の表
3. 母集団 × 戦略 × 券種の回収率の表。**信頼区間を必ず併記する（`R-025`）**
4. **`g1` 母集団の回収率には検出力に関する注記を必ず添える。** 対象レース数と `4/√(レース数)` を併記する（`D-008` / `D-071` / `Q-033`）
5. 年別内訳（`D-018` / `R-027`）

## 制約

- **予測対象レースのオッズを特徴量にしない（`D-002` / `R-018`）** という制約は本仕様に適用されない。本仕様は特徴量を作らず、**市場そのものを基準線として測る**モジュールである。`003-features.md` の `build_features` を呼ばない
- **学習をG1に絞らない（`D-003`）** の趣旨に沿い、`all` 母集団を第一の観測対象とする（`D-071`）
- **封印セットを参照しない（`D-017` / `D-076`）**
- **回収率を信頼区間なしに報告しない（`D-008` / `R-025`）**
- **年代別成績を併記する（`D-018` / `R-027`）**
- 返還（取消・除外）を購入したことにしない（`D-073` / `architecture.md` 6.4）
- 同一入力から同一出力が再現できること（`R-021`）。ブートストラップの `seed` を固定する

## テスト観点

| # | 入力 | 期待 |
|---|---|---|
| 1 | オッズ `[2.0, 4.0, 4.0]` の3頭立て | `p = [0.5, 0.25, 0.25]`、和が `1.0` |
| 2 | 1着が同着2頭のレース | 両馬の `y = 0.5`、`Σy = 1`（`D-074`） |
| 3 | `status='出走取消'` の馬を含むレース | 正規化の分母に含まれない。購入対象にもならない（`D-073`） |
| 4 | `status='競走中止'` の馬を含むレース | 正規化の分母に**含まれる**。購入対象にもなり、的中しない |
| 5 | `popularity` と `odds_win` 順位が食い違うレース（同値オッズ） | `favorite` が `popularity=1` の馬を選ぶ（`D-077`） |
| 6 | 複勝の払戻が2行しかないレース（`n_entries>=8`） | 3着の馬を買っても的中しない。着順から3着払いを仮定しない（`D-072`） |
| 7 | `payouts` に該当行が無い購入 | 払戻0円。例外にしない |
| 8 | 封印期間内のG1 | どちらの母集団にも現れない。`n_sealed_g1_excluded` に計上される（`D-076`） |
| 9 | 封印期間内の非G1レース | **除外されない**（`D-017`: 封印はG1の評価にのみ適用） |
| 10 | 同じ `conn` と同じ `seed` で2回実行 | CIを含めてビット完全一致（`R-021`） |
| 11 | `uniform` × `単勝` を全レースで実行 | `roi` が `favorite` より低い値になる（実測: 2026-08-19 時点で `uniform≈0.705` / `favorite≈0.800`） |
| 12 | `favorite` × `単勝` を全レースで実行 | `roi` が概ね 0.75〜0.80（`architecture.md` P-2 の記述） |
| 13 | 1レースだけのブートストラップ | CI が縮退しても例外を出さない |
| 14 | `by_era` の各年の `n_races` の合計 | 母集団全体の `n_races` と一致する |

**観点12は実データでの検算である。** 控除率20%という外部から与えられた既知の値（`favorite` の理論的な期待回収率）とほぼ一致することで、`payouts` の結合と回収率計算の正しさを確認する。

**観点11は当初「`uniform` も控除率の理論値に近づく」と予想して書いたが、実装後に実データで検算した結果、誤りだと分かった（2026-08-19）。** 人気別の単勝回収率を独立に集計すると、1番人気0.800から18番人気0.541まで概ね単調に下がる（favorite-longshot bias。人気の低い馬ほど、票が集める人気に見合わない回収率になる）。`uniform` は不利な人気薄を多く含むぶん `favorite` より低くなるのが正しい挙動であり、`payouts` の結合ロジックが誤っているわけではない。両戦略の生SQLによる独立検算（`baseline.py` を経由しない集計）で一致することを確認済み。

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-033` | 取り込み済みデータが3年分しかない | `g1` 母集団が36レースにとどまり、回収率は参考値。本仕様は期間に依存しない形で書いてあり、データが増えれば再実行するだけでよい |
| `Q-018` | 複勝・ワイドの発走前オッズが無い | **本仕様をブロックしない**（`D-072` により回収率は `payouts` から計算できる）。ブロックするのは `P-4` のEV計算 |
| `Q-022` | B判定に必要な累積封印セット数 | 本仕様の範囲外。`010-backtest.md` の責務 |
