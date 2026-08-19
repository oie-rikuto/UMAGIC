# 015 確率校正

| | |
|---|---|
| Phase | P-3 |
| 関連決定 | `D-001` `D-003` `D-017` `D-054` `D-071` `D-079` `D-082` `D-084` `D-086` `D-095` `D-097` `D-098` `D-099` |
| 関連要件 | `R-001` `R-002` `R-021` `R-023` |
| 先行仕様 | `007-stage2-ranker.md` `014-training-pipeline.md` |
| 状態 | Draft |

## 目的

`007-stage2-ranker.md` が出す未校正のレース内 softmax を、G1での確率として正しい尺度に直す。`D-003` のG1特化3手段のうち3番目。

**順位は変えない。** 変えるのは確率の尺度だけ。

## 入出力

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import polars as pl

T_MIN = 0.1
T_MAX = 10.0
T_NO_CALIBRATION = 1.0


@dataclass(frozen=True)
class Calibrator:
    """温度スケーリング（`D-097`）。fold ごとに1つ持つ（`D-098`）。"""

    temperature: float
    n_races_fit: int          # 推定に使ったG1レース数
    n_runners_fit: int
    logloss_before: float     # T=1.0 のときの LogLoss
    logloss_after: float      # 推定した T での LogLoss
    at_bound: bool            # T が [T_MIN, T_MAX] の境界に張り付いたか

    def apply(self, scores: pl.DataFrame) -> pl.DataFrame: ...
        """列 `race_id, horse_id, score` を受け取り `win_prob` を付けて返す。"""


def fit_calibrator(oof: pl.DataFrame) -> Calibrator: ...
    """`oof` は列 `race_id, horse_id, score, is_winner` を持つ。
    G1のみに絞った out-of-fold 予測（`D-098`）。"""


def softmax_by_race(scores: pl.DataFrame, *, temperature: float) -> pl.DataFrame: ...
```

## 仕様

### 1. 校正データの作成（`D-098`）

```python
blocks = cross_fit_blocks(fold, n_blocks=4)      # 014-training-pipeline.md

for block in blocks:
    rest = fold の学習期間のうち block を除いた部分
    m = Stage2を rest で学習                      # 007-stage2-ranker.md
    score[block] = m.predict(block)

oof = score を race_id で絞り込み（races.grade = 'G1' のみ）
```

| 項目 | 規則 |
|---|---|
| 母集団 | `races.grade = 'G1'` のレースのみ（`D-098`） |
| 予測値 | クロスフィッティングの out-of-fold（`D-098`） |
| 分割 | `014-training-pipeline.md` の `cross_fit_blocks()` を流用（`D-086` と同じ） |
| 推定の単位 | **fold ごと**（`D-054` / 原則7） |
| 対象行 | `007-stage2-ranker.md` 1節の推論対象と同じ（`出走取消` / `競走除外` を除く） |
| 封印セット | 含まれない（`D-079` により fold の学習期間から既に除かれている） |

`is_winner` は `finish_pos = 1` かどうか。**1着同着は両方 `True` になる。**

### 2. 温度の推定（`D-097`）

```python
# レース r の馬 i について
p_i(T) = exp(score_i / T) / Σ_{j ∈ r} exp(score_j / T)

# 1着同着は正解ラベルを等分する（D-074 と同じ扱い）
y_i = 1 / |{j ∈ r : finish_pos_j = 1}|   （finish_pos_i = 1 のとき）
y_i = 0                                   （それ以外）

LogLoss(T) = −(1/N) Σ_r Σ_{i ∈ r} y_i · log(p_i(T))     # N はレース数

T* = argmin_{T ∈ [T_MIN, T_MAX]} LogLoss(T)
```

| 項目 | 値 |
|---|---|
| `T_MIN` | `0.1` |
| `T_MAX` | `10.0` |
| 最適化 | 1次元・有界。**決定的な手法に限る**（黄金分割探索など）。反復回数と収束判定を固定し、`R-021` を満たすこと |
| 最小化する量 | **LogLoss**（`R-023` の判定指標と一致させる） |

`scipy` は依存に含まれていない（`D-042`）。**新たな依存を足さずに実装する。**

**`T` が境界に張り付いた場合は `at_bound = True` を立てる。** 例外にはしないが、校正が成立していない兆候として `meta.json` とレポートに残す。

**学習期間にG1が1件も無い fold では `T = T_NO_CALIBRATION`（`1.0`）にフォールバックする**（`D-098`）。`n_races_fit = 0` として記録する。

### 3. 適用範囲（`D-099`）

| 対象 | 校正 |
|---|---|
| `g1` 母集団の評価（`D-071`） | **適用する** |
| 実運用の予測（`R-001`＝JRA平地G1） | **適用する** |
| `all` 母集団の評価（`D-071`） | **適用しない。** 未校正であることをレポートに明示する |

### 4. 出力

```python
win_prob_i = exp(score_i / T) / Σ_{j ∈ race} exp(score_j / T)
```

| 項目 | 規則 |
|---|---|
| `R-002` の担保 | レース内の `win_prob` の合計が `1.0 ± 1e-6` |
| 順位 | **変わらない**（`T > 0` の単調変換） |
| `T = 1.0` のとき | `007-stage2-ranker.md` の出力と一致する |

### 5. 保存

`D-082` の `meta.json` に加える。

```json
{
  "calibration": {
    "temperature": 1.83,
    "n_races_fit": 192,
    "n_runners_fit": 2784,
    "logloss_before": 2.31,
    "logloss_after": 2.24,
    "at_bound": false
  }
}
```

## 制約

- **校正はG1データで行う（`D-003`-3 / `D-098`）。** 重賞全体や全レースに広げない
- **校正器も fold ごとに推定し直す（`D-054` / `R-019` / 原則7）。** 全期間で一度だけ当てない
- **封印セットを校正に使わない（`D-017` / `D-079`）。** fold の学習期間から既に除かれている
- **校正データは out-of-fold であること（`D-098`）。** in-sample の予測で温度を当てると `T ≈ 1` に収束し、**モデルは正常に動き LogLoss が改善しないだけなのでテストに出ない**
- **出走全頭に確率を出す（`R-002`）。** レース内の合計が `1.0 ± 1e-6`
- **順位を変えない。** `T > 0` の制約がこれを保証する
- **同一入力から同一出力が再現できる（`R-021`）。** 最適化は決定的な手法を使い、`T` を `meta.json` に記録する
- 予測対象レースのオッズを特徴量にしない（`D-002` / `R-018`）。本仕様はスコアのみを扱い特徴量に触れない
- 回収率を信頼区間なしに報告しない（`D-008` / `R-025`）。本仕様は回収率を扱わない

## テスト観点

| # | 入力 | 期待 |
|---|---|---|
| 1 | 任意の `T > 0` で `apply()` | `win_prob` のレース内合計が `1.0 ± 1e-6`（`R-002`） |
| 2 | `T = 1.0` | `007-stage2-ranker.md` の素の softmax と一致する |
| 3 | `T > 1.0` | 確率が均一方向に寄る（最大値が下がる） |
| 4 | `T < 1.0` | 確率が尖る方向に寄る（最大値が上がる） |
| 5 | 任意の `T > 0` | レース内の `win_prob` の順位が `score` の順位と一致する |
| 6 | 完全に校正済みのスコア（人工データ） | `T*` が `1.0` 付近に収束する |
| 7 | 過度に尖ったスコア（人工データ） | `T* > 1.0` になり `logloss_after < logloss_before` |
| 8 | 1着同着2頭を含むレース | 正解ラベルが各 `0.5`、`Σy = 1`（`D-074` と同じ） |
| 9 | G1が0件の学習期間 | `T = 1.0`、`n_races_fit = 0`。例外にしない（`D-098`） |
| 10 | 極端なスコア（境界に張り付く入力） | `at_bound = True` が立つ。例外にしない |
| 11 | 同じ `oof` で2回 `fit_calibrator()` | `T` がビット完全一致（`R-021`） |
| 12 | `oof` に非G1レースを混ぜて渡す | **呼び出し側でG1に絞る**契約であることをテストで固定する（`D-098`） |
| 13 | `all` 母集団の評価経路 | 校正が適用されていない（`D-099`） |

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-037` | 単一の `T` が頭数4〜18で妥当か未検証 | 少頭数G1の確率が系統的に歪む可能性。実装は単一 `T` で進め、必要なら頭数依存の形に差し替える |
| `Q-036` | 素の softmax の尺度が未検証 | 飽和していると温度で戻せる範囲を超える可能性がある。`logloss_before` / `logloss_after` の差で効果を確認する |
| `Q-033` | 取り込み済みデータが3年分（10年分に拡張中） | fold が0本になり校正器も作れない |
| `Q-022` | B判定に必要な累積標本数 | 本仕様の範囲外。`010-backtest.md` の責務 |
