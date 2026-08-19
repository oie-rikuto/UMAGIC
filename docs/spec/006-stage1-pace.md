# 006 Stage 1 レース質予測

| | |
|---|---|
| Phase | P-3 |
| 関連決定 | `D-003` `D-007` `D-012` `D-021` `D-024` `D-025` `D-028` `D-029` `D-042` `D-054` `D-058` `D-082` `D-086` `D-087` `D-088` `D-089` `D-090` `D-091` |
| 関連特徴量 | `F-101` `F-102` `F-104` `F-803` `F-804` |
| 関連要件 | `R-018` `R-019` `R-021` |
| 先行仕様 | `001-schema.md` `003-features.md` `014-training-pipeline.md` |
| 状態 | Draft |

## 目的

出走馬の構成からレースの想定ペース（`F-102`）を予測する。`D-007` の2段階アーキテクチャの前段で、**対象レースの実測ラップが発走後にしか判明しないという制約を回避するために存在する**。

出力の `F-102` は `003-features.md` の `F-104`（`= F-102 × F-103_z`）の入力になる。

## 入出力

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import duckdb
import polars as pl


class Stage1Model(Protocol):
    """Stage 1 のモデル実装（`D-088`）。複数の実装を比較できる。"""

    def fit(self, x: pl.DataFrame, y: pl.Series, *, sample_weight: pl.Series | None,
            seed: int) -> None: ...
    def predict(self, x: pl.DataFrame) -> pl.Series: ...
    def save(self, out_dir: Path, meta: dict) -> None: ...
    @classmethod
    def load(cls, model_dir: Path) -> tuple["Stage1Model", dict]: ...


def build_target(conn: duckdb.DuckDBPyConnection, race_ids: list[int]) -> pl.DataFrame: ...
    """列: race_id, f102_actual, n_laps。`laps` が無いレースは行を含めない（`D-091`）。"""


def build_inputs(
    conn: duckdb.DuckDBPyConnection, race_ids: list[int], *, as_of: date,
) -> pl.DataFrame: ...
    """列: race_id, 入力特徴量...（2節）。"""


def predict_f102(
    model: Stage1Model, conn: duckdb.DuckDBPyConnection, race_ids: list[int], *, as_of: date,
) -> pl.DataFrame: ...
    """列: race_id, f102。`003-features.md` の `F-104` に渡す形。"""
```

## 仕様

### 1. 目的変数（`D-087`）

```python
N        = そのレースの laps の本数            # = ceil(distance / 200)
上がり3F = lap[N-2] + lap[N-1] + lap[N]        # 常に600m
前半合計 = sum(lap[1] .. lap[N-3])
前半距離 = distance − 600

f102_actual = 上がり3F / 3 − 前半合計 / (前半距離 / 200)   # 単位: 秒/200m
```

**前半の除数は区間数ではなく実距離ベース（`前半距離/200`）。** 端数距離では1本目が100m（1150mのみ150m）と短く、区間数で割ると前半平均が低く出る（`D-087` の追記）。距離が200mの倍数のときは `前半距離/200 = N-3` となり一致する。

| 項目 | 規則 |
|---|---|
| 符号 | **大きいほどハイペース**（前半が速く上がりが遅い） |
| 単位 | 秒/200m |
| 正規化 | **しない。** 生値のまま出す（`D-087`） |
| `N < 4` のレース | 目的変数を作れない。学習から除外する |

**`laps` が1行も無いレースは学習から除外する（`D-091`）。** 推論は制限しない。除外件数をログに残し、`012-data-quality.md` の `laps_coverage` と照合できるようにする。

**端数距離の扱いは実測で確認済み（`D-087` の追記、`Q-035` は解決）。** 12,678レースページで、本数が `ceil(距離/200)`・端数が先頭・末尾3本が常に600mであることを確認した。

### 2. 入力

すべてレース単位（1レース1行）。

| 列 | 出所 | 確定時刻（`D-028`） |
|---|---|---|
| `f101_min` | `F-101` の最小値（`D-089`） | `木曜` |
| `f101_mean` | `F-101` の平均（`D-089`） | `木曜` |
| `f101_q25` | `F-101` の下位25%点（`D-089`） | `木曜` |
| `f101_n_missing` | `F-101` が欠損している頭数（`D-089`） | `木曜` |
| `n_starters` | `races.n_starters`（`D-012`） | `木曜` |
| `distance` | `races.distance` | `木曜` |
| `surface` | `races.surface` | `木曜` |
| `direction` | `races.direction` | `木曜` |
| `course` | `races.course` | `木曜` |
| `race_class` | `races.race_class`（`D-049`） | `木曜` |
| `weather` | `races.weather`（`F-804`） | **`当日`** |
| `track_condition` | `races.track_condition` を順序尺度に写す（`001-schema.md`） | **`当日`** |

**`F-101` は馬単位で計算してから集約する（`D-089`）。** `003-features.md` の `compute_f101` を `as_of` 付きで呼ぶ。

**対象レースの `laps` を入力にしてはならない（`D-007` / 原則5）。** 実測ラップは目的変数としてのみ使う。

**対象レースのオッズ・人気を入力にしてはならない（`D-002` / `R-018`）。**

### 3. 暫定経路（`D-090`）

暫定予測（木曜、`D-024`）でも**当日予測と同一のモデル**を使う。`track_condition` は欠損値として渡す。

| 経路 | `weather` | `track_condition` |
|---|---|---|
| 本命（当日） | `races.weather`（実測） | 実測 |
| 暫定（木曜） | `races.weather_forecast`（`D-029`。過去分は `NULL`、`Q-021`） | **欠損** |

暫定用の別モデルを学習しない。

### 4. 学習（`D-088` / `014-training-pipeline.md`）

| 項目 | 規則 |
|---|---|
| 学習単位 | レース（1レース1行） |
| 学習母集団 | `014-training-pipeline.md` の fold の学習期間。`D-003` によりG1に絞らない |
| `sample_weight` | `014-training-pipeline.md` の `sample_weights()`（`D-081`）。レース単位なのでそのまま使える |
| 乱数シード | fold の `seed`（`D-085`） |
| 既定のモデル | LightGBM `objective='regression'`（`D-088`） |

**Stage 1 も fold ごとに学習し直す（`D-054`）。** 全期間で一度だけ学習しない。

### 5. Stage 2 へ渡すときのクロスフィッティング（`D-086`）

```python
blocks = cross_fit_blocks(fold, n_blocks=4)      # 014-training-pipeline.md

for block in blocks:
    rest = fold の学習期間のうち block を除いた部分
    m = Stage1Model(); m.fit(build_inputs(rest), build_target(rest), seed=fold.seed)
    f102[block] = m.predict(build_inputs(block))

m_full = Stage1Model(); m_full.fit(build_inputs(学習期間全体), ..., seed=fold.seed)
f102[検証期間] = m_full.predict(build_inputs(検証期間))
```

Stage 1 の学習回数は fold あたり `n_blocks + 1` 回になる。

### 6. モデルの保存

LightGBM 実装は `D-082` の形式（`model.txt` ＋ `meta.json`）に従う。他の実装は自前の直列化方式を定義してよいが、`meta.json` は共通形式を守る（`D-088`）。

`meta.json` に Stage 1 固有の項目を加える。

```json
{
  "stage": 1,
  "model_kind": "lightgbm_regression",
  "n_excluded_no_laps": 42,
  "...": "014-training-pipeline.md の共通項目"
}
```

## 制約

- **対象レースの実測ラップを入力にしない（`D-007` / `domain-knowledge.md` 5節 原則5）。** 実測ラップは目的変数としてのみ使う。**これを破ると即リークになる**
- **予測対象レースのオッズを特徴量にしない（`D-002` / `R-018`）**
- **学習をG1に絞らない（`D-003`）。** Stage 1 の学習母集団は fold の学習期間の全レース
- **集計統計量の推定期間も `as_of` で切る（`R-019` / 原則7）。** Stage 1 の学習期間は fold の学習期間に閉じる（`D-054`）。全期間で一度だけ学習しない
- **`F-102` は `race_level=True`（`D-021`）。** レース内相対化（`F-901`）を適用しない
- **同日レースを参照しない。** Stage 1 の入力に `F-501` / `F-502` を含めない（`D-010` の例外は当該2特徴量に限る）
- **同一入力から同一出力が再現できる（`R-021`）。** fold の `seed` を使う
- 回収率を信頼区間なしに報告しない（`D-008` / `R-025`）。本仕様は回収率を扱わないが、`010-backtest.md` に渡す形を壊さない

## テスト観点

| # | 入力 | 期待 |
|---|---|---|
| 1 | 10本のラップ `[12.0]*10` | `f102_actual = 0.0`（前半平均も上がり平均も12.0） |
| 2 | 前半が速く上がりが遅いラップ | `f102_actual > 0`（ハイペース、`D-087` の符号） |
| 3 | 前半が遅く上がりが速いラップ | `f102_actual < 0`（スローペース） |
| 4 | 2400m（12本）と2500m（13本、1本目が100m） | どちらも「200mあたりの秒数」に揃う。**端数距離で前半が短く出ない**（`D-087` の追記） |
| 5 | `laps` が1行も無いレース | `build_target()` の戻り値に**行が現れない**（`D-091`） |
| 6 | `laps` が3本以下のレース | 同上（`N < 4` で前半が作れない） |
| 7 | `laps` が無いレースへの `predict_f102()` | 値が出る。例外にしない（`D-091`） |
| 8 | `F-101` が全馬欠損のレース | `f101_min` / `f101_mean` / `f101_q25` が `NaN`、`f101_n_missing = n_starters`（`D-089`） |
| 9 | 「逃げ馬1頭＋差し馬17頭」と「先行馬6頭」 | `f101_mean` が同値でも `f101_min` / `f101_q25` が異なる（`D-089` の動機） |
| 10 | `track_condition` を欠損にして `predict_f102()` | 値が出る。例外にしない（`D-090`） |
| 11 | 同じ fold・同じ `seed` で2回学習 | 予測がビット完全一致（`R-021`） |
| 12 | `build_inputs()` の列 | `laps` 由来の列を1つも含まない（`D-007` / 原則5） |
| 13 | `build_inputs()` の列 | `odds_win` / `popularity` を1つも含まない（`R-018`） |
| 14 | fold の学習期間より後のレース | `build_inputs(as_of=fold.valid_start)` の対象に含まれない（`R-019`） |
| 15 | `save()` → `load()` | `meta.json` の `n_excluded_no_laps` が保持される |

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-021` | 暫定予測時点の天候・馬場情報の取得元 | 暫定経路を過去データで完全に再現できない（`D-090`）。劣化幅は「`track_condition` を欠損にした推論」との比較で測る |
| `Q-033` | 取り込み済みデータが3年分（10年分に拡張中） | 3年分では `014-training-pipeline.md` の fold が0本になり学習できない |
| `Q-034` | JpnIが学習データに入っていない | Stage 1 の学習母集団からもJpnIが欠ける。`R-004` が満たされない |
| `Q-019` | `当日 T-?` の確定分が未確認 | `weather` / `track_condition` の `minutes_before_post` が `None`。本命経路の締切判定ができない |
