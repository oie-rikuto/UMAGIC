# 014 学習パイプライン

| | |
|---|---|
| Phase | P-3 |
| 関連決定 | `D-003` `D-007` `D-008` `D-017` `D-025` `D-027` `D-042` `D-054` `D-075` `D-079` `D-080` `D-081` `D-082` `D-083` `D-084` `D-085` `D-086` |
| 関連要件 | `R-004` `R-021` `R-022` |
| 先行仕様 | `001-schema.md` `003-features.md` `005-baseline.md` |
| 状態 | Draft |

## 目的

walk-forward の fold を生成し、`sample_weight` と乱数シードを与え、学習済みモデルを保存・読み込む。**モデルそのものは定義しない**（Stage 1 は `006-stage1-pace.md`、Stage 2 は `007-stage2-ranker.md`）。

`003-features.md` より後、`006` / `007` より前に固める。fold と `sample_weight` が決まっていないとモデルを評価できない。

## 入出力

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import duckdb
import polars as pl


@dataclass(frozen=True)
class Fold:
    """walk-forward の1 fold（`D-080`）。"""

    index: int                    # 0始まり
    train_start: date
    train_end: date               # 検証開始の前日（含む）
    valid_start: date
    valid_end: date               # 含む
    seed: int                     # グローバルシードからの派生（D-085）

    @property
    def inner_valid_start(self) -> date: ...   # 学習期間末尾1年の開始（D-084）


@dataclass(frozen=True)
class TrainingSet:
    """1 fold ぶんの学習・検証データ。"""

    fold: Fold
    train: pl.DataFrame           # race_id, horse_id, 特徴量列..., sample_weight, group
    valid: pl.DataFrame           # 同上


def make_folds(
    conn: duckdb.DuckDBPyConnection,
    *,
    today: date,
    train_years: int | None = None,     # None で expanding、整数で sliding（D-080）
    min_train_years: int = 3,           # D-080
    sealed_years: int = 3,              # D-017
    seed: int = 20260819,               # D-085
) -> list[Fold]: ...


def sample_weights(
    conn: duckdb.DuckDBPyConnection,
    race_ids: list[int],
    *,
    class_weights: dict[str | None, float],
) -> pl.DataFrame: ...              # race_id, sample_weight（D-081）


def cross_fit_blocks(fold: Fold, *, n_blocks: int = 4) -> list[tuple[date, date]]: ...
    """Stage 1 のクロスフィッティング用に学習期間を時系列で分割する（`D-086`）。"""


def save_model(booster, meta: dict, out_dir: Path) -> None: ...   # D-082
def load_model(model_dir: Path) -> tuple[object, dict]: ...       # D-082
```

## 仕様

### 1. fold の生成（`D-080`）

#### 1.1 対象期間

```sql
-- 学習・検証の対象になりうるレース（D-079: 封印G1は学習からも除く）
SELECT race_id, date, grade
FROM races
WHERE NOT (grade = 'G1' AND date BETWEEN :sealed_start AND :sealed_end)
ORDER BY race_id
```

`sealed_start` / `sealed_end` は `umagic.sealed.sealed_range(today, sealed_years)`。**封印されるのはG1の行だけで、同じ期間の非G1レースは学習に使う（`D-079`）。**

#### 1.2 fold の境界

| 項目 | 規則 |
|---|---|
| 検証期間 | 1暦年（`1/1` 〜 `12/31`） |
| 最初の検証年 | `対象データの最初の年 + min_train_years` |
| 最後の検証年 | 対象データの最後の年 |
| `train_end` | `valid_start` の前日 |
| `train_start` | `train_years is None` なら対象データの最初の日、そうでなければ `valid_start - train_years年` |

**`train_end < valid_start` が常に成り立つ（`R-022`）。** 同日を両側に入れない。

#### 1.3 fold 別シード（`D-085`）

```python
rng = random.Random(seed)
for i in range(n_folds):
    fold_seed = rng.randrange(2**32)
```

`seed + i` としない。導出した `fold_seed` を LightGBM の `seed` / `bagging_seed` / `feature_fraction_seed` に渡す。

### 2. `sample_weight`（`D-081`）

```python
sample_weight_i = class_weights.get(races.grade_i, 1.0)
```

`class_weights` の既定値は本仕様では固定しない。`P-3` のハイパーパラメータ探索で決める（`D-051` の `k`、`D-059` の `N` と同じ扱い）。

| `races.grade` | 備考 |
|---|---|
| `G1` `G2` `G3` `L` | 取り込み済み3年分で観測済み |
| `NULL` | 非重賞。取り込み済みデータの94%（9,409/9,987） |
| JpnI相当 | **未取得（`Q-034`）。** 辞書に無いクラスは `1.0` にフォールバックする |

### 3. ネストしたハイパーパラメータ探索（`D-084`）

```
outer fold: [        学習期間         ][ 検証期間 ]
            [ inner学習    ][ inner検証 ]
                            └ 末尾1年
```

| 項目 | 規則 |
|---|---|
| `inner_valid_start` | `train_end - 1年 + 1日` |
| `inner_valid_end` | `train_end` |
| `inner_train` | `train_start` 〜 `inner_valid_start` の前日 |

ハイパーパラメータは inner 検証の成績で選ぶ。**outer 検証期間の成績を探索に使わない。**

`min_train_years=3` のとき inner 学習は2年になる。**`min_train_years < 3` は `ValueError` とする**（inner学習が1年以下になり本節の分割が成立しない）。

### 4. Stage 1 のクロスフィッティング（`D-086`）

Stage 2 の学習データに与える `F-102` を out-of-fold 予測で作る。

```python
# fold の学習期間を時系列で n_blocks 分割する（ランダム分割にしない）
blocks = cross_fit_blocks(fold, n_blocks=4)

for k, (block_start, block_end) in enumerate(blocks):
    rest = 学習期間のうち block k を除いた部分
    stage1_k = Stage1を rest で学習
    F-102[block k] = stage1_k.predict(block k)

# 検証期間ぶんは学習期間全体で学習したモデルから出す
stage1_full = Stage1を学習期間全体で学習
F-102[検証期間] = stage1_full.predict(検証期間)
```

| 項目 | 規則 |
|---|---|
| `n_blocks` の既定値 | `4` |
| 分割の向き | **時系列順**。ランダム分割にしない（`D-054` / 原則7） |
| ブロックの境界 | 学習期間を日数でほぼ等分し、レースの日をまたがない位置で切る |

Stage 1 の学習回数は fold あたり `n_blocks + 1` 回になる。

### 5. モデルの保存（`D-082`）

```
<out_dir>/
  model.txt     # Booster.save_model()
  meta.json
```

`meta.json` の内容。

```json
{
  "fold": {
    "index": 0,
    "train_start": "2015-01-01", "train_end": "2017-12-31",
    "valid_start": "2018-01-01", "valid_end": "2018-12-31",
    "seed": 1234567890
  },
  "feature_names": ["f101", "f101_unavailable", "..."],
  "class_weights": {"G1": 5.0, "G2": 3.0, "G3": 2.0, "L": 1.5},
  "git_commit": "<git rev-parse HEAD>",
  "uv_lock_sha256": "<uv.lock の SHA-256>",
  "lightgbm_params": {"...": "..."}
}
```

**`feature_names` は順序を保持して記録する。** 読み込み時に入力の列順と突き合わせ、一致しなければ `ValueError` とする。

### 6. walk-forward の実行と集計（`D-083`）

```python
def run_walk_forward(...) -> pl.DataFrame:
    """全 fold の検証予測を縦に積んだ DataFrame を返す。

    列: race_id, horse_id, fold_index, y_true, y_pred
    """
```

**指標は計算しない。** 全 fold の予測を1つの集合に積んで返し、LogLoss / Brier / 回収率の計算は `010-backtest.md` が行う（`D-083`）。fold ごとに指標を出して平均しない。

`005-baseline.md` の市場確率と比較するときは、**同じレース集合に対して `005` の指標を計算し直す**（`D-075`）。

## 制約

- **検証は walk-forward のみ（`D-008` / `R-022`）。** ランダムCVを行うコードを置かない。`train_end < valid_start` をテストで示す
- **学習をG1に絞らない（`D-003`）。** fold の対象は全レース。G1特化はクラス特徴量・`sample_weight`・校正の3手段で行う
- **封印セットを学習・検証の両方から除外する（`D-017` / `D-079`）**
- **集計統計量の推定期間も `as_of` で切る（`R-019` / 原則7）。** `build_features(conn, as_of=fold.valid_start)` を fold ごとに呼ぶ。Stage 1 の学習期間も fold の学習期間に閉じる（`D-054`）
- **予測対象レースのオッズを特徴量にしない（`D-002` / `R-018`）。** 本仕様は特徴量を作らないが、`003-features.md` の出力をそのまま渡す
- **同一入力から同一出力が再現できる（`R-021`）。** 乱数シードを固定し、`meta.json` に記録する
- 回収率を信頼区間なしに報告しない（`D-008` / `R-025`）。本仕様は指標を計算しないため直接の対象外だが、`010-backtest.md` に渡す形を壊さない

## テスト観点

| # | 入力 | 期待 |
|---|---|---|
| 1 | 2015-2024のデータ、既定値 | fold数が6（検証2018〜2023）。各 fold で `train_end < valid_start` |
| 2 | `train_years=3` | 各 fold の `train_start` が `valid_start` の3年前。expanding にならない |
| 3 | `train_years=None` | 全 fold の `train_start` が同じ（対象データの最初の日） |
| 4 | 封印期間内のG1 | どの fold の学習・検証にも現れない（`D-079`） |
| 5 | 封印期間内の**非G1**レース | 学習に**現れる**（`D-079`。封印されるのはG1の行だけ） |
| 6 | 同じ `seed` で2回 `make_folds()` | 全 fold の `seed` が一致（`R-021`） |
| 7 | `seed` を1つ変える | fold の `seed` が全て変わる。`seed+i` のような連番にならない（`D-085`） |
| 8 | `min_train_years=2` | `ValueError`（inner学習が1年以下になり `D-084` が成立しない） |
| 9 | `class_weights={"G1": 5.0}` | G1のレースが5.0、G2/G3/L/NULLが1.0（`D-081`） |
| 10 | `class_weights` に無いクラス | `1.0` にフォールバックする。例外にしない |
| 11 | `cross_fit_blocks(fold, n_blocks=4)` | 4ブロックが時系列順に並び、重複せず学習期間を覆う（`D-086`） |
| 12 | `save_model()` → `load_model()` | `meta.json` の `feature_names` が順序込みで一致する |
| 13 | 保存時と違う列順で `load_model()` の結果を使う | `ValueError`（`D-082`） |
| 14 | `run_walk_forward()` の戻り値 | `fold_index` が全 fold ぶん含まれ、`(race_id, horse_id)` が fold をまたいで重複しない |

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-034` | JpnIが学習データに入っていない | `class_weights` にJpnIのキーが無い。`R-004` が満たされない。fold生成自体はブロックされない |
| `Q-033` | 取り込み済みデータが3年分（10年分に拡張中） | 3年分では `min_train_years=3` で fold が0本になる。10年分の取得完了を待つ |
| `Q-005` | 学習データの遡及年数 | `train_years` を探索対象にすることで `P-3` で解決できる（`D-080`） |
| `Q-025` | 混合効果モデル（`F-301`）の実装方法 | `F-302` が `NaN` のまま学習する。fold生成には影響しない |
| `Q-007` | Stage 1 の目的変数の具体形 | `006-stage1-pace.md` の責務。`D-086` のクロスフィッティングは目的変数の形に依存しない |
