# 007 Stage 2 適性照合ランカー

| | |
|---|---|
| Phase | P-3 |
| 関連決定 | `D-002` `D-003` `D-006` `D-007` `D-011` `D-021` `D-032` `D-034` `D-042` `D-054` `D-058` `D-062` `D-081` `D-082` `D-084` `D-085` `D-086` `D-092` `D-093` `D-094` `D-095` `D-096` |
| 関連特徴量 | `F-101`〜`F-104` `F-201`〜`F-203` `F-302` `F-303` `F-501`〜`F-503` `F-601`〜`F-603` `F-701`〜`F-704` `F-801`〜`F-804` `F-901`〜`F-903` |
| 関連要件 | `R-002` `R-003` `R-018` `R-019` `R-021` `R-023` |
| 先行仕様 | `003-features.md` `006-stage1-pace.md` `014-training-pipeline.md` |
| 状態 | Draft |

## 目的

`003-features.md` の全特徴量と Stage 1 の出力（`F-102`）を入力に、出走各馬のスコアを出し、レース内 softmax で未校正の勝率に変換する。`D-007` の2段階アーキテクチャの後段。

**確率の校正は行わない**（`015-calibration.md` の責務、`D-003`-3 / `D-095`）。

## 入出力

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import polars as pl


@dataclass(frozen=True)
class CategoryMapping:
    """低頻度カテゴリの丸め表（`D-092`）。fold の学習期間から作る。"""

    column: str                       # 'sire_id' など
    keep: frozenset[int]              # そのまま残すカテゴリ
    other_code: int                   # 「その他」に割り当てる値


def build_category_mappings(
    train: pl.DataFrame, *, min_count: int,
) -> dict[str, CategoryMapping]: ...


def apply_category_mappings(
    df: pl.DataFrame, mappings: dict[str, CategoryMapping],
) -> pl.DataFrame: ...


def build_labels(conn: duckdb.DuckDBPyConnection, race_ids: list[int]) -> pl.DataFrame: ...
    """列: race_id, horse_id, label（`D-093` / `D-094`）。"""


def fit_stage2(
    x: pl.DataFrame, label: pl.Series, group: pl.Series, *,
    sample_weight: pl.Series, seed: int, params: dict,
    inner_x: pl.DataFrame, inner_label: pl.Series, inner_group: pl.Series,
) -> tuple[object, dict]: ...
    """戻り値: (Booster, inner検証の指標)。early stopping は LogLoss で行う。"""


def predict_win_prob(
    booster, x: pl.DataFrame, race_id: pl.Series,
) -> pl.DataFrame: ...
    """列: race_id, horse_id, score, win_prob。`win_prob` はレース内 softmax（`D-095`）。"""
```

## 仕様

### 1. 学習母集団（`D-094`）

| `status`（`D-011`） | 学習 | 推論 |
|---|---|---|
| `出走` | 含む | 含む |
| `降着` | 含む（`finish_pos` は公式着順、`D-034`） | 含む |
| `競走中止` | **含む**（ラベル0） | 含む |
| `失格` | **含む**（ラベル0） | 含む |
| `出走取消` | **除外** | **除外** |
| `競走除外` | **除外** | **除外** |

**`D-003` によりG1に絞らない。** 学習母集団は `014-training-pipeline.md` の fold の学習期間の全レース。

### 2. ラベル（`D-093`）

| `finish_pos` | `label` |
|---|---|
| `1` | `3` |
| `2` | `2` |
| `3` | `1` |
| `4` 以上 | `0` |
| `NULL`（`競走中止` / `失格`） | `0` |

**同着は同じ `finish_pos` を持つため同じラベルになる。** 1着が2頭同着なら両方 `3`。

### 3. `group`

LightGBM の `group` は**レース単位**。`(race_id, horse_id)` で整列したうえで、`race_id` ごとの行数を渡す（`D-055` の決定的順序に従う）。

```python
df = df.sort(["race_id", "horse_id"])
group = df.group_by("race_id", maintain_order=True).len()["len"]
```

### 4. カテゴリ変数（`D-092`）

| 列 | 型 |
|---|---|
| `sire_id` / `damsire_id` / `jockey_id` / `trainer_id` | カテゴリ |

```python
# fold の学習期間だけで出現回数を数える（R-019 / 原則7）
counts = train.group_by(col).len()
keep = counts.filter(pl.col("len") >= min_count)[col]
# keep に無い値と、検証期間・推論時の未知の値は other_code に落とす
```

| 項目 | 規則 |
|---|---|
| `min_count` の既定値 | **固定しない。** `P-3` の探索で決める |
| 対応表の作成範囲 | **fold の学習期間のみ**（`R-019`） |
| 未知のカテゴリ | `other_code` に落とす。例外にしない |
| LightGBM への渡し方 | `categorical_feature` に列名を指定 |

**丸め後のカテゴリ集合は fold ごとに異なる。** モデルを fold をまたいで流用しない（`D-092`）。対応表は `meta.json` に記録する。

### 5. 入力特徴量

`003-features.md` の `build_features()` の出力全列と、`006-stage1-pace.md` の `F-102` を結合したもの。

| 群 | 備考 |
|---|---|
| `F-101` `F-103` | 生値・`_z`・`_rank` の3列（`003` 共通規約4） |
| `F-102` | Stage 1 の出力。学習側はクロスフィッティング由来（`D-086`） |
| `F-104` | `F-102 × F-103_z`（`race_level=True`。相対化しない、`D-021`） |
| `F-2xx` `F-3xx` `F-5xx` `F-6xx` `F-7xx` `F-8xx` | `003-features.md` のとおり |
| `<feature>_unavailable` | `D-058` の指示子。**落とさない**（`D-096`） |

**列の集合を fold 間で固定する。** 全行が欠損の列（`F-302` は `Q-025`、`F-203` は `Q-030`、`F-502` は `Q-031` により現状すべて `NaN`）も落とさずに渡す。fold ごとに列を落とす実装にしない（`D-082` の `feature_names` が fold 間で食い違う）。

**予測対象レースのオッズ・人気を入力にしない（`D-002` / `R-018`）。**

### 6. 学習

| 項目 | 規則 |
|---|---|
| 目的関数 | `lambdarank`（`D-042`） |
| `sample_weight` | `014-training-pipeline.md` の `sample_weights()`（`D-081`）。レース単位の重みを、そのレースの全出走行に配る |
| 乱数シード | fold の `seed`（`D-085`） |
| inner 検証 | 学習期間の末尾1年（`D-084`） |
| early stopping / 探索の選択指標 | **レース内 softmax 後の LogLoss** |
| 併記する指標 | `NDCG@3`。`meta.json` に残す |

**選択は LogLoss で行い、`NDCG@3` は記録のみ。** 両方を残すことで、順位性能と確率性能の乖離に気づける。

`R-023`（A判定）が LogLoss で判定されることと、探索の選択指標が一致する。

### 7. 出力（`D-095`）

```python
win_prob_i = exp(score_i) / Σ_{j ∈ race} exp(score_j)
```

| 項目 | 規則 |
|---|---|
| 温度パラメータ | **導入しない** |
| 正規化の範囲 | レース内（1節の推論対象馬） |
| `R-002` の担保 | レース内の `win_prob` の合計が `1.0 ± 1e-6` |

**これは未校正の確率である。** G1での校正は `015-calibration.md` が行う（`D-003`-3）。

**素の softmax の尺度は未検証（`Q-036`）。** 1頭に飽和する、または全馬が均一に潰れる可能性があり、実データでの確認前に妥当と断定しない。

### 8. モデルの保存

`D-082` の形式（`model.txt` ＋ `meta.json`）に従う。Stage 2 固有の項目を加える。

```json
{
  "stage": 2,
  "objective": "lambdarank",
  "category_mappings": {
    "sire_id": {"other_code": -1, "n_kept": 312, "min_count": 20}
  },
  "inner_logloss": 2.41,
  "inner_ndcg3": 0.58,
  "...": "014-training-pipeline.md の共通項目"
}
```

## 制約

- **予測対象レースのオッズを特徴量にしない（`D-002` / `R-018`）。** `F-701` / `F-703` が使う**過去レースの**人気のみが例外（`Q-006`）
- **学習をG1に絞らない（`D-003`）。** G1特化はクラス特徴量・`sample_weight`（`D-081`）・校正（`015`）の3手段
- **リーク防止の7原則（`domain-knowledge.md` 5節）。** 特に原則7: カテゴリの丸め表を fold の学習期間だけで作る（`D-092`）。`F-102` は学習側がクロスフィッティング由来であること（`D-086`）
- **集計統計量の推定期間も `as_of` で切る（`R-019`）。** `build_features(conn, as_of=fold.valid_start)` を fold ごとに呼ぶ
- **`race_level=True` の特徴量を相対化しない（`D-021`）。** `F-104` を積の後に相対化すると想定ペースの大きさが消える
- **出走全頭に確率を出す（`R-002` / `R-003`）。** 過去走が少ない馬を除外しない。確率0にしない
- **同一入力から同一出力が再現できる（`R-021`）。** fold の `seed` を使う
- 回収率を信頼区間なしに報告しない（`D-008` / `R-025`）。本仕様は回収率を扱わないが、`010-backtest.md` に渡す形を壊さない

## テスト観点

| # | 入力 | 期待 |
|---|---|---|
| 1 | 任意のレースの推論結果 | `win_prob` のレース内合計が `1.0 ± 1e-6`（`R-002`） |
| 2 | 全特徴量が `NaN` の馬を含むレース | その馬にも `win_prob` が付き、合計が `1.0` を保つ（`R-003`） |
| 3 | `finish_pos` が `1,2,3,4` の4頭 | `label` が `3,2,1,0`（`D-093`） |
| 4 | 1着同着2頭 | 両方の `label` が `3`（`D-093`） |
| 5 | `status='競走中止'` の馬 | `label=0` で学習に**含まれる**（`D-094`） |
| 6 | `status='出走取消'` の馬 | 学習・推論の**どちらにも現れない**（`D-094`） |
| 7 | `group` の合計 | 学習行数と一致する |
| 8 | 学習期間に1回だけ現れる `sire_id` | `min_count` 以上なら残り、未満なら `other_code`（`D-092`） |
| 9 | 検証期間にのみ現れる `sire_id` | `other_code` に落ちる。例外にしない（`D-092`） |
| 10 | カテゴリ丸め表の作成 | **検証期間の行を数えていない**（`R-019` / 原則7） |
| 11 | 全行が `NaN` の列（`F-302` など） | 落とされずに `feature_names` に残る（`D-082` の一貫性） |
| 12 | 同じ fold・同じ `seed` で2回学習 | 予測がビット完全一致（`R-021`） |
| 13 | `build_features()` の出力に `odds_win` / `popularity` | 含まれない（`R-018`） |
| 14 | `meta.json` | `inner_logloss` と `inner_ndcg3` の両方が記録される |
| 15 | `sample_weight` | 同じレースの全出走行が同じ値を持つ（`D-081`） |

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-036` | 素の softmax の尺度が未検証 | `D-095` の妥当性。飽和・均一化のいずれでも指標が悪くなるだけでテストに出ない |
| `Q-025` | `F-301` の実装方法 | `F-302` の列が全行 `NaN` のまま学習する（`D-060`） |
| `Q-030` | `F-203` の数式 | `F-203` の列が存在しない |
| `Q-031` | `F-502` の基準タイム | `F-502` の列が存在しない |
| `Q-032` | `F-601` の高強度指標 | `F-601` は着順・着差・上がり3F順位のみ |
| `Q-033` | 取り込み済みデータが3年分（10年分に拡張中） | fold が0本になり学習できない |
| `Q-034` | JpnIが学習データに入っていない | `R-004` が満たされない。ダート路線の過去走集計が虫食い |
| `Q-006` | 過去オッズの利用範囲 | `F-701` / `F-703` に限定という暫定方針のまま |
