# 004 リーク検査テスト

| | |
|---|---|
| Phase | P-1 |
| 関連決定 | `D-002` `D-007` `D-010` `D-017` `D-024` `D-028` `D-053` `D-054` `D-055` `D-056` |
| 関連特徴量 | `F-101` `F-102` `F-104` `F-201` `F-301` `F-501` `F-502` `F-603` `F-701` `F-703` `F-902` |
| 関連要件 | `R-018` `R-019` `R-020` `R-028` |
| 先行仕様 | `001-schema.md` `003-features.md` |
| 状態 | Draft |

## 目的

`domain-knowledge.md` 5節のリーク防止7原則と、封印セットの不参照を、実行可能なテストとして定義する。**モデルの精度で気づけない種類の誤りを機械的に落とす**ことがこの仕様の役割であり、テストが通ることが `P-1` の完了条件になる（`architecture.md` 7節）。

## 入出力

- **入力**: `003-features.md` が定義する特徴量生成の入口、および検査用のデータベース
- **出力**: `pytest` の合否

| ファイル | データ | 実行 |
|---|---|---|
| `tests/test_leakage.py` | 合成 fixture | CIで常時（`R-020` / `D-053`） |
| `tests/test_leakage_realdata.py` | `data/umagic.duckdb` | `pytest -m realdata`。手動（`D-053`） |

## 仕様

### `003-features.md` に要求するインターフェース

**この節は検査可能性のための契約である（`D-054` / `D-055`）。** 満たさない実装では以下のテストが書けない。

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

import duckdb
import polars as pl

# D-028 の確定時刻。当日のものは発走からの相対分を持つ
Timing = Literal["水曜", "木曜", "当日"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str                      # 'F-101' など
    timing: Timing
    minutes_before_post: int | None  # timing='当日' のときのみ。未確認は None（Q-019）


class FeatureRegistry(Protocol):
    """全 F-xxx とその確定時刻。R-028 の検査対象。"""

    def all(self) -> list[FeatureSpec]: ...

    def columns_for(self, route: Literal["暫定", "本命"]) -> list[str]: ...


def build_features(
    conn: duckdb.DuckDBPyConnection,
    *,
    as_of: date,
    race_ids: list[int] | None = None,
) -> pl.DataFrame: ...
```

`build_features` の契約:

| 項目 | 要求 |
|---|---|
| `as_of` | **`as_of` 以降のデータを一切参照しない。** 集計統計量の推定期間も含む（`R-019`） |
| 戻り値 | `race_id` `horse_id` と特徴量列。`(race_id, horse_id)` で一意 |
| 決定性 | 同じ入力・同じ `as_of` なら**ビット完全一致**する（`D-055` / `R-021`） |
| 集計順序 | 集約前に一意キーで整列する。加算順が実行ごとに変わらない（`D-055`） |

### 検査一覧

| # | 対応する原則 | `test_` 関数 | 何を見るか |
|---|---|---|---|
| 1 | 原則1 | `test_no_future_columns` | 特徴量列が発走後に確定する列に由来しない |
| 2 | 原則2 | `test_past_aggregation_is_strict` | 過去成績の集計が `race_date < target_race_date` |
| 3 | 原則3 | `test_same_day_uses_strict_race_number` | 同日利用が `F-501`/`F-502` に限られ、判定が厳密不等号 |
| 4 | 原則4 | `test_no_future_form_of_same_horse` | 対象レース以降のその馬の成績を使わない |
| 5 | 原則5 | `test_target_race_outcome_excluded` | 対象レースのラップ・タイム・着順が入力にない |
| 6 | 原則6 | `test_target_race_odds_excluded` | 対象レースのオッズ・人気が入力にない |
| 7 | 原則7 | `test_as_of_recomputation_invariance` | **as-of を変えた二重実行で過去分が一致**（中核） |
| 8 | `R-028` | `test_route_respects_deadline` | 経路の締切より後に確定する特徴量が入らない |
| 9 | `D-017` | `test_sealed_set_not_read` | 封印セットのレースを読まない |

### 7. 原則7（中核・`D-054`）

```python
def test_as_of_recomputation_invariance(conn):
    D1, D2 = date(2023, 1, 1), date(2024, 1, 1)      # D1 < D2

    x1 = build_features(conn, as_of=D1)
    x2 = build_features(conn, as_of=D2)

    key = ["race_id", "horse_id"]
    overlap = x2.join(x1.select(key), on=key, how="semi")

    assert_frame_equal(
        x1.sort(key), overlap.sort(key), check_exact=True,
    )
```

**`check_exact=True` にする（`D-055`）。** 許容誤差を置かない。

対象は `build_features` が返す**全列**であり、`F-902` の `μ_global` や `F-201` の embedding を名指ししない。名指しすると推定器を足したときに漏れる。

**Stage 1 の出力 `F-102` もこの検査に含める（`D-054`）。** Stage 1 は `as_of` までのデータだけで学習する。

以下は**必ず落ちる**ことを別テストで確認する（検査が機能していることの検査）。

| 仕込む欠陥 | 期待 |
|---|---|
| `μ_global` を `as_of` で切らず全期間で計算する | `test_as_of_recomputation_invariance` が落ちる |
| Stage 1 を全期間で学習する | 同上 |

### 3. 原則3（同日レースの例外規定）

`F-501` `F-502` **のみ**が同日レースを参照できる（`D-010`）。

```sql
-- 許可される参照。厳密不等号かつ同一競馬場
SELECT ... FROM races prev
WHERE prev.date = :target_date
  AND prev.course = :target_course
  AND prev.race_number < :target_race_number
```

| 誤り | 何が起きるか |
|---|---|
| `<=` にする | 対象レース自身が入る |
| `course` を外す | 複数場開催で前後関係が壊れる（`D-010`） |
| 日付だけで引く | 後続レースが入る |

`F-501` `F-502` 以外の特徴量が同日のレースを参照していないことを併せて検査する。

### 8. `R-028`（締切）

```python
ROUTE_DEADLINE = {"暫定": "木曜", "本命": "当日 T-15"}
```

| 経路 | 使える `timing` |
|---|---|
| 暫定 | `水曜` `木曜` |
| 本命 | `水曜` `木曜` `当日` かつ `minutes_before_post >= 15` |

**`minutes_before_post` が `None`（`T-?`）の特徴量は、当日確定として扱い暫定経路から除外する。** 本命経路で締切を超えるかは判定できない（`Q-019`）。判定できないことをテストの通過として扱わないよう、該当する特徴量の一覧を出力する。

合成特徴量の確定時刻が入力のうち最も遅いものに一致することを併せて検査する（`D-028`）。

### 9. 封印セット（`D-056`）

封印セットは固定年ではなくローリング（`D-017`）。範囲は計算で求める。

```python
def sealed_range(today: date, n_years: int = 3) -> tuple[date, date]:
    """まだ開発に使っていない直近 n_years 年。D-017。"""
```

開発用の経路が、この範囲の `grade='G1'` のレースを読まないことを検査する。**学習データ（全レース）は封印対象ではない**ため、G1に限定する。

### 合成 fixture が持つべき性質

実データ固有の並びを再現する（`D-053`）。以下を含まない fixture では検査が素通りする。

| 性質 | 何を検出するため |
|---|---|
| 同日・同競馬場に連続するR番号 | 原則3の厳密不等号 |
| 同日・**複数競馬場**の開催 | `course` を外した誤り |
| 同一馬が複数レースに出走（時系列） | 原則2・原則4 |
| 少走馬（1〜2走）と多走馬の混在 | `F-902` の縮約が `as_of` で切れているか |
| 期間をまたぐレース（`D1` の前後） | 原則7 |

## 制約

- **予測対象レースのオッズを特徴量にしない（`D-002` / `R-018`）。** `F-701` `F-703` が使う**過去レースの**人気は対象外
- **学習はG1に絞らない（`D-003`）。** 封印セットの検査（#9）はG1のみを対象とし、学習データ全体を制限しない
- **同日レースの例外は `F-501` `F-502` に限る（`D-010`）。** 他の特徴量に広げない
- 対象レースの実測ラップは Stage 1 の目的変数としてのみ使う（`D-007`）
- 回収率は信頼区間なしに報告しない（`D-008`）。本仕様は回収率を扱わない

## テスト観点

「検査が機能していること」を確認する。**欠陥を仕込んで落ちることを見る。**

| # | 仕込む欠陥 | 落ちる検査 |
|---|---|---|
| 1 | 過去集計を `<=` にする | `test_past_aggregation_is_strict` |
| 2 | 同日判定から `course` を外す | `test_same_day_uses_strict_race_number` |
| 3 | 同日判定を `<=` にする | 同上 |
| 4 | `F-601` に同日以降の成績を混ぜる | `test_no_future_form_of_same_horse` |
| 5 | 対象レースの `time_sec` を特徴量に入れる | `test_target_race_outcome_excluded` |
| 6 | 対象レースの `odds_win` を特徴量に入れる | `test_target_race_odds_excluded` |
| 7 | `μ_global` を全期間で推定する | `test_as_of_recomputation_invariance` |
| 8 | Stage 1 を全期間で学習する | 同上 |
| 9 | 暫定経路に `F-603`（馬体重・当日確定）を入れる | `test_route_respects_deadline` |
| 10 | 封印セットのG1を学習データに含める | `test_sealed_set_not_read` |

**欠陥を仕込まない状態で全検査が通ること**も併せて確認する。

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-019` | `当日 T-?` の確定分が未確認 | 該当特徴量は本命経路の締切判定ができない。暫定経路からの除外のみ検査する |
| `Q-025` | 混合効果モデル・階層ベイズの実装方法 | `F-301` の推定が fold ごとに現実的な時間で終わるかが未検証。`D-054` の Stage 1 再学習がこれに加算される |
| `Q-028` | 合成 fixture の規模 | 検査が意味を持つ最小のレース数・馬数が未定 |
