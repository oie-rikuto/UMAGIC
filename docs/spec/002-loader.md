# 002 ローダー

| | |
|---|---|
| Phase | P-0 |
| 関連決定 | `D-005` `D-009` `D-014` `D-023` `D-026` `D-034` `D-035` `D-037` `D-038` `D-039` |
| 関連要件 | `R-011` `R-012` `R-013` `R-015` `R-016` `R-017` |
| 先行仕様 | `001-schema.md` |
| 状態 | Draft |

## 目的

外部ソースのページを取得し、`001-schema.md` の中間スキーマに変換して書き込む。ソース固有の知識をこの層に閉じ込め、以降の層がソースを意識しないようにする（`D-009`）。

## 入出力

- **入力**: 対象期間（日付範囲）。または個別の `source_key`
- **出力**: `001-schema.md` の各テーブルへの書き込み。加えて本仕様が定義する運用テーブル（`fetch_log` / `rejected_rows`）

## 仕様

### ページ選択規則（`D-037`）

| 対象 | 引くページ | URL |
|---|---|---|
| 日付から `race_id` を列挙 | `day_index` | `https://db.netkeiba.com/race/list/{YYYYMMDD}/` |
| 確定済みレース | `archive` | `https://db.netkeiba.com/race/{race_id}/` |
| 発走前のレース | `shutuba` | `https://race.netkeiba.com/race/shutuba.html?race_id={race_id}` |

- **確定済みレースに `race.netkeiba.com/race/result.html` を使わない。** 年代でも分岐しない
- `shutuba` から取り込んだ行は、レース確定後に `archive` の値で**上書きする**
- 由来を保持するため、`fetch_log.page_kind` に `archive` / `shutuba` / `day_index` を記録する。`runners.source` は両方とも `netkeiba_jra` になり区別できない

### インターフェース

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol

PageKind = Literal["day_index", "archive", "shutuba"]
Outcome  = Literal["ok", "empty", "http_error", "parse_error"]


@dataclass(frozen=True)
class RawPage:
    source:     str          # 'netkeiba_jra' | 'netkeiba_nar' | 'jrdb'
    page_kind:  PageKind
    source_key: str          # day_index なら YYYYMMDD、それ以外はレースのキー
    url:        str
    body:       bytes        # 生バイト列。復号前
    encoding:   str
    fetched_at: datetime
    from_cache: bool


@dataclass(frozen=True)
class RejectedRow:
    source_key: str
    row_ref:    str | None   # 馬番など。行を特定できない場合は None
    reason:     str
    raw:        str


@dataclass(frozen=True)
class ParsedRace:
    race:     dict                 # races 1行分
    runners:  list[dict]           # runners
    payouts:  list[dict]           # payouts
    odds:     list[dict]           # odds
    laps:     list[dict]           # laps
    rejected: list[RejectedRow]    # 棄却した行（R-013）


class Fetcher(Protocol):
    """取得の責務。レート制限・キャッシュ・robots.txt 確認を担う。"""

    def get(self, url: str, *, source: str, page_kind: PageKind,
            source_key: str) -> RawPage: ...


class Source(Protocol):
    """ソース固有の知識を閉じ込める。D-009 の差し替え点。"""

    name: str

    def list_race_keys(self, day: date) -> list[str]: ...

    def url_for(self, source_key: str, page_kind: PageKind) -> str: ...

    def parse(self, page: RawPage) -> ParsedRace: ...
```

`Fetcher` と `Source` を分けるのは、レート制限とキャッシュが全ソース共通であるのに対し、URL構成とパースがソース固有であることによる。

### 正規化の対応表

`archive` の表記を中間スキーマの値へ写す。**表にない値に遭遇した行は棄却する**（`D-034` / `R-013`）。

#### 着順欄 → `status` / `finish_pos`

| 表記 | `status` | `finish_pos` |
|---|---|---|
| `1` `2` … | `出走` | その数値 |
| `4(降)` のように数値＋`(降)` | `降着` | 括弧の前の数値 |
| `中` | `競走中止` | `NULL` |
| `除` | `競走除外` | `NULL` |
| 上記以外 | — | **行を棄却** |

**`出走取消` と `失格` の表記は未確認**（`Q-023`）。表に無いため現状では棄却される。

#### 馬体重欄 → `horse_weight` / `weight_diff`

| 表記 | `horse_weight` | `weight_diff` |
|---|---|---|
| `490(-2)` | `490` | `-2` |
| `490(0)` | `490` | `0` |
| `計不` | `NULL` | `NULL` |
| 空 | `NULL` | `NULL` |

#### その他

| 列 | 表記 | 変換 |
|---|---|---|
| 単勝 | `---` / 空 | `NULL` |
| 単勝 | `31.4` | `31.4` |
| タイム | `1:08.0` | `68.0`（秒） |
| タイム | 空 | `NULL` |
#### コーナー通過順 → `runners.corners`

**レース単位の「コーナー通過順位」テーブルの行数を、そのレースのコーナー数とする。** 馬ごとの `通過` 列だけを見て決めない。

| コーナー数 | `通過` 列 | `corners` |
|---|---|---|
| 4 | `12-13-6-6` | `[12, 13, 6, 6]` |
| 2 | `3-2` | `[3, 2]` |
| **0（直線競走）** | `13`（1要素が入る） | **`[]`。通過列の値は採らない** |
| >0 | 空 | `NULL`（取得できていない） |
| >0 | 要素数がコーナー数と一致しない | **行を棄却** |

**直線競走でも `通過` 列は空にならず1要素が入る。** この値はコーナー通過順ではないため取り込まない。取り込むと `corners[0]` が1コーナーの通過順として、`corners[-1]` が4コーナーの通過順として読まれ、`F-101` と `F-501` の意味が反転する。

**`[]` と `NULL` を区別する（`R-015`）。** `[]` はコーナーが存在しない、`NULL` は取得できていないを表す。

### 同定（`D-035` / `D-038`）

```python
def resolve(conn, entity_type: str, source: str, source_key: str) -> int:
    row = conn.execute(
        "SELECT internal_id FROM source_ids "
        "WHERE entity_type = ? AND source = ? AND source_key = ?",
        [entity_type, source, source_key],
    ).fetchone()
    if row is not None:
        return row[0]

    new_id = next_internal_id(conn, entity_type)
    conn.execute(
        "INSERT INTO source_ids VALUES (?, ?, ?, ?, now())",
        [entity_type, new_id, source, source_key],
    )
    return new_id
```

**名寄せを行わない。** 馬名・生年による照合はしない。同じ実体が別ソースで別の内部IDを持つ状態を許容する（`Q-024`）。

### 取得（`D-014` / `D-039` / `R-016`）

| 項目 | 仕様 |
|---|---|
| `robots.txt` | 取得開始前に1回確認する。禁止されていれば**中断する** |
| 最小間隔 | 既定5.0秒。設定で**伸ばす方向にのみ**変更できる |
| キャッシュキー | URL |
| キャッシュ本体 | 生バイト列を gzip 圧縮して保存。**期限を設けない** |
| キャッシュヒット時 | HTTPリクエストを発行しない。`RawPage.from_cache = True` |
| User-Agent | 連絡先を含む固定文字列 |

**キャッシュはリポジトリにコミットしない**（`R-017` / `D-014` 条件1）。

### 運用テーブル

中間スキーマ（`001-schema.md`）は `D-009` によりソース非依存でなければならないため、`url` や `page_kind` を持つ以下の表はこちらで定義する。

```sql
CREATE TABLE fetch_log (
    url          VARCHAR   PRIMARY KEY,
    source       VARCHAR   NOT NULL,
    page_kind    VARCHAR   NOT NULL,
    source_key   VARCHAR   NOT NULL,
    http_status  INTEGER,
    outcome      VARCHAR   NOT NULL,
    detail       VARCHAR,
    fetched_at   TIMESTAMP NOT NULL,
    CHECK (outcome IN ('ok', 'empty', 'http_error', 'parse_error')),
    CHECK (page_kind IN ('day_index', 'archive', 'shutuba'))
);

CREATE TABLE rejected_rows (
    source       VARCHAR   NOT NULL,
    source_key   VARCHAR   NOT NULL,
    row_ref      VARCHAR,
    reason       VARCHAR   NOT NULL,
    raw          VARCHAR,
    fetched_at   TIMESTAMP NOT NULL
);
```

`fetch_log` が **「未取得」と「取得済みだが空」を区別する**（`R-015`）。行が無ければ未取得、`outcome = 'empty'` なら取得済みで中身が無い。

### 失敗時の挙動（`R-015`）

| 事象 | 挙動 |
|---|---|
| HTTPエラー | `fetch_log.outcome = 'http_error'` を記録し、**次の `source_key` へ進む** |
| 空テンプレート（着順テーブルが無い） | `outcome = 'empty'` を記録し、次へ進む |
| パース例外 | `outcome = 'parse_error'` を記録し、次へ進む |
| 未知の着順マーカー | その**行のみ**棄却して `rejected_rows` に記録。レースの他の行は取り込む |
| `robots.txt` が取得を禁止 | **全体を中断する**（`D-014` 条件3） |

`robots.txt` 以外で全体を停止しない。

## 制約

- **予測対象レースのオッズを特徴量にしない（`D-002` / `R-018`）。** ローダーは `odds_win` と `odds` を取り込むが、これらは市場確率ベースラインと期待値計算専用である
- **学習はG1に絞らない（`D-003`）。** 取得対象を `grade` で絞り込まない。対象はJRA平地全レースとJpnI（`D-025`）
- **同日レースの前後判定は `(date, course, race_number)`（`D-010`）。** `race_number` を `race_id` から導出せず、ページから取得した値を入れる
- **レート制限・キャッシュ・非再配布を実装から外さない**（`D-014` / `R-016` / `R-017`）
- 回収率は信頼区間なしに報告しない（`D-008`）。本仕様は `payouts` を欠損なく取り込むことのみ担保する

## テスト観点

### 単体

| # | 入力 | 期待 |
|---|---|---|
| 1 | 着順欄 `4(降)` | `status='降着'`, `finish_pos=4` |
| 2 | 着順欄 `中` | `status='競走中止'`, `finish_pos IS NULL` |
| 3 | 着順欄 `除` | `status='競走除外'`, `finish_pos IS NULL` |
| 4 | 着順欄 `取` など表に無い値 | 行を棄却し `rejected_rows` に1件 |
| 5 | 馬体重 `計不` | `horse_weight IS NULL`, `weight_diff IS NULL` |
| 6 | 馬体重 `490(-2)` | `490`, `-2` |
| 7 | タイム `1:08.0` | `68.0` |
| 8 | 単勝 `---` | `NULL` |
| 9 | コーナー数0 + 通過 `13` | `corners = []`（`[13]` にしない） |
| 10 | コーナー数4 + 通過 空 | `corners IS NULL` |
| 10b | コーナー数4 + 通過 `3-2` | 行を棄却（要素数不一致） |
| 11 | 同じ `source_key` で `resolve` を2回 | 同じ内部IDが返り、`source_ids` は1行のまま |
| 12 | キャッシュ済みURLを再取得 | HTTPが発行されず `from_cache=True` |

### 実データによる受け入れケース

`tools/q011_feasibility/raw/` に取得済みのページを fixture に使う。

| `race_id` | 検証する性質 | 期待 |
|---|---|---|
| `201405010303` | **不良オッズ**（`D-037`） | `archive` を採用し、単勝オッズが人気と単調になる。`result` を採用した場合は単調性検査に落ちる |
| `202305021211` | 競走中止 | 18行、`status='競走中止'` が1件 |
| `201905030611` | 競走除外・`計不` | 15行、`status='競走除外'` が2件、その2件は `horse_weight IS NULL` |
| `202007010811` | 降着 | `status='降着'` が1件、`finish_pos=4` |
| `202306040911` | 短距離のコーナー数 | コーナー表2行、`corners` の要素数が全行2 |
| `202304020211` | **直線競走**（アイビスサマーD、芝直線1000m） | コーナー表0行、`corners = []`。通過列の1要素を採らない |
| `200505050810` | 古いレース | `archive` のみで全項目が揃う。`shutuba` を引かない |

コーナー数と `corners` の要素数の対応は実データで確認済み。**直線競走だけが一致しない。**

| レース | 距離 | コーナー表の行数 | 通過列の要素数 |
|---|---|---|---|
| アイビスサマーD2023 | 芝直線1000m | **0** | **1** |
| スプリンターズS2023 | 芝1200m | 2 | 2 |
| 東京3R（2014) | ダ1600m | 2 | 2 |
| 日本ダービー2023 | 芝2400m | 4 | 4 |
| ジャパンカップ2005 | 芝2400m | 4 | 4 |

**`200505050810` と `200605050810` は `shutuba` / `result` が空テンプレートを返すことを確認済み**（両ページとも0頭、`archive` のみ18頭・11頭）。この2レースは `D-037` の「年代で分岐しない」設計の回帰テストになる。

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-023` | 出走取消・失格の着順マーカーが未観測 | 対応表に2行足りず、該当行は棄却される。`rejected_rows` に溜まるので発生は検知できる |
| `Q-024` | JRDB合流時の名寄せ | `D-038` により `P-4` まで影響しない |
| `Q-020` | 棄却行の許容水準が未定 | `rejected_rows` の件数をどこで異常とみなすかが決まらない |
| `Q-021` | 天気予報の取得元が未定 | `races.weather_forecast` を埋める経路がまだ無い |
| `Q-018` | 複勝・ワイドの過去オッズが取得できない | `odds` テーブルは単勝以外ほぼ埋まらない |
