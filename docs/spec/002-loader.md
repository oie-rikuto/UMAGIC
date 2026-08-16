# 002 ローダー

| | |
|---|---|
| Phase | P-0 |
| 関連決定 | `D-005` `D-009` `D-014` `D-023` `D-026` `D-034` `D-035` `D-037` `D-038` `D-039` `D-043` `D-044` `D-045` |
| 関連要件 | `R-011` `R-012` `R-013` `R-015` `R-016` `R-017` `R-021` |
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
- 由来を保持するため、`fetch_log.page_kind` に `archive` / `shutuba` / `day_index` を記録する。`runners.source` は両方とも `netkeiba_jra` になり区別できない。**`day_index` も記録する**。記録しないと、その日のレースを1件も取り込めなかったこと自体が `fetch_incomplete`（`012-data-quality.md`）に現れない

### `day_index` からの列挙

**中央のセクションのみを対象とする。** 同じページに地方競馬が並ぶが、ID体系が異なりスコープ外（`D-025`）。中央セクションの終端は次の見出し（＝地方の開始）。

**障害競走は距離表記で除外する（`D-047`）。** `archive` を引く前に弾く。

| 距離表記 | 扱い |
|---|---|
| `芝1600m` / `ダ1400m` | 取り込む |
| `障2910m` | **除外する。`archive` を引かない** |

競馬場コードが `01`〜`10` 以外の行も除外する（地方競馬場）。

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
| `取` | `出走取消`（`D-048`） | `NULL` |
| 上記以外 | — | **行を棄却** |

`取` は `P-0` の3年分取り込み（2022〜2024年、137,575出走行）で220件確認した。`失格` の表記は未確認のまま（`Q-023`）。表に無いため現状では棄却される。3年間で1件も出現しなかった。

#### 馬体重欄 → `horse_weight` / `weight_diff`

| 表記 | `horse_weight` | `weight_diff` |
|---|---|---|
| `490(-2)` | `490` | `-2` |
| `490(0)` | `490` | `0` |
| `計不` | `NULL` | `NULL` |
| 空 | `NULL` | `NULL` |

#### 距離・馬場 → `races.surface` / `direction` / `distance`

レース見出しの先頭トークンから読む。**先頭に錨を打ち、部分一致で探さない。**

| 表記 | `surface` | `direction` | `distance` |
|---|---|---|---|
| `芝右2000m` | `芝` | `右` | `2000` |
| `芝左2400m` | `芝` | `左` | `2400` |
| `ダ左1600m` | `ダート` | `左` | `1600` |
| `芝直線1000m` | `芝` | `直線` | `1000` |
| `芝右 外1200m` | `芝` | `右` | `1200` |
| `芝右 外-内3200m` | `芝` | `右` | `3200` |
| `芝右 内2周3600m` | `芝` | `右` | `3600` |
| `障芝 ダート2910m` | **読まない** | — | — |

**方向の後ろにコース形状が入る。** 阪神・京都の3200mは外回りから内回りへ入るため `外-内` になる。中山3600mは1周が短く2周するため `内2周` になり、**周回数の数字が距離の直前に挟まる**。「距離の直前は数字以外」という前提はここで破れるため、形状の語を数え上げるのではなく**「数字+m」が最初に現れる位置**を探す。区切りの `/` を跨がせない（距離が無いとき天候側の数字を拾わないため）。

保存済み3,101ページ・距離トークン53種類で全数検証済み（2026-08-17）。読めないのは意図的に除外している障害の1件のみ。

**障害の表記は意図的に読まない（`D-025` / `D-047`）。** `day_index` の段階で除外しているが、すり抜けても距離が読めず `parse_error` になり、静かに混入しない。`障芝 ダート2910m` は `芝` も `ダ` も内側に含むため、**先頭に錨を打つことでのみ弾ける**。

#### その他

| 列 | 表記 | 変換 |
|---|---|---|
| 単勝 | `---` / 空 | `NULL` |
| 単勝 | `31.4` | `31.4` |
| タイム | `1:08.0` | `68.0`（秒） |
| タイム | 空 | `NULL` |
#### コーナー通過順 → `races.corner_nos` / `runners.corners`

**レース単位の「コーナー通過順位」テーブルが両方の列の元データになる。** 馬ごとの `通過` 列だけを見て決めない。

##### `races.corner_nos`（`D-043`）

**行見出しからコーナー番号を採る。** 行数ではなく見出しの数値を使う。

| 行見出し | `corner_nos` |
|---|---|
| `1コーナー` `2コーナー` `3コーナー` `4コーナー` | `[1, 2, 3, 4]` |
| `3コーナー` `4コーナー` | `[3, 4]` |
| テーブルが0行（直線競走） | `[]` |
| テーブルそのものが無い | `NULL` |

行見出しが `Nコーナー` の形に一致しない行があれば、**`corner_nos = NULL` としてレースを取り込み**、`rejected_rows` に `reason = 'corner_header_unparsed'` を記録する。番号を推測で補わない。

##### `runners.corners`

`n = len(corner_nos)` とする。`corners[i]` は `corner_nos[i]` のコーナーの通過順を表す。

| `corner_nos` | `status` | `通過` 列 | `corners` |
|---|---|---|---|
| `[1,2,3,4]` | `出走` / `降着` | `12-13-6-6` | `[12, 13, 6, 6]` |
| `[3,4]` | `出走` / `降着` | `3-2` | `[3, 2]` |
| `[]`（直線競走） | 任意 | `13`（1要素が入る） | **`[]`。通過列の値は採らない** |
| 非空 | `競走中止` / `失格` | 空 | `NULL`（`D-044`。**正常**） |
| 非空 | `出走取消` / `競走除外` | 空 | `NULL`（正常） |
| 非空 | `出走` / `降着` | 空 | `NULL`（取得できていない） |
| 非空 | 任意 | 要素数が `n` と一致しない | **`NULL`。行は取り込む**（`D-044`） |
| `NULL` | 任意 | 任意 | `NULL`。通過列の値は採らない |

**要素数不一致で行を棄却しない（`D-044`）。** `corners` のみ `NULL` として行は取り込み、`rejected_rows` に `reason = 'corners_length_mismatch'` と生値を記録する。要素数が途中で切れた値は実データで観測されていないため、棄却規則を置くと**観測されていない事象のために正常な行を落とす**側に倒れる。件数は `012-data-quality.md` の `rejected_rate` に現れる。

**直線競走でも `通過` 列は空にならず1要素が入る。** この値はコーナー通過順ではないため取り込まない。取り込むと `corners[0]` が1コーナーの通過順として、`corners[-1]` が4コーナーの通過順として読まれ、`F-101` と `F-501` の意味が反転する。

**`[]` と `NULL` を区別する（`R-015`）。** `[]` はコーナーが存在しない、`NULL` はこの馬の通過順が存在しないを表す。原因の切り分けは `001-schema.md` の対応表による。

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
| `robots.txt` | **取得するホストごとに**開始前に1回確認する。1つでも禁止していれば**中断する** |
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

**`fetch_log` は URL ごとに1行を保ち、再取得時は上書きする（`D-045`）。** 試行履歴は残さない。

```sql
INSERT INTO fetch_log (url, source, page_kind, source_key, http_status, outcome, detail, fetched_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (url) DO UPDATE SET
    http_status = excluded.http_status,
    outcome     = excluded.outcome,
    detail      = excluded.detail,
    fetched_at  = excluded.fetched_at;
```

`INSERT` のまま再実行すると主キー違反で落ちる。行を増やす形にすると同じキャッシュからの2回目で行数が変わり、`R-021`（`fetched_at` を除いて完全一致）が成立しない。

`rejected_rows` は取り込みのたびに `source_key` 単位で**全削除してから挿入する**。追記のみにすると再実行で件数が倍になり、`rejected_rate` が実態から離れる。

### 失敗時の挙動（`R-015`）

| 事象 | 挙動 |
|---|---|
| HTTPエラー | `fetch_log.outcome = 'http_error'` を記録し、**次の `source_key` へ進む** |
| 空テンプレート（着順テーブルが無い） | `outcome = 'empty'` を記録し、次へ進む |
| パース例外 | `outcome = 'parse_error'` を記録し、次へ進む |
| **`day_index` の取得・パース失敗** | `page_kind='day_index'` で `outcome` を記録し、**次の日へ進む**。1日分の失敗で範囲全体を止めない |
| **`day_index` に中央開催が無い** | `outcome = 'empty'` を記録する。**異常ではない**（平日は大半がこれ） |
| **書き込み時の制約違反** | `outcome = 'parse_error'` を記録し、次へ進む。パースが不完全だったことの現れとして扱う |
| 未知の着順マーカー | その**行のみ**棄却して `rejected_rows` に記録。レースの他の行は取り込む |
| コーナー表の見出しが解釈できない | `corner_nos = NULL` でレースを取り込み、`rejected_rows` に記録。**行は捨てない**（`D-043`） |
| 通過列の要素数が `corner_nos` と不一致 | `corners = NULL` で行を取り込み、`rejected_rows` に記録（`D-044`） |
| `robots.txt` が取得を禁止 | **全体を中断する**（`D-014` 条件3） |

`robots.txt` 以外で全体を停止しない。**`db.netkeiba.com` と `race.netkeiba.com` の両方を確認する。** `D-037` により本番の主ソースは `db.netkeiba.com` であり、こちらを確認しない実装は `D-014` 条件3を満たさない。

**「次へ進む」は日次インデックスにも適用する。** 20年分は数万ページ規模（`Q-001` 備考）で、`D-014` 条件2 の5秒間隔では実行が数十時間に及ぶ。この間に一過性のHTTPエラーが一度も起きない前提は成り立たないため、**1日分の失敗で範囲全体を落とさない**。

### 再開

長時間の取り込みは中断されうる。再開時は **`fetch_log` に `outcome = 'ok'` で残っているレースを再取得しない**。`ok` 以外（`empty` / `http_error` / `parse_error`）は取り直す。

書き込みが途中で落ちたレースは行が欠けた状態で残りうるが、`fetch_log` が `parse_error` になるため再開の対象に含まれ、次回の実行で取り直される。

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
| 3b | 着順欄 `取` | `status='出走取消'`, `finish_pos IS NULL`（`D-048`） |
| 4 | 着順欄 `失` など表に無い値 | 行を棄却し `rejected_rows` に1件 |
| 5 | 馬体重 `計不` | `horse_weight IS NULL`, `weight_diff IS NULL` |
| 6 | 馬体重 `490(-2)` | `490`, `-2` |
| 7 | タイム `1:08.0` | `68.0` |
| 8 | 単勝 `---` | `NULL` |
| 9 | コーナー表0行 + 通過 `13` | `corner_nos = []`、`corners = []`（`[13]` にしない） |
| 10 | 見出し `1〜4コーナー` + 通過 空 + `status='出走'` | `corner_nos = [1,2,3,4]`、`corners IS NULL` |
| 10b | 見出し `1〜4コーナー` + 通過 `3-2` | `corners IS NULL` で**行は取り込む**。`rejected_rows` に `corners_length_mismatch` が1件 |
| 10c | 見出し `3コーナー` `4コーナー` + 通過 `3-2` | `corner_nos = [3,4]`、`corners = [3,2]` |
| 10d | 見出し `1〜4コーナー` + 通過 空 + `status='競走中止'` | `corners IS NULL`。`rejected_rows` は0件 |
| 10e | 見出しが `Nコーナー` に一致しない | `corner_nos IS NULL`、`corners IS NULL`。`rejected_rows` に `corner_header_unparsed` が1件 |
| 11 | 同じ `source_key` で `resolve` を2回 | 同じ内部IDが返り、`source_ids` は1行のまま |
| 12 | キャッシュ済みURLを再取得 | HTTPが発行されず `from_cache=True` |
| 13 | 同じ URL を2回 `fetch_log` に記録 | 主キー違反にならず1行のまま。`outcome` と `fetched_at` が2回目の値 |
| 14 | 同じキャッシュから2回取り込む（`R-021`） | 全テーブルの行数が一致し、`fetched_at` を除いて値が完全一致 |

### 実データによる受け入れケース

`tools/q011_feasibility/raw/` に取得済みのページを fixture に使う。

| `race_id` | 検証する性質 | 期待 |
|---|---|---|
| `201405010303` | **不良オッズ**（`D-037`） | `archive` を採用し、単勝オッズが人気と単調になる。`result` を採用した場合は単調性検査に落ちる |
| `202305021211` | 競走中止 | 18行、`status='競走中止'` が1件。`corner_nos=[1,2,3,4]`、中止馬は `corners IS NULL` で `rejected_rows` は0件 |
| `201905030611` | 競走除外・`計不` | 15行、`status='競走除外'` が2件、その2件は `horse_weight IS NULL` かつ `corners IS NULL`。`corner_nos=[3,4]` |
| `202007010811` | 降着 | `status='降着'` が1件、`finish_pos=4` |
| `202306040911` | 短距離のコーナー数 | `corner_nos=[3,4]`、完走馬の `corners` は全行2要素。**要素数から1角と読まない** |
| `200505050810` | 古いレース | `archive` のみで全項目が揃う。`shutuba` を引かない |
| `202304020211` | **直線競走**（アイビスサマーD、芝直線1000m） | コーナー表0行、`corner_nos=[]`、`corners=[]`。通過列の1要素を採らない。**ページ未取得**（`Q-026`） |

コーナー表の見出しと `corners` の要素数の対応は保存済みページで確認済み。**要素数だけではコーナー番号が決まらない**（`D-043`）。

| レース | 距離 | コーナー表の見出し | 通過列の要素数 |
|---|---|---|---|
| スプリンターズS2023 | 芝1200m | `3コーナー` `4コーナー` | 2 |
| 東京3R（2014) | ダ1600m | `3コーナー` `4コーナー` | 2 |
| ユニコーンS2019 | ダ1600m | `3コーナー` `4コーナー` | 2 |
| 日本ダービー2023 | 芝2400m | `1〜4コーナー` | 4 |
| ジャパンカップ2005 | 芝2400m | `1〜4コーナー` | 4 |

**完走しなかった馬の `通過` 列は空になる。** 日本ダービー2023 の競走中止馬（馬番17）はレース単位のコーナー表には4コーナーすべてに現れるが、馬ごとの `通過` 列は空。ユニコーンS2019 の競走除外2頭も同じ（`D-044`）。

**アイビスサマーD2023 の行だけは保存済みページで確認できていない**（`Q-026`）。直線競走は `[]` と `NULL` の区別が効く唯一のケースであり、ページを取得するまで確定として扱わない。

**`200505050810` と `200605050810` は `shutuba` / `result` が空テンプレートを返すことを確認済み**（両ページとも0頭、`archive` のみ18頭・11頭）。この2レースは `D-037` の「年代で分岐しない」設計の回帰テストになる。

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-023` | 失格の着順マーカーが未観測（出走取消は `D-048` で解決） | 対応表に1行足りず、該当行は棄却される。`rejected_rows` に溜まるので発生は検知できる |
| `Q-024` | JRDB合流時の名寄せ | `D-038` により `P-4` まで影響しない |
| `Q-020` | 棄却行の許容水準が未定 | `rejected_rows` の件数をどこで異常とみなすかが決まらない |
| `Q-026` | 直線競走（`202304020211`）のページが未取得 | 受け入れケース1件と単体テスト観点9が実データで確認できない。合成データでは検証できる |
| `Q-021` | 天気予報の取得元が未定 | `races.weather_forecast` を埋める経路がまだ無い |
| `Q-018` | 複勝・ワイドの過去オッズが取得できない | `odds` テーブルは単勝以外ほぼ埋まらない |
