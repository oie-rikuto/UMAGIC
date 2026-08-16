# 001 中間スキーマ

| | |
|---|---|
| Phase | P-0 |
| 関連決定 | `D-009` `D-010` `D-011` `D-012` `D-020` `D-026` `D-029` `D-034` `D-035` `D-036` `D-043` `D-044` `D-046` |
| 関連特徴量 | `F-101` `F-102` `F-501` `F-502` `F-801` `F-804` `F-901` |
| 関連要件 | `R-011` `R-012` `R-013` `R-014` `R-015` |
| 状態 | Draft |

## 目的

データ取得元に依存しない中間スキーマを定義する。ソース別ローダー（`002-loader.md`）はこの形に変換して書き込み、以降のすべての層はこの形だけを読む。

## 入出力

- **入力**: ソース別ローダーが生成した正規化済みレコード
- **出力**: DuckDB のデータベースファイル `data/umagic.duckdb`

## 仕様

### 共通規約

| 項目 | 規約 |
|---|---|
| 方言 | DuckDB（`D-036`） |
| 日付 | `DATE`。時刻は `TIME` または `TIMESTAMP` |
| 全テーブル共通列 | `source` `fetched_at` を持ち、いずれも `NOT NULL`（`D-026`）。**対象は本仕様の7テーブル**（`D-046`） |
| `source` の値 | `netkeiba_jra` / `netkeiba_nar` / `jrdb` |
| 主キー | 内部ID（連番）。ソース側のIDは `source_ids` で紐づける（`D-035`） |
| 可変長データ | LIST 型（`D-036`） |

**作成順**: `source_ids` → `horses` → `races` → `runners` → `payouts` → `odds` → `laps`。外部キーの依存による。

以下のDDLは DuckDB 1.4.5 で実行し、全テーブルが作成できること、および「テスト観点」の各制約が意図どおり違反を拒否することを確認済み。**`races.corner_nos`（`D-043`）を加えた形で再実行済み**（2026-08-16）。

### 識別子

```sql
CREATE TABLE source_ids (
    entity_type  VARCHAR   NOT NULL,   -- 'race' | 'horse' | 'jockey' | 'trainer'
    internal_id  BIGINT    NOT NULL,
    source       VARCHAR   NOT NULL,
    source_key   VARCHAR   NOT NULL,   -- ソース側のID。netkeiba のレースIDは12桁
    fetched_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (entity_type, source, source_key),
    CHECK (entity_type IN ('race', 'horse', 'jockey', 'trainer'))
);

CREATE UNIQUE INDEX ux_source_ids_internal
    ON source_ids (entity_type, internal_id, source);
```

`UNIQUE` を貼るのは、1つの内部IDが**同一ソース内で複数のキーを持たない**ことを保証するため。ソースが異なる行は並んでよい（合流時に netkeiba とJRDBの両方が並ぶ）。

### races

```sql
CREATE TABLE races (
    race_id           BIGINT    PRIMARY KEY,
    date              DATE      NOT NULL,
    course            VARCHAR   NOT NULL,
    race_number       SMALLINT  NOT NULL,
    post_time         TIME,
    distance          SMALLINT  NOT NULL,
    surface           VARCHAR   NOT NULL,
    direction         VARCHAR,
    grade             VARCHAR,
    track_condition   VARCHAR,
    weather           VARCHAR,
    weather_forecast  VARCHAR,
    n_entries         SMALLINT  NOT NULL,
    n_starters        SMALLINT  NOT NULL,
    prize             BIGINT,
    corner_nos        SMALLINT[],
    source            VARCHAR   NOT NULL,
    fetched_at        TIMESTAMP NOT NULL,
    UNIQUE (date, course, race_number),
    CHECK (n_entries >= n_starters),
    CHECK (n_starters >= 0),
    CHECK (race_number >= 1),
    CHECK (distance > 0),
    CHECK (surface IN ('芝', 'ダート', '障害')),
    CHECK (direction IS NULL OR direction IN ('右', '左', '直線')),
    CHECK (track_condition IS NULL
           OR track_condition IN ('良', '稍重', '重', '不良'))
);
```

| 列 | 内容 |
|---|---|
| `race_number` | R番号。`D-010` の前後判定キー。`race_id` から導出せず列として持つ |
| `post_time` | 発走時刻。`D-024` の締切（`T-15`）の基準 |
| `prize` | 1着賞金（円）。`F-803` がレースの格の代理変数として使う |
| `weather_forecast` | 予報値（`D-029`）。**実測 `weather` と混ぜない。過去分は `NULL`**（`Q-021`） |
| `corner_nos` | このレースで記録されたコーナーの番号を順に持つ（`D-043`）。直線競走は `[]`、取得できていなければ `NULL` |

`UNIQUE (date, course, race_number)` は `D-010` の判定キーが一意であることを保証する。

**`corner_nos` は `runners.corners` の添字の意味を与える（`D-043`）。** `runners.corners[i]` は `corner_nos[i]` のコーナーの通過順である。

| レース | 距離 | `corner_nos` |
|---|---|---|
| 日本ダービー2023 | 芝2400m | `[1, 2, 3, 4]` |
| スプリンターズS2023 | 芝1200m | `[3, 4]` |
| ユニコーンS2019 | ダ1600m | `[3, 4]` |
| アイビスサマーD2023 | 芝直線1000m | `[]` |

**要素数からコーナー番号を逆算しない。** 4角を超えるコースがあり、要素数と番号の対応は一意でない。1角の通過順を必要とする特徴量（`F-101`）は `corner_nos` に `1` を含むレースのみを対象とする。

**`track_condition` は順序尺度である。** 特徴量層で数値化する際の対応は以下に固定する。DDL では文字列のまま持つ。

| 値 | 順序 |
|---|---|
| `良` | 0 |
| `稍重` | 1 |
| `重` | 2 |
| `不良` | 3 |

### runners

```sql
CREATE TABLE runners (
    race_id         BIGINT    NOT NULL REFERENCES races(race_id),
    horse_id        BIGINT    NOT NULL REFERENCES horses(horse_id),
    number          SMALLINT  NOT NULL,
    frame           SMALLINT,
    jockey_id       BIGINT,
    trainer_id      BIGINT,
    weight_carried  DECIMAL(4,1),
    horse_weight    SMALLINT,
    weight_diff     SMALLINT,
    age             SMALLINT,
    sex             VARCHAR,
    odds_win        DECIMAL(7,1),
    popularity      SMALLINT,
    status          VARCHAR   NOT NULL,
    finish_pos      SMALLINT,
    margin          VARCHAR,
    time_sec        DECIMAL(6,1),
    last_3f         DECIMAL(4,1),
    corners         SMALLINT[],
    source          VARCHAR   NOT NULL,
    fetched_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (race_id, horse_id),
    UNIQUE (race_id, number),
    CHECK (status IN ('出走', '降着', '競走中止', '失格', '出走取消', '競走除外')),
    CHECK (number >= 1),
    CHECK (finish_pos IS NULL OR finish_pos >= 1)
);
```

**`finish_pos` に一意制約を張らない。** 同着では同一着順が複数行に現れる（`R-014`）。

| 列 | 内容 |
|---|---|
| `finish_pos` | **公式着順のみ**（`D-034`）。入線順は保持しない |
| `margin` | 着差。ソースの表記をそのまま保持する（`ハナ` `クビ` `アタマ` `3.1/2` `同着` 等）。数値化は `003-features.md` の責務 |
| `time_sec` | 走破タイム（秒）。ソースの `1:08.7` 形式を秒に変換して格納する |
| `corners` | コーナー通過順。**要素数はレースによって変わり、添字の意味は `races.corner_nos` が与える**（`D-043`）。コーナーのないコースは空配列 |

**`corners` の空配列と `NULL` を区別する（`R-015`）。** 空配列は「このレースにコーナーが存在しない」、`NULL` は「この馬の通過順が存在しない」を表す。

**`NULL` の原因は `corners` 単独では判定できない（`D-044`）。** `races.corner_nos` と `status` との組で読む。

| `races.corner_nos` | `status` | `corners` | 読み方 |
|---|---|---|---|
| `[]` | 任意 | `[]` | レースにコーナーが無い |
| 非空 | `出走` / `降着` | 要素数が `corner_nos` と一致 | 正常 |
| 非空 | `競走中止` / `失格` | `NULL` | **正常。** ソースが記録していない |
| 非空 | `出走取消` / `競走除外` | `NULL` | 正常。ゲートを出ていない |
| 非空 | `出走` / `降着` | `NULL` | **異常。** 取得できていない |

最終行だけが取得漏れであり、`012-data-quality.md` の `corners_missing` が検出する。

### `status` の定義

`D-011` の6状態。**2つの軸が独立している**ので混同しないこと。

| `status` | ゲートを出たか | 返還 | `n_starters` | `finish_pos` | `time_sec` | 着順欄の表記 |
|---|---|---|---|---|---|---|
| `出走` | 出た | なし | 数える | 必須 | 必須 | 数値 |
| `降着` | 出た | なし | 数える | 必須 | 必須 | 数値 + `(降)` |
| `競走中止` | 出た | なし | 数える | `NULL` | `NULL` | `中` |
| `失格` | 出た | なし | 数える | `NULL` | 未確定 | **未観測**（`Q-023`） |
| `出走取消` | 出ていない | あり | 数えない | `NULL` | `NULL` | `取`（`D-048`） |
| `競走除外` | 出ていない | あり | 数えない | `NULL` | `NULL` | `除` |

```
n_entries  = runners の全行数
n_starters = status IN ('出走','降着','競走中止','失格') の行数
```

**着順マーカーが上表にないものだった場合、その行を取り込まず除外ログに記録する（`D-034` / `R-013`）。** 取消と失格の表記は実データで観測できていないため、推測で追加しない。

### horses

```sql
CREATE TABLE horses (
    horse_id    BIGINT    PRIMARY KEY,
    name        VARCHAR   NOT NULL,
    birth       DATE,
    sire_id     BIGINT,
    dam_id      BIGINT,
    damsire_id  BIGINT,
    source      VARCHAR   NOT NULL,
    fetched_at  TIMESTAMP NOT NULL
);
```

`sire_id` / `dam_id` / `damsire_id` に外部キーを張らない。**種牡馬・繁殖牝馬はJRAで走っていないことがあり、`horses` に行が存在しない場合がある。**

### payouts / odds

```sql
CREATE TABLE payouts (
    race_id      BIGINT     NOT NULL REFERENCES races(race_id),
    bet_type     VARCHAR    NOT NULL,
    comb_key     VARCHAR    NOT NULL,
    combination  SMALLINT[] NOT NULL,
    payout       INTEGER    NOT NULL,
    popularity   SMALLINT,
    source       VARCHAR    NOT NULL,
    fetched_at   TIMESTAMP  NOT NULL,
    PRIMARY KEY (race_id, bet_type, comb_key),
    CHECK (payout >= 0)
);

CREATE TABLE odds (
    race_id      BIGINT       NOT NULL REFERENCES races(race_id),
    bet_type     VARCHAR      NOT NULL,
    comb_key     VARCHAR      NOT NULL,
    combination  SMALLINT[]   NOT NULL,
    odds_low     DECIMAL(8,1) NOT NULL,
    odds_high    DECIMAL(8,1) NOT NULL,
    as_of        TIMESTAMP    NOT NULL,
    source       VARCHAR      NOT NULL,
    fetched_at   TIMESTAMP    NOT NULL,
    PRIMARY KEY (race_id, bet_type, comb_key, as_of),
    CHECK (odds_high >= odds_low),
    CHECK (odds_low > 0)
);
```

**同着では同一券種に複数の払戻が出る。** `comb_key` が異なるため主キーは衝突しない。

`payout` は100円あたりの払戻額（円）。

#### `combination` の正規形（`D-036`）

| `bet_type` | 頭数 | 並び |
|---|---|---|
| `単勝` `複勝` | 1 | — |
| `枠連` `馬連` `ワイド` | 2 | 馬番の**昇順** |
| `馬単` | 2 | **着順** |
| `三連複` | 3 | 馬番の**昇順** |
| `三連単` | 3 | **着順** |

`comb_key` は `combination` の要素をハイフンで連結した文字列とする（例: `[5,9]` → `5-9`）。`枠連` は馬番ではなく枠番を格納する。

### laps

```sql
CREATE TABLE laps (
    race_id     BIGINT       NOT NULL REFERENCES races(race_id),
    furlong_no  SMALLINT     NOT NULL,
    lap_sec     DECIMAL(4,1) NOT NULL,
    source      VARCHAR      NOT NULL,
    fetched_at  TIMESTAMP    NOT NULL,
    PRIMARY KEY (race_id, furlong_no),
    CHECK (furlong_no >= 1),
    CHECK (lap_sec > 0)
);
```

1ハロンごとのラップタイム。`furlong_no` は先頭から1始まり。**`D-007` の Stage 1 の目的変数の元データ**であり、対象レースの値を特徴量に混入させてはならない（`R-018` の対象外だが `domain-knowledge.md` 5節 原則5に該当）。

## 制約

- **予測対象レースのオッズを特徴量にしない（`D-002` / `R-018`）。** `odds` と `runners.odds_win` は市場確率ベースラインと期待値計算専用であり、特徴量層から対象レースの行を参照しない
- **学習はG1に絞らない（`D-003`）。** このスキーマに `grade` によるフィルタを埋め込まない
- **同日レースの前後判定は `(date, course, race_number)` の厳密不等号（`D-010`）。** `UNIQUE (date, course, race_number)` がこのキーの一意性を保証する
- **`fetched_at` は取得時点を表し、`as_of` はデータが有効な時点を表す。** 両者を同じ意味で使わない
- 回収率は信頼区間なしに報告しない（`D-008`）。本仕様は `payouts` がその計算に足る形であることのみ担保する

## テスト観点

| # | 検証内容 | 期待 |
|---|---|---|
| 1 | `n_entries < n_starters` の行を挿入 | `CHECK` 違反で拒否 |
| 2 | 同一 `(date, course, race_number)` を2件挿入 | `UNIQUE` 違反で拒否 |
| 3 | 同一レースに `finish_pos = 3` を2行挿入 | **成功する**（同着） |
| 4 | `status` に定義外の値を挿入 | `CHECK` 違反で拒否 |
| 5 | `odds_high < odds_low` を挿入 | `CHECK` 違反で拒否 |
| 6 | `corners = []` と `corners = NULL` を挿入 | どちらも成功し、区別して読める |
| 7 | `corner_nos = [3,4]` のレースに `corners = [3,2]` を挿入 | 成功し、`corners[1]` が3コーナーの値として読める |
| 8 | `corner_nos = []` と `corner_nos = NULL` を挿入 | どちらも成功し、区別して読める |

### 実データによる受け入れケース

`D-023` と `Q-023` で取得済みのレースをそのまま fixture に使う。

| `race_id`（netkeiba） | レース | 検証する性質 | 期待 |
|---|---|---|---|
| `202305021211` | 日本ダービー2023 | 競走中止 | 全18行、`n_starters = 18`、`finish_pos` が非 `NULL` なのは17行。`corner_nos = [1,2,3,4]`、中止馬（馬番17）は `corners IS NULL` |
| `201905030611` | ユニコーンS2019 | 競走除外 | 全15行、`n_starters = 13`、`除` の2行は `finish_pos IS NULL` かつ `corners IS NULL`。`corner_nos = [3,4]` |
| `202007010811` | 高松宮記念2020 | 降着 | `status = '降着'` の行が1件、`finish_pos = 4`、`margin IS NULL` |
| `202306040911` | スプリンターズS2023 | 短距離のコーナー数 | `corner_nos = [3,4]`、完走馬の `corners` は全行2要素 |
| `200505050810` | ジャパンカップ2005 | 外国馬 | 外国調教馬の行が存在し、`horses` に過去走が0件 |
| `202009020204` | 障害4歳以上未勝利 | 同着 | `finish_pos` が `1,2,3,3,5,…`。`複勝` の払戻が4件、`ワイド` が5件。**ページ未取得**（`Q-026`） |

`202305021211` `201905030611` `202306040911` の `corner_nos` と `corners` は保存済みページで確認済み（`D-043` / `D-044`）。

**`202009020204` は障害競走であり、`D-025` により学習データには含まれない。** 記録形式の検証にのみ使う。同着の平地の実例は未取得（`Q-023`）。

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-023` | 失格の着順マーカーが未観測（出走取消は `D-048` で解決） | `status` の対応表が1行埋まらない。未知マーカーは取込失敗で運用上は閉じている |
| `Q-018` | 複勝・ワイドの過去発走前オッズが取得できない | `odds` は形として定義したが、**過去分は単勝以外ほぼ埋まらない** |
| `Q-021` | 天気予報の取得元が未定 | `races.weather_forecast` は当面すべて `NULL` |
| `Q-020` | 必須項目を欠く行の除外率の許容水準が未定 | `R-013` の合格判定が定まらない |
| `Q-026` | 同着レース（`202009020204`）のページが未取得 | 受け入れケース1件が実行できない。同着の制約はテスト観点3で合成データにより検証する |
