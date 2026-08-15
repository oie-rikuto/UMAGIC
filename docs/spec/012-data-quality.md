# 012 データ品質検査

| | |
|---|---|
| Phase | P-0 |
| 関連決定 | `D-011` `D-012` `D-034` `D-037` `D-040` `D-041` |
| 関連要件 | `R-013` `R-014` `R-015` |
| 先行仕様 | `001-schema.md` `002-loader.md` |
| 状態 | Draft |

## 目的

取り込んだデータの妥当性を検査し、`P-0` の完了を判定する。不良を見つけても削除せず、印を付けて下流に判断を渡す（`D-041`）。

## 入出力

- **入力**: `001-schema.md` の各テーブル、`002-loader.md` の `fetch_log` / `rejected_rows`
- **出力**: `quality_runs` / `quality_findings` への追記、Markdownレポート、終了コード（`fail` が1件でもあれば非0）

## 仕様

### 検査の2系統

| 系統 | いつ動くか | 対象 | 落ちたときの挙動 |
|---|---|---|---|
| 行単位（`R-013`） | 取り込み時 | 1行 | その行を取り込まず `rejected_rows` へ（`002-loader.md` の責務） |
| バッチ（`R-014`） | 取り込み後 | レース／テーブル全体 | **本仕様の責務。** `quality_findings` に記録し、削除はしない |

**本仕様が定義するのはバッチ側のみ。** 行単位の棄却規則は `002-loader.md` にある。

### 結果テーブル

```sql
CREATE TABLE quality_runs (
    run_id      BIGINT    PRIMARY KEY,
    started_at  TIMESTAMP NOT NULL,
    scope_from  DATE,
    scope_to    DATE,
    n_races     BIGINT    NOT NULL,
    n_runners   BIGINT    NOT NULL
);

CREATE TABLE quality_findings (
    run_id     BIGINT    NOT NULL,
    check_id   VARCHAR   NOT NULL,
    severity   VARCHAR   NOT NULL,
    race_id    BIGINT,
    horse_id   BIGINT,
    detail     VARCHAR,
    CHECK (severity IN ('fail', 'warn'))
);
```

`quality_findings` が `D-041` の「印」にあたる。下流は `race_id` で引いて用途ごとに除外を選ぶ。

### 検査一覧

| `check_id` | 重大度 | 内容 |
|---|---|---|
| `headcount_starters` | `fail` | `n_starters` がゲートを出た行数と一致する |
| `headcount_entries` | `fail` | `n_entries` が全行数と一致する |
| `finish_pos_rank` | `fail` | 着順が同着を許容した順位付けとして矛盾しない |
| `status_columns` | `fail` | `status` と `finish_pos` / `time_sec` の `NULL` 可否が一致する |
| `payout_horses` | `fail` | 払戻の馬番が実在し、ゲートを出ている |
| `odds_monotonic` | `fail` | 人気が小さい馬ほど単勝オッズが小さい |
| `corners_uniform` | `fail` | 同一レース内で `corners` の要素数が揃っている |
| `rejected_rate` | `warn` | 棄却行の率 |
| `fetch_incomplete` | `warn` | `fetch_log` に `ok` 以外がある率 |
| `odds_coverage` | `warn` | `odds` の券種別カバレッジ |
| `laps_coverage` | `warn` | ラップが存在しないレースの率 |

`warn` に閾値を設けない（`D-040`）。件数と年代別分布をレポートに出す。

### `fail` 系の検査（実行可能なSQL）

いずれも**違反行を返す**。0行なら通過。

```sql
-- headcount_starters
SELECT r.race_id FROM races r JOIN runners ru USING (race_id)
GROUP BY r.race_id, r.n_starters
HAVING r.n_starters <> COUNT(*) FILTER (WHERE ru.status IN ('出走','降着','競走中止','失格'));

-- headcount_entries
SELECT r.race_id FROM races r JOIN runners ru USING (race_id)
GROUP BY r.race_id, r.n_entries
HAVING r.n_entries <> COUNT(*);

-- finish_pos_rank
WITH ranked AS (
    SELECT race_id, horse_id, finish_pos,
           RANK() OVER (PARTITION BY race_id ORDER BY finish_pos) AS expected
    FROM runners WHERE finish_pos IS NOT NULL
)
SELECT race_id, horse_id, finish_pos, expected FROM ranked WHERE finish_pos <> expected;

-- status_columns
SELECT race_id, horse_id, status FROM runners
WHERE (status IN ('出走','降着') AND (finish_pos IS NULL OR time_sec IS NULL))
   OR (status IN ('競走中止','失格','出走取消','競走除外') AND finish_pos IS NOT NULL);

-- payout_horses
SELECT p.race_id, p.bet_type, p.comb_key, n
FROM payouts p, UNNEST(p.combination) AS t(n)
WHERE p.bet_type <> '枠連'
  AND NOT EXISTS (
      SELECT 1 FROM runners ru
      WHERE ru.race_id = p.race_id AND ru.number = n
        AND ru.status IN ('出走','降着','競走中止','失格'));

-- odds_monotonic
WITH o AS (
    SELECT race_id, horse_id, popularity, odds_win,
           LAG(odds_win) OVER (PARTITION BY race_id ORDER BY popularity) AS prev
    FROM runners WHERE popularity IS NOT NULL AND odds_win IS NOT NULL
)
SELECT race_id, horse_id, popularity, odds_win, prev
FROM o WHERE prev IS NOT NULL AND odds_win < prev;

-- corners_uniform
SELECT race_id FROM runners WHERE corners IS NOT NULL
GROUP BY race_id HAVING COUNT(DISTINCT len(corners)) > 1;
```

`finish_pos_rank` が `RANK()` を使うのは、`RANK()` が「1 + 自分より上位の馬の数」と一致し、同着で順位が飛ぶ挙動が `R-014` の不変条件と同じであるため。

`payout_horses` が `枠連` を除くのは、`枠連` の `combination` が馬番ではなく枠番を保持するため（`001-schema.md`）。

### `warn` 系の検査

分母を明示する。年代別（`races.date` の年）にも分解して出す。

| `check_id` | 分子 | 分母 |
|---|---|---|
| `rejected_rate` | `rejected_rows` の件数 | `runners` の行数 ＋ `rejected_rows` の件数 |
| `fetch_incomplete` | `fetch_log` で `outcome <> 'ok'` の件数 | `fetch_log` の全件数 |
| `odds_coverage` | 券種ごとに `odds` が存在するレース数 | `races` の件数 |
| `laps_coverage` | `laps` が1行も無いレース数 | `races` の件数 |

`odds_coverage` は `Q-018`（複勝・ワイドの過去オッズが取得できない）の影響を数値で可視化する。**単勝以外はほぼ0%になる見込みだが、実測していない。**

### レポート

Markdownで出力する。

- `fail` は検査ごとの違反件数と、先頭20件の `race_id`
- `warn` は率と、**年代別の内訳**
- `quality_runs` に記録した実行範囲と件数

## 制約

- **不良データを削除・上書きしない（`D-041`）。** 本仕様が書き込むのは `quality_runs` と `quality_findings` のみ
- **リーク検査は本仕様の対象外。** `domain-knowledge.md` 5節の7原則の検証は `004-leakage-test.md`（`P-1`）が担う。本仕様は「取り込んだ値が内部で矛盾していないか」だけを見る
- 予測対象レースのオッズを特徴量にしない（`D-002`）。`odds_monotonic` は品質検査であって特徴量ではない
- 学習はG1に絞らない（`D-003`）。検査対象を `grade` で絞らない
- 回収率は信頼区間なしに報告しない（`D-008`）。本仕様のレポートは回収率を扱わない

## テスト観点

### `fail` 系が違反を検出すること

DuckDB 1.4.5 で `001-schema.md` のDDLを作成し、以下を投入して**検出されることを確認済み**。

| 投入した状態 | 検出する検査 |
|---|---|
| 競走中止が1頭いるのに `n_starters` が完走数になっている | `headcount_starters` |
| 出走なのに `finish_pos` が `NULL` | `status_columns` |
| 払戻に存在しない馬番 | `payout_horses` |
| 人気1位が12.3倍、人気2位が4.2倍（`D-037` の不良を再現） | `odds_monotonic` |
| 同一レースで `corners` が `[1,1]` と `[1,2,3,4]` | `corners_uniform` |

### `finish_pos_rank` の同着の扱い

同着を許容し、矛盾のみを検出することを6ケースで**確認済み**。

| 着順 | 判定 |
|---|---|
| `1,2,3,4` | 通過 |
| `1,2,2,4` | 通過（2着同着） |
| `1,1,3` | 通過（1着同着） |
| `1,2,2,3` | **検出**（2着同着なら次は4着） |
| `1,1,2` | **検出**（1着同着なら次は3着） |
| `1,2,4` | **検出**（3着が欠番） |

### 実データによる受け入れケース

| `race_id` | 期待 |
|---|---|
| `202305021211` 日本ダービー2023 | 全検査通過。`n_starters = 18` で完走17頭でも `headcount_starters` に落ちない |
| `201905030611` ユニコーンS2019 | 全検査通過。`n_entries = 15` / `n_starters = 13` |
| `202007010811` 高松宮記念2020 | 全検査通過。降着馬の `finish_pos = 4` が `finish_pos_rank` に落ちない |
| `202009020204` 同着レース | `finish_pos_rank` を通過する |
| `201405010303` 東京3R | **`archive` から取り込めば `odds_monotonic` を通過し、`result` から取り込めば落ちる**（`D-037`） |

最後の1件が、`D-037` のページ選択規則に対する回帰テストになる。

## 未決事項

| ID | 内容 | この仕様書への影響 |
|---|---|---|
| `Q-020` | 棄却行・欠損の許容水準が未定 | `rejected_rate` と `fetch_incomplete` に閾値を置けない。`warn` 止まりで `P-0` の完了判定に入らない |
| `Q-023` | 出走取消・失格の着順マーカーが未観測 | 該当行は棄却されるため `rejected_rate` に現れる。件数が偏っていれば表記の存在に気づける |
| `Q-018` | 複勝・ワイドの過去オッズが取得できない | `odds_coverage` が低い値を返すのは異常ではない |
