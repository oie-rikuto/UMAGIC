# 013 馬場差推定

| | |
|---|---|
| Phase | P-1（実行は `P-3` の walk-forward に組み込む） |
| 関連決定 | `D-003`, `D-012`, `D-054`, `D-058`, `D-060`, `D-086`, `D-104`, `D-105`, `D-106`, `D-107` |
| 関連特徴量 | `F-301`, `F-302` |
| 状態 | Draft |

## 目的

走破タイムからレース効果（馬場差・展開・メンバーレベル）を分離し、残る馬効果を「その馬の真の能力」として出す。出力は `F-302` として Stage 2 の入力になる。

## 入出力

```python
from dataclasses import dataclass
from datetime import date
import duckdb
import polars as pl


@dataclass(frozen=True)
class VariantFit:
    """1回の推定結果。`as_of` 時点で利用可能なデータのみから推定したもの。"""

    as_of: date                 # この推定が有効になる日（推定に使ったデータはすべて `as_of` より前）
    horse_effects: pl.DataFrame # horse_id: Int64, effect: Float64
    race_effects: pl.DataFrame  # race_id: Int64, effect: Float64
    n_rows: int                 # 推定に使った出走行数
    n_iter: int                 # 反復回数
    converged: bool             # 収束条件を満たして停止したか
    sigma2_error: float         # 残差分散
    sigma2_race: float          # レース効果の分散
    sigma2_horse: float         # 馬効果の分散
    scale: pl.DataFrame         # surface: Utf8, distance: Int16, sd: Float64（D-105 の標準化に使った値）
    main_component: frozenset[int]  # 最大連結成分に属する race_id（識別可能な範囲）


def fit_track_variant(
    conn: duckdb.DuckDBPyConnection,
    race_ids: list[int],
    *,
    as_of: date,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> VariantFit:
    """`race_ids` の出走行から `F-301` を推定する。`race_ids` はすべて `as_of` より前のレースであること。"""


def horse_effect_series(fits: list[VariantFit]) -> pl.DataFrame:
    """複数の `VariantFit` を `F-302` の入力形式に束ねる。

    戻り値の列: `horse_id: Int64`, `as_of: Date`, `effect: Float64`。
    `(horse_id, as_of)` で一意。`as_of` の昇順に整列済み（`join_asof` の前提）。
    """
```

`003-features.md` 側の接続点（`D-060` / `D-107`）:

```python
def attach_f302(df: pl.DataFrame, horse_effects: pl.DataFrame) -> pl.DataFrame:
    """`df` に `f302` と `f302_unavailable` を追加する。

    `horse_effects` は `horse_effect_series()` の戻り値。`df` は `horse_id` と
    `date`（対象レースの日付）を持つこと。**`date` 未満で最も新しい `as_of` の
    行を `join_asof(by="horse_id")` で引く**（`D-107`）。
    """
```

## 仕様

### 1. 対象母集団

| 条件 | 値 |
|---|---|
| レース | `race_ids` に含まれ、かつ `date < as_of` |
| 出走行 | `runners.status IN ('出走', '降着', '失格')` かつ `time_sec IS NOT NULL` |
| 除外 | `競走中止` / `競走除外` / `出走取消`（`time_sec` が存在しない） |

**G1に絞らない**（`D-003`）。芝・ダートの全レースを使う。

### 2. 目的変数（`D-105`）

```
y_ij = time_sec_ij / sd(surface_j, distance_j)
```

`sd(surface, distance)` は、**その条件のレース内偏差の標準偏差**を推定期間内で求めた値。

```sql
SELECT r.surface, r.distance, STDDEV(ru.time_sec - m.race_mean) AS sd
FROM runners ru
JOIN races r USING (race_id)
JOIN (
    SELECT race_id, AVG(time_sec) AS race_mean
    FROM runners
    WHERE time_sec IS NOT NULL AND status IN ('出走','降着','失格')
    GROUP BY race_id
) m USING (race_id)
WHERE ru.time_sec IS NOT NULL
  AND ru.status IN ('出走','降着','失格')
  AND r.race_id = ANY(?)          -- 推定期間のレースのみ（D-054 原則7）
GROUP BY r.surface, r.distance
```

`sd` が求まらない、または `sd < MIN_SD`（`= 1e-3`）の `(surface, distance)` はその条件のレースごと推定から除外する。標本の薄い条件で目的変数が発散するのを防ぐ。

### 3. モデル

```
y_ij = μ + β_(surface, distance_band) + γ_(race_class) + u_j + v_i + ε_ij
```

| 記号 | 意味 | 種別 |
|---|---|---|
| `i` | 馬 | |
| `j` | レース | |
| `β` | 芝ダ×距離帯の固定効果 | 固定 |
| `γ` | クラスの固定効果 | 固定 |
| `u_j` | レース効果（馬場差 + 展開 + メンバーレベル） | ランダム |
| `v_i` | 馬効果（`F-302` として出力する） | ランダム |

距離帯は `D-066` の4区分を使う。`race_class` が `NULL` の行は `"不明"` という単一の水準に寄せる（`races.race_class` は7,314件が `NULL`）。

**レース効果はレース単位である**（セッション単位ではない）。`F-301` のカタログ定義どおり。

### 4. 推定手順（`D-104`）

```
入力: y（出走行ごと）、race_id、horse_id、固定効果の設計行列 X
初期化: v_i = 0（全馬）、u_j = 0（全レース）、θ = OLS(y ~ X)

for iter in 1..max_iter:
    r      = y - X·θ - v_i                       # レース効果以外を除いた残り
    ū_j    = mean(r, by=race_id)
    n_j    = count(r, by=race_id)
    u_j    = (n_j · ū_j) / (n_j + k_race)        # 縮約

    r      = y - X·θ - u_j                       # 馬効果以外を除いた残り
    v̄_i    = mean(r, by=horse_id)
    n_i    = count(r, by=horse_id)
    v_i    = (n_i · v̄_i) / (n_i + k_horse)       # 縮約

    θ      = OLS(y - u_j - v_i ~ X)              # 固定効果を更新

    ε      = y - X·θ - u_j - v_i
    σ²_ε   = var(ε)
    σ²_u   = var(u_j)   （レース単位、重みなし）
    σ²_v   = var(v_i)   （馬単位、重みなし）
    k_race  = σ²_ε / max(σ²_u, MIN_VAR)
    k_horse = σ²_ε / max(σ²_v, MIN_VAR)

    if max(|Δu|) < tol かつ max(|Δv|) < tol:
        converged = True; break
```

| 定数 | 値 | 意味 |
|---|---|---|
| `MIN_VAR` | `1e-8` | 分散成分の下限。`k` の発散を防ぐ（`D-102` と同種のガード） |
| `MIN_SD` | `1e-3` | 2節の条件別標準偏差の下限 |
| `max_iter` | `100` | |
| `tol` | `1e-6` | 標準化後の尺度での変化量 |

初回反復では `σ²_u` / `σ²_v` が未定のため、`k_race = k_horse = 1.0` から始める。

**中心化**: 各反復の末尾で `u_j` と `v_i` の平均を0に揃え、差分を `μ`（`θ` の切片）へ移す。交差ランダム効果は加法定数の分だけ不定であり、揃えないと反復ごとに値が漂って `R-021` を満たさない。

### 5. 連結成分の扱い

馬とレースの二部グラフ（同じ馬が複数レースに出ることで辺が張られる）は、成分ごとに加法定数が独立に不定である。したがって**成分をまたぐ効果は比較できない**。4節の中心化は全体に対して1回行うため、この不定性を解消しない。

| 手順 | 内容 |
|---|---|
| 1 | Union-Find で `race_id` の連結成分を求める（辺は「同じ馬が両方のレースに出た」） |
| 2 | 最大成分を `main_component` とする |
| 3 | **最大成分に属さないレースの `race_id` と、そこにしか出走していない馬の `horse_id` は `horse_effects` / `race_effects` に含めない** |

`horse_effects` に現れない馬は `attach_f302()` で `f302 = NaN` / `f302_unavailable = 1` になる（`D-058`）。

**この分断はデータ窓の末尾で必ず発生する。** 実測（2015-2024、`Q-025` に記録）では最大成分が98.07%で、外れた625レースはすべて2024-06-01以降の新馬・未勝利だった。as-of で切る以上、各ブロック境界の直前にデビューした馬が同じ島を作る。仕様上の異常ではない。

### 6. 実行の粒度（`D-106`）

fold ごとに5回推定する。

| 用途 | 推定に使う期間 | 割り当て先 | `as_of` |
|---|---|---|---|
| ブロック `b`（`b = 1..4`）の out-of-fold | 学習期間からブロック `b` を除いた残り | ブロック `b` の行 | ブロック `b` の開始日 |
| 学習期間全体 | 学習期間すべて | 検証期間の行 | `fold.valid_start` |

ブロックの分割は `training.cross_fit_blocks(fold, n_blocks=4)` をそのまま使う。**独自の分割を実装しない。**

## 制約

- **学習は全レースで行う**（`D-003`）。G1だけを取り出して推定してはならない。接続が弱い部分集団では馬効果とレース効果が分離できない
- **予測対象レースのオッズを一切使わない**（`D-002`）。本仕様が触れる列は `time_sec` / `race_id` / `horse_id` / `surface` / `distance` / `race_class` / `status` のみ
- **推定期間も as-of で切る**（`D-054` 原則7）。2節の `sd`、3節の固定効果、4節の分散成分のすべてが対象。`as_of` 以降のレースを1行でも含めてはならない
- **対象レース自身を推定に含めない。** `F-301` の確定時刻は `木曜`（`domain-knowledge.md` 4節）であり、対象レースの `time_sec` は発走後にしか判明しない。含めれば `D-007` が Stage 1 で回避しているのと同種のリークになる
- **簡易版のフォールバックを実装しない**（`D-060`）。推定できない馬は `NaN` + 指示子であって、馬場差を引かない補正タイムで代用しない
- **同じ入力から同じ出力が再現できること**（`R-021`）。4節の中心化と、集計順序の決定性（`D-055`）を守る。`~1e-11` 以下の浮動小数点丸め誤差は `D-102` の扱いに従う
- **`n_starters` を分母に使う箇所では実出走頭数を使う**（`D-012`）

## テスト観点

| # | 観点 | 期待 |
|---|---|---|
| 1 | 合成データ: レース効果 `[+1, 0, -1]`、馬効果0で3レース | `race_effects` が縮約後の比で `+1 : 0 : -1` の順序を保つ |
| 2 | 合成データ: 全馬同能力・全レース同条件 | `horse_effects` がすべて0近傍（`|v| < 0.05`） |
| 3 | 合成データ: 1頭だけ明確に速い馬を全レースに混ぜる | その馬の `effect` が他のどの馬より小さい（速い＝タイムが小さい） |
| 4 | 中心化 | `horse_effects["effect"].mean()` と `race_effects["effect"].mean()` がともに `|x| < 1e-9` |
| 5 | 縮約 | 出走1回の馬の `|effect|` が、同じ平均残差で20回走った馬の `|effect|` より小さい |
| 6 | 分散成分 | `sigma2_error` / `sigma2_race` / `sigma2_horse` がすべて正で有限 |
| 7 | 標準化 | `scale` が `(surface, distance)` ごとに1行。芝1200mとダート2400mで `sd` が2倍以上違う（実データ） |
| 8 | as-of（原則7） | `as_of` より後のレースを `race_ids` に足しても、`as_of` 以前だけで推定した結果と**ビット一致**する（足した分は `date < as_of` で落ちるため） |
| 9 | 対象レースの非混入 | 対象レース自身を `race_ids` に含めても、その `time_sec` が `horse_effects` に影響しない（`date < as_of` で除外される） |
| 10 | 連結成分 | 他とまったく馬を共有しないレース群を作ると、それらの `race_id` が `main_component` に含まれず、出力にも現れない |
| 11 | 欠損 | `horse_effects` に無い馬が `attach_f302()` で `f302 = NaN` / `f302_unavailable = 1` になる |
| 12 | as-of 結合 | `as_of` が対象行の日付と一致しなくても、**日付未満で最も新しい**推定値が引かれる。対象行の日付以降の `as_of` は引かれない |
| 13 | 再現性（`R-021`） | 同じ入力で3回実行し、`horse_effects` の差が `1e-11` 以下 |
| 14 | 決定性（`D-055`） | 行の投入順を変えても出力が一致する |
| 15 | 収束（合成データ） | 小規模な合成データ（本仕様書の合成テスト規模）では `converged=True` かつ `n_iter < max_iter`。実データ規模での収束特性は `D-108` を参照（`converged=False` のまま `max_iter` に達することがあるが、`horse_effects` の値自体は実務上問題にならない精度で安定する） |
| 16 | 空入力 | `race_ids=[]` で例外を出さず、空の `horse_effects` を返す |
| 17 | `sd` の下限 | 1レースしかない `(surface, distance)` の条件が推定から除外され、例外にならない |
| 18 | リーク検査 | `tests/test_leakage.py` の9件が緑のまま。`F-302` を有効にした状態で実データ版（`pytest -m realdata`）も通る |

## 未決事項

| ID | 内容 | 何がブロックされるか |
|---|---|---|
| `Q-041` | `F-302` の補正タイム集計（最高値・直近平均・トレンド）を実装するか。`D-107` で当面は馬効果のみとした | `domain-knowledge.md` の `F-302` の定義の一部が未実装のまま残る。`F-301` の効果測定後に判断する |
| `Q-042` | レース効果に吸われる「メンバーレベル」を分離すべきか。`domain-knowledge.md` は「強いメンバーが揃ったレースは速い時計が出るが、それは馬場が速いからではない」と警告しており、現仕様では馬場差・展開・メンバーレベルが1つの `u_j` に同居する | `F-301` の `race_effects` を「馬場差」として単体で使う用途（`F-502` の基準タイムなど）。`F-302`（馬効果）としての利用はブロックしない |
| `Q-043` | 斤量補正・ペース補正を目的変数に入れるか。`domain-knowledge.md` の補正タイムの式は `走破タイム − 基準タイム − 馬場差 − ペース補正 − 斤量補正` だが、本仕様は馬場差（レース効果）のみを引く | 補正タイムを人が読む用途。`F-302` の予測力そのものはブロックしない |
| `Q-025` | **Resolved**（`D-104`〜`D-107`） | — |
