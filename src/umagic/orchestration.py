"""`P-3` 学習パイプラインの結線（`predict_fold` の実装）。

`014-training-pipeline.md` の `run_walk_forward()` が要求する
`predict_fold(conn, fold) -> DataFrame` を、`003-features.md`（全特徴量）・
`006-stage1-pace.md`（Stage 1）・`007-stage2-ranker.md`（Stage 2）・
`015-calibration.md`（確率校正）を実際に結線して実装する。各仕様書
自身がこの結線を「呼び出し側（学習の orchestration 層）の責務」として
外に出しており、`006-stage1-pace.md` / `007-stage2-ranker.md` それぞれの
モジュールの型シグネチャ一覧自体、そのための関数を持たない。

## fold あたりの学習回数

`cross_fit_blocks()`（`014-training-pipeline.md` / `D-086`）の**同じ
4分割を2箇所で使う**。

| 用途 | 参照 | 回数 |
|---|---|---|
| Stage 1 の out-of-fold `F-102`（Stage 2 の学習データに使う） | `D-086` | `n_blocks` 回 + 学習期間全体で1回 |
| Stage 2 自身の out-of-fold スコア（校正データ作成） | `015-calibration.md` 1節 / `D-098` | `n_blocks` 回 |
| Stage 2 の本番モデル（検証期間の予測・実運用） | `007-stage2-ranker.md` | 1回 |

Stage 1 の `F-102` は行単位で既に out-of-fold（Stage 1 のクロスフィットは
Stage 2 全体を1回組み立てる**前に**完了している）なので、Stage 2 の
校正用クロスフィットに再度ネストした Stage 1 の作り直しは不要（`D-086`
のクロスフィットが提供する OOF 性質は行に付随する）。

## クロスフィット用サブモデルの inner 検証（本ファイル固有の設計判断）

`014-training-pipeline.md` 3節（`D-084`）のネスト検証は **fold 全体で
学習する「本番」の Stage 2 モデル1本にのみ適用する。** クロスフィット
用のサブモデル（校正データ作成の `n_blocks` 回、Stage 1 の `n_blocks`
回）には適用しない。

理由: `cross_fit_blocks()` の境界と `fold.inner_valid_start`（学習期間
末尾1年）は独立に決まるため、学習期間が短い初期 fold ではブロックの
1つが inner 検証区間の大半と重なり、「そのブロックを除いた rest」から
inner 検証区間が事実上消えるケースがある。Stage 1 の実装（`stage1.py`
`LightGBMStage1Model`）が既にラウンド数を固定（`num_boost_round=50`、
inner 検証を使わない）としているのと対称に、Stage 2 のクロスフィット
用サブモデルも「本番モデルの inner 検証で選んだラウンド数
（`best_iteration`）を固定で流用する」方式にする。

**`early_stopping_rounds` を大きくして早期終了を抑止する案は機能しない
（実装当初はこれを採用しており、後日の誤りとして訂正した）。**
LightGBM の早期終了コールバックは、猶予ラウンド数を超えたかに関わらず
**最終ラウンドで無条件に発火し**、その時点の検証セットで最良だった
ラウンドまでモデルをロールバックする。ダミーの inner セットを渡す
このユースケースでは、そのロールバック先に意味が無く、指定した
ラウンド数より少ないラウンド数に痩せてしまう。正しくは
`fit_stage2_fixed_rounds()`（`007-stage2-ranker.md` の型シグネチャに
無い追加関数。早期終了の仕組みそのものを使わない）を使う。

## 確率校正の適用範囲（`D-099`）と `run_walk_forward()` の1列制約

`run_walk_forward()` の出力スキーマは `race_id, horse_id, fold_index,
y_true, y_pred` の1列で固定（`014-training-pipeline.md` 6節）。一方
`D-099` は「校正は `g1` 母集団の評価にのみ適用し、`all` 母集団には
適用しない」としており、母集団ごとに異なる `y_pred` が必要になる。

この2つは両立しないため、`predict_fold()`（`run_walk_forward()` に
直接渡す方）は**校正前の生スコアを返す**。校正は
`Stage2FoldRunner.fold_calibrators`（fold ごとに保持する `Calibrator`）
を使って `run_walk_forward()` の呼び出しの**後で**、`g1` 母集団の行に
だけ別途適用する（`apply_g1_calibration()`）。`010-backtest.md`
（`P-4`、未作成）がこの母集団ごとの指標計算を正式に引き取るまでの
つなぎとして、`run_p3_completion_check()` に最小限の実装を置く。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb
import polars as pl

from umagic.calibration import Calibrator, fit_calibrator
from umagic.sealed import is_sealed
from umagic.features.build import FeatureFn, build_features
from umagic.features.f101 import compute_f101
from umagic.features.f103 import compute_f103
from umagic.features.f104 import compute_f104
from umagic.features.f201 import compute_f201
from umagic.features.f202 import compute_f202
from umagic.features.f302 import attach_f302
from umagic.features.f303 import compute_f303
from umagic.features.f304 import compute_f304
from umagic.features.f501 import compute_f501
from umagic.features.f503 import compute_f503
from umagic.features.f601 import compute_f601
from umagic.features.f602 import compute_f602
from umagic.features.f603 import compute_f603
from umagic.features.f701 import compute_f701
from umagic.features.f702 import compute_f702
from umagic.features.f703 import compute_f703
from umagic.features.f704 import compute_f704
from umagic.features.f801 import compute_f801
from umagic.features.f802 import compute_f802
from umagic.features.f803 import compute_f803
from umagic.features.f804 import compute_f804
from umagic.features.f805 import compute_f805
from umagic.features.f806 import compute_f806
from umagic.features.f807 import compute_f807
from umagic.features.f808 import compute_f808
from umagic.features.relative import relativize
from umagic.stage1 import LightGBMStage1Model, build_inputs as stage1_build_inputs
from umagic.stage1 import build_target as stage1_build_target
from umagic.stage1 import predict_f102
from umagic.track_variant import fit_track_variant, horse_effect_series
from umagic.stage2 import (
    apply_category_mappings,
    build_category_mappings,
    build_labels,
    fit_stage2,
    fit_stage2_fixed_rounds,
    predict_win_prob,
    race_group,
)
from umagic.training import DEFAULT_SEALED_YEARS, Fold, cross_fit_blocks, run_walk_forward, sample_weights

# ---------------------------------------------------------------------------
# 003-features.md の全特徴量（F-102/F-104/F-302 は別扱い。431節の確定時刻表どおり）
# F-203（Q-030）/ F-502（Q-031）は未実装のため含まない
# ---------------------------------------------------------------------------

FEATURE_FNS: list[FeatureFn] = [
    compute_f101, compute_f103, compute_f201, compute_f202, compute_f303, compute_f304,
    compute_f501, compute_f503, compute_f601, compute_f602, compute_f603,
    compute_f701, compute_f702, compute_f703, compute_f704,
    compute_f801, compute_f802, compute_f803, compute_f804, compute_f805,
    compute_f806, compute_f807, compute_f808,
]

# race_level=True の列（003-features.md 4節）。F-901（relativize）を適用しない
RACE_LEVEL_COLUMNS = frozenset({
    "f501", "f501_unavailable",
    "f503_meeting_no", "f503_meeting_day",
    "f803_distance", "f803_surface", "f803_direction", "f803_n_starters",
    "f803_season", "f803_prize", "f803_race_class", "f803_weight_rule",
    "f804_weather", "f804_weather_forecast", "f804_track_condition",
    "f102", "f104",
})

# 文字列型のカテゴリ列（stage2._CATEGORY_COLS と揃える。D-092の丸めを適用する）
CATEGORY_COLUMNS = frozenset({
    "sire_id", "damsire_id", "jockey_id", "trainer_id",
    "f602_prev_grade",
    "f803_surface", "f803_direction", "f803_race_class", "f803_weight_rule", "f803_season",
    "f804_weather", "f804_weather_forecast",
    "f805_sex",
    "f807_prev_surface", "f807_prev_course",
})

# D-081 / D-092: 2026-08-24、実データでのハイパーパラメータ探索により選定
# （D-101 追記）。**探索は学習期間末尾1年の全レース（G1に限らない）の
# inner LogLossで比較しており、その指標では「G1重み強め」が最良だった
# （2.2301）。しかし実際に7fold walk-forwardを回してG1のLogLossを直接
# 比較したところ、「G1重み強め」はG1で明確に悪化した（2.2985→2.4030）。
# 探索指標とR-023の目的指標（G1 LogLoss）が乖離していたため、G1実測が
# 良かったこちらの値（探索前の暫定値と同じ）に戻す。** `min_category_count`
# は探索・G1実測のどちらでも問題は見られず、そのまま採用する
DEFAULT_CLASS_WEIGHTS: dict[str | None, float] = {"G1": 5.0, "G2": 3.0, "G3": 2.0, "L": 1.5}
DEFAULT_MIN_CATEGORY_COUNT = 20
DEFAULT_N_BLOCKS = 4
DEFAULT_NUM_BOOST_ROUND = 200
DEFAULT_EARLY_STOPPING_ROUNDS = 20

# D-113（2026-08-25）: `Q-039` の G1 OOF 探索で、`all`・G1 OOF 双方
# 3fold全て改善する唯一の設定群だった。7fold walk-forward（`R-023`の
# 判定指標そのもの）で確認し、`all`が7fold全て（-0.028〜-0.039の狭い
# 幅）、G1（135レース集計）が-0.0270、いずれも改善したため既定にした
# （`v6`基準比: G1 model 2.3355→2.3085、all model 2.2194→2.1853）。
# LightGBM既定は両方とも10.0。
DEFAULT_CAT_SMOOTH = 200.0
DEFAULT_CAT_L2 = 200.0

# D-133（2026-08-26）: Plackett-Luce top-3 を既定の目的関数にする。
# `R-029` が測る勝利確率の尤度そのものを最適化する（`lambdarank` は
# NDCG を最適化しており目的が一致していなかった、`D-130`）。
# 7fold walk-forward で `all` 捕捉率 0.6767→0.6951・G1 0.5308→0.5932。
# 学習率とヘシアン下限は PL と組で必要になる（`D-130`）。
DEFAULT_OBJECTIVE = "pl3"
DEFAULT_LEARNING_RATE = 0.02
DEFAULT_MIN_SUM_HESSIAN = 1.0
DEFAULT_NUM_BOOST_ROUND_PL = 1200


def _race_ids_by_date(conn: duckdb.DuckDBPyConnection, race_ids: list[int]) -> pl.DataFrame:
    if not race_ids:
        return pl.DataFrame(schema={"race_id": pl.Int64, "date": pl.Date, "grade": pl.Utf8})
    return conn.execute(
        "SELECT race_id, date, grade FROM races WHERE race_id = ANY(?) ORDER BY race_id", [race_ids]
    ).pl()


def _ids_in_range(dated: pl.DataFrame, start: date, end: date) -> list[int]:
    return dated.filter((pl.col("date") >= start) & (pl.col("date") <= end))["race_id"].to_list()


def _race_ids_in_range(
    conn: duckdb.DuckDBPyConnection, start: date, end: date, *, today: date, sealed_years: int,
) -> list[int]:
    """指定期間のレースIDを、封印G1を除いて返す（`D-017` / `D-079`）。

    `training.make_folds()` は fold の**境界年**を封印除外後の母集団から
    決めるが、fold の日付範囲そのもの（`train_start`〜`train_end` /
    `valid_start`〜`valid_end`）は封印を知らない。境界年が非封印でも、
    その範囲内の個々のレースが封印期間に入っていることは普通に起きる
    （`today` に近い年ほど）。ここで実際の `race_id` を確定する際に
    `is_sealed()` を適用しないと、学習にも検証にも封印G1が混入する。
    """
    rows = conn.execute(
        "SELECT race_id, date, grade FROM races WHERE date >= ? AND date <= ? ORDER BY race_id",
        [start, end],
    ).pl()
    if rows.is_empty():
        return []
    mask = [
        not is_sealed(d, g, today=today, n_years=sealed_years)
        for d, g in zip(rows["date"].to_list(), rows["grade"].to_list())
    ]
    return rows.filter(pl.Series(mask, dtype=pl.Boolean))["race_id"].to_list()


# ---------------------------------------------------------------------------
# Stage 1: クロスフィッティング（D-086）
# ---------------------------------------------------------------------------

def stage1_fit_all(
    conn: duckdb.DuckDBPyConnection, fold: Fold, train_ids: list[int], *,
    n_blocks: int = DEFAULT_N_BLOCKS,
) -> tuple[pl.DataFrame, LightGBMStage1Model]:
    """学習期間ぶんの Stage 1 を1回のクエリでまとめて処理する（`D-086`）。

    `stage1_build_target()`/`stage1_build_inputs()` は対象レース自身の
    過去走だけを見る、行ごとに独立な計算である（他のレースが同じ
    バッチに含まれるかに依存しない）。したがって学習期間全体で1回だけ
    呼び、クロスフィットの `n_blocks` 個のブロックにはその結果を
    スライスして与えれば、ブロックの数だけ同じ高コストな SQL
    （`F-101` の相関自己結合を含む）を再実行せずに済む
    （元は fold あたり `n_blocks + 1` 回呼んでいた）。

    戻り値: `(学習期間の out-of-fold F-102, 学習期間全体で学習したモデル)`。
    後者は呼び出し側が検証期間の予測に使う（`stage1_fit_full()` の役割を
    兼ねる）。`train_ids` は呼び出し側が確定済みの学習期間レースID
    （封印G1を除いたもの）を渡す。
    """
    empty_oof = pl.DataFrame(schema={"race_id": pl.Int64, "f102": pl.Float64})
    full_model = LightGBMStage1Model()
    if not train_ids:
        return empty_oof, full_model

    target = stage1_build_target(conn, train_ids)
    # 予測（`predict_f102`）は laps の有無に依存しない（`D-091`）ため、
    # 入力は target で絞る前の train_ids 全体について作る
    x = stage1_build_inputs(conn, train_ids, as_of=fold.valid_start)
    merged = x.join(target.select(["race_id", "f102_actual"]), on="race_id", how="inner")
    if merged.is_empty():
        return empty_oof, full_model  # 学習可能な行が無い（未fit のモデルを返す）

    full_model.fit(
        merged.drop(["race_id", "f102_actual"]), merged["f102_actual"],
        sample_weight=None, seed=fold.seed,
    )

    train_dated = _race_ids_by_date(conn, train_ids)
    blocks = cross_fit_blocks(fold, n_blocks=n_blocks)
    oof_parts: list[pl.DataFrame] = []
    for b_start, b_end in blocks:
        block_ids = set(_ids_in_range(train_dated, b_start, b_end))
        if not block_ids:
            continue
        rest_merged = merged.filter(~pl.col("race_id").is_in(block_ids))
        block_x = x.filter(pl.col("race_id").is_in(block_ids))
        if rest_merged.is_empty() or block_x.is_empty():
            continue
        block_model = LightGBMStage1Model()
        block_model.fit(
            rest_merged.drop(["race_id", "f102_actual"]), rest_merged["f102_actual"],
            sample_weight=None, seed=fold.seed,
        )
        preds = block_model.predict(block_x.drop("race_id"))
        oof_parts.append(pl.DataFrame({
            "race_id": block_x["race_id"].to_list(), "f102": preds.to_list(),
        }))

    if not oof_parts:
        return empty_oof, full_model
    return pl.concat(oof_parts), full_model


# ---------------------------------------------------------------------------
# F-301: 馬場差推定のクロスフィッティング（D-106）
# ---------------------------------------------------------------------------

def track_variant_fit_all(
    conn: duckdb.DuckDBPyConnection, fold: Fold, train_ids: list[int], *,
    n_blocks: int = DEFAULT_N_BLOCKS,
) -> pl.DataFrame:
    """fold あたり5回の `F-301` 推定をまとめ、`F-302` が使える時系列にする（`D-106`）。

    `training.cross_fit_blocks()` の `n_blocks` 個のブロックそれぞれについて
    「そのブロックを除いた残りで推定し `as_of` をブロック開始日にする」、
    加えて「学習期間全体で推定し `as_of` を `fold.valid_start` にする」の
    計 `n_blocks + 1` 回を実行する（`stage1_fit_all()` と対称の設計）。

    戻り値は `horse_effect_series()` が返す `(horse_id, as_of, effect)` の
    時系列で、`attach_f302()` に as-of 結合でそのまま渡せる。学習期間の
    行はブロック `b` の `as_of`（=ブロック `b` の開始日）が「その行自身の
    日付未満で最も新しい」ものとして引かれ、検証期間の行は学習期間全体の
    推定（`as_of=fold.valid_start`）を引く（ブロック境界日はすべて
    `valid_start` より前のため）。
    """
    if not train_ids:
        return horse_effect_series([])

    train_dated = _race_ids_by_date(conn, train_ids)
    blocks = cross_fit_blocks(fold, n_blocks=n_blocks)

    fits = []
    for b_start, b_end in blocks:
        block_ids = set(_ids_in_range(train_dated, b_start, b_end))
        rest_ids = [r for r in train_ids if r not in block_ids]
        if not rest_ids:
            continue
        fits.append(fit_track_variant(conn, rest_ids, as_of=b_start))

    fits.append(fit_track_variant(conn, train_ids, as_of=fold.valid_start))
    return horse_effect_series(fits)


# ---------------------------------------------------------------------------
# Stage 2: 特徴量行列の組み立て（003 全特徴量 + F-102/F-104/F-302 の接続）
# ---------------------------------------------------------------------------

def assemble_stage2_matrix(
    conn: duckdb.DuckDBPyConnection, race_ids: list[int], *, as_of: date, f102: pl.DataFrame,
    horse_effects: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Stage 2 の入力行列を組み立てる。

    `f102`: `race_id, f102` の2列。`race_ids` に含まれるが `f102` に
    無いレースは `f102` が `null` になる（`D-091`。ラップ本数不足など）。

    `horse_effects`: `track_variant_fit_all()` が返す `F-301` の時系列
    （`D-106`）。省略時（`None`）は空扱いになり `f302` が全行 `NaN`
    になる（`D-060`、`013` を呼ばない経路の後方互換）。
    """
    if not race_ids:
        return pl.DataFrame(schema={"race_id": pl.Int64, "horse_id": pl.Int64})

    base = build_features(conn, as_of=as_of, race_ids=race_ids, feature_fns=FEATURE_FNS)
    base = base.join(f102, on="race_id", how="left")

    n_starters = conn.execute(
        "SELECT race_id, n_starters FROM races WHERE race_id = ANY(?)", [race_ids]
    ).pl()
    base = base.join(n_starters, on="race_id", how="left")

    base = compute_f104(base)  # f102 × f103_z（f103 の relativize は本関数が内部で行う）

    if horse_effects is None:
        horse_effects = pl.DataFrame(
            schema={"horse_id": pl.Int64, "as_of": pl.Date, "effect": pl.Float64}
        )
    # attach_f302() は `date`（対象レースの日付）で as-of 結合する（D-107）。
    # 一時的に結合し、使い終わったら落とす（呼び出し側の後続 join と
    # 列名が衝突しないようにするため。n_starters と同じ扱い）
    dated_for_f302 = conn.execute(
        "SELECT race_id, date FROM races WHERE race_id = ANY(?)", [race_ids]
    ).pl()
    base = base.join(dated_for_f302, on="race_id", how="left")
    base = attach_f302(base, horse_effects)
    base = base.drop("date")

    # F-901（レース内相対化）: race_level でも category でも unavailable 指示子でもない列
    # 「f103」は compute_f104 が内部で既に relativize 済み（f103_z / f103_rank が
    # 存在する）。その2列も除かないと z-score の z-score のような無意味な列が
    # 二重に作られる（実装ミスで一度混入させ、テストで発見・修正した）
    skip = RACE_LEVEL_COLUMNS | CATEGORY_COLUMNS | {
        "race_id", "horse_id", "n_starters", "f103", "f103_z", "f103_rank",
    }
    value_cols = [
        c for c in base.columns
        if c not in skip and not c.endswith("_unavailable")
    ]
    for col in value_cols:
        base = relativize(base, col, race_id_col="race_id", n_starters_col="n_starters")

    return base.drop("n_starters")


def _feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("race_id", "horse_id", "label", "sample_weight", "date")]


# ---------------------------------------------------------------------------
# fold ごとの実行本体
# ---------------------------------------------------------------------------

@dataclass
class Stage2FoldRunner:
    """`run_walk_forward()` に渡す `predict_fold` callback を提供する。

    fold ごとに校正器（`Calibrator`）を `fold_calibrators` に、検証期間の
    生スコア（校正前）を `fold_valid_scores` に保持する（校正の適用は
    `run_walk_forward()` の外、`D-099` の理由はモジュール docstring を
    参照。`fold_valid_scores` が要る理由は `apply_g1_calibration()` の
    docstring を参照）。

    `today`/`sealed_years` は `run_walk_forward()` に渡すものと**同じ値を
    渡すこと**。fold の日付範囲から実際の `race_id` を確定する際に、
    封印G1（`D-017`）を除外するために使う。
    """

    today: date
    sealed_years: int = DEFAULT_SEALED_YEARS
    n_blocks: int = DEFAULT_N_BLOCKS
    class_weights: dict[str | None, float] = field(default_factory=lambda: dict(DEFAULT_CLASS_WEIGHTS))
    min_category_count: int = DEFAULT_MIN_CATEGORY_COUNT
    # `lr=0.02` では早期終了が350〜750ラウンド付近で来るため上限を上げる（`D-133`）
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND_PL
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS
    # F-301/F-302（D-104〜D-108）を組み込むか。**既定は無効**（D-110）。
    # D-109（build_features() のバグ修正）を反映した汚染されていない
    # 基準でも、7fold walk-forward で all 母集団が7fold全て悪化した
    # （+0.047〜+0.087、v6→v7）。R-023 を改善するはずの機能が悪化させて
    # いるため、原因（Q-041/Q-042）が分かるまで既定で無効にする
    include_track_variant: bool = False
    # 目的関数（`D-095` / `D-130` / `D-133`）。`R-029` が測る勝利確率の
    # 尤度そのものを最適化する Plackett-Luce top-3 を既定にする。
    # `lambdarank` は NDCG（順位）を最適化しており目的が一致していなかった。
    # **`learning_rate` と `min_sum_hessian_in_leaf` の既定も併せて変えている**
    # （下記 `extra_lgb_params`。既定の `min_data_in_leaf=1` のままでは
    # ヘシアン `p(1-p)` が0に寄ったとき葉の値が発散しうる）
    objective: str = DEFAULT_OBJECTIVE
    # LightGBM の `params` にそのまま合流する追加設定。既定は `D-113`
    # で確定した `cat_smooth`/`cat_l2`（`Q-039`）。`categorical_feature` は
    # `predict_fold()` が自前で組み立てるため、ここに含めても上書きされる
    extra_lgb_params: dict = field(
        default_factory=lambda: {
            "cat_smooth": DEFAULT_CAT_SMOOTH, "cat_l2": DEFAULT_CAT_L2,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "min_sum_hessian_in_leaf": DEFAULT_MIN_SUM_HESSIAN,
        }
    )
    fold_calibrators: dict[int, Calibrator] = field(default_factory=dict)
    fold_inner_metrics: dict[int, dict] = field(default_factory=dict)
    fold_valid_scores: dict[int, pl.DataFrame] = field(default_factory=dict)
    # 校正データ作成に使った G1 の out-of-fold スコア（`race_id, horse_id,
    # score, is_winner, n_starters`）。fold ごとの `Calibrator` は既に
    # `fold_calibrators` にあるが、`Q-037`（頭数帯ごとの T の妥当性確認）
    # には元データが要るため、診断用に別途保持する
    fold_oof_g1: dict[int, pl.DataFrame] = field(default_factory=dict)

    def _race_ids(self, conn: duckdb.DuckDBPyConnection, start: date, end: date) -> list[int]:
        return _race_ids_in_range(conn, start, end, today=self.today, sealed_years=self.sealed_years)

    def predict_fold(self, conn: duckdb.DuckDBPyConnection, fold: Fold) -> pl.DataFrame:
        train_ids = self._race_ids(conn, fold.train_start, fold.train_end)
        valid_ids = self._race_ids(conn, fold.valid_start, fold.valid_end)

        # --- Stage 1（D-086） ---
        f102_train_oof, stage1_full = stage1_fit_all(conn, fold, train_ids, n_blocks=self.n_blocks)
        f102_valid = predict_f102(stage1_full, conn, valid_ids, as_of=fold.valid_start)
        f102_all = pl.concat([
            f102_train_oof.select(["race_id", "f102"]),
            f102_valid.select(["race_id", "f102"]),
        ])

        # --- F-301（D-106） ---
        horse_effects = (
            track_variant_fit_all(conn, fold, train_ids, n_blocks=self.n_blocks)
            if self.include_track_variant else None
        )

        # --- Stage 2 入力行列（train + valid まとめて1回、fold.valid_start で as_of） ---
        all_ids = train_ids + valid_ids
        x_all = assemble_stage2_matrix(
            conn, all_ids, as_of=fold.valid_start, f102=f102_all, horse_effects=horse_effects,
        )
        labels_all = build_labels(conn, all_ids)
        dated = _race_ids_by_date(conn, all_ids)

        data = (
            x_all.join(labels_all, on=["race_id", "horse_id"], how="inner")
            .join(dated.select(["race_id", "date"]), on="race_id", how="left")
            .sort(["race_id", "horse_id"])
        )

        train_data = data.filter(pl.col("race_id").is_in(train_ids)).sort(["race_id", "horse_id"])
        valid_data = data.filter(pl.col("race_id").is_in(valid_ids)).sort(["race_id", "horse_id"])

        mappings = build_category_mappings(
            train_data, min_count=self.min_category_count, columns=CATEGORY_COLUMNS,
        )
        train_data = apply_category_mappings(train_data, mappings)
        valid_data = apply_category_mappings(valid_data, mappings)

        sw = sample_weights(conn, train_ids, class_weights=self.class_weights)
        # D-055: 結合後の行順を明示的に保証する（race_group() が (race_id, horse_id)
        # 連続を前提とするため、join の出力順に暗黙に依存しない）
        train_data = train_data.join(sw, on="race_id", how="left").sort(["race_id", "horse_id"])

        feature_cols = _feature_columns(train_data)
        cat_idx = [feature_cols.index(c) for c in CATEGORY_COLUMNS if c in feature_cols]
        params = {**self.extra_lgb_params, "categorical_feature": cat_idx}

        inner_train = train_data.filter(pl.col("date") < fold.inner_valid_start).sort(["race_id", "horse_id"])
        inner_valid = train_data.filter(pl.col("date") >= fold.inner_valid_start).sort(["race_id", "horse_id"])

        # --- D-084: inner 検証によるラウンド数の選択（本番モデルのみ） ---
        selection_booster, inner_metrics = fit_stage2(
            objective=self.objective,
            x=inner_train.select(feature_cols), label=inner_train["label"], group=race_group(inner_train),
            sample_weight=inner_train["sample_weight"], seed=fold.seed, params=params,
            inner_x=inner_valid.select(feature_cols), inner_label=inner_valid["label"],
            inner_group=race_group(inner_valid),
            num_boost_round=self.num_boost_round, early_stopping_rounds=self.early_stopping_rounds,
        )
        self.fold_inner_metrics[fold.index] = inner_metrics
        n_rounds = max(1, selection_booster.best_iteration)

        # --- 本番モデル: 学習期間全体で学習、選ばれたラウンド数を固定
        #     （早期終了そのものを使わない。理由は fit_stage2_fixed_rounds() の docstring） ---
        final_booster = fit_stage2_fixed_rounds(
            objective=self.objective,
            x=train_data.select(feature_cols), label=train_data["label"], group=race_group(train_data),
            sample_weight=train_data["sample_weight"], seed=fold.seed, params=params, num_boost_round=n_rounds,
        )

        # --- 校正データ作成（015 1節 / D-098）: 同じ4分割で Stage 2 自身をクロスフィット ---
        blocks = cross_fit_blocks(fold, n_blocks=self.n_blocks)
        # dated（train_ids + valid_ids ぶんを既に取得済み）から絞る。再クエリしない
        train_dated = dated.filter(pl.col("race_id").is_in(train_ids))
        oof_scores_parts: list[pl.DataFrame] = []
        for b_start, b_end in blocks:
            block_ids = set(_ids_in_range(train_dated, b_start, b_end))
            if not block_ids:
                continue
            block_data = train_data.filter(pl.col("race_id").is_in(block_ids)).sort(["race_id", "horse_id"])
            rest_data = train_data.filter(~pl.col("race_id").is_in(block_ids)).sort(["race_id", "horse_id"])
            if block_data.is_empty() or rest_data.is_empty():
                continue
            block_model = fit_stage2_fixed_rounds(
                objective=self.objective,
                x=rest_data.select(feature_cols), label=rest_data["label"], group=race_group(rest_data),
                sample_weight=rest_data["sample_weight"], seed=fold.seed, params=params, num_boost_round=n_rounds,
            )
            pred = predict_win_prob(
                block_model, block_data.select(feature_cols), block_data["race_id"], block_data["horse_id"],
            )
            oof_scores_parts.append(pred.select(["race_id", "horse_id", "score"]))

        oof_scores = pl.concat(oof_scores_parts) if oof_scores_parts else pl.DataFrame(
            schema={"race_id": pl.Int64, "horse_id": pl.Int64, "score": pl.Float64}
        )
        g1_ids = set(train_dated.filter(pl.col("grade") == "G1")["race_id"].to_list())
        is_winner = conn.execute(
            "SELECT race_id, horse_id, (finish_pos = 1) AS is_winner FROM runners "
            "WHERE race_id = ANY(?) AND status IN ('出走', '降着', '競走中止', '失格')",
            [list(g1_ids)] if g1_ids else [[]],
        ).pl()
        oof_g1 = (
            oof_scores.filter(pl.col("race_id").is_in(g1_ids))
            .join(is_winner, on=["race_id", "horse_id"], how="inner")
        )
        self.fold_calibrators[fold.index] = fit_calibrator(oof_g1)
        if not oof_g1.is_empty():
            n_starters_g1 = conn.execute(
                "SELECT race_id, n_starters FROM races WHERE race_id = ANY(?)",
                [oof_g1["race_id"].unique().to_list()],
            ).pl()
            self.fold_oof_g1[fold.index] = oof_g1.join(n_starters_g1, on="race_id", how="left")

        # --- 検証期間の予測（校正前。y_pred は生スコアの softmax） ---
        valid_pred = predict_win_prob(
            final_booster, valid_data.select(feature_cols), valid_data["race_id"], valid_data["horse_id"],
        )
        # apply_g1_calibration() が校正前の生スコアを引き直せるように保持する
        # （win_prob から score を逆算することはできない。同モジュールの
        # apply_g1_calibration() docstring を参照）
        self.fold_valid_scores[fold.index] = valid_pred.select(["race_id", "horse_id", "score"])

        out = valid_pred.select(["race_id", "horse_id", "win_prob"]).join(
            valid_data.select(["race_id", "horse_id", "label"]), on=["race_id", "horse_id"],
        )
        # D-074: 1着同着は正解ラベルを同着頭数で等分する（005/015 と同じ扱い）
        n_winners = (
            out.filter(pl.col("label") == 3).group_by("race_id").agg(pl.len().alias("n_winners"))
        )
        out = out.join(n_winners, on="race_id", how="left")
        return out.select(
            "race_id", "horse_id",
            pl.when(pl.col("label") == 3).then(1.0 / pl.col("n_winners")).otherwise(0.0).alias("y_true"),
            pl.col("win_prob").alias("y_pred"),
        )


def apply_g1_calibration(
    conn: duckdb.DuckDBPyConnection, walk_forward_out: pl.DataFrame, runner: Stage2FoldRunner,
) -> pl.DataFrame:
    """`walk_forward_out`（`run_walk_forward()` の生スコア出力）のうち G1 の行だけ、
    fold ごとの `Calibrator` で校正し直す（`D-099`）。

    `all` 母集団の指標は `walk_forward_out` をそのまま使う（校正しない）。

    **`y_pred` に `Calibrator.apply()` を直接使わない。** `y_pred` は
    `predict_fold()` が返す時点で既にレース内 softmax 済みの `win_prob`
    （`T=1`）であり、`Calibrator.apply()`（内部で `softmax(score/T)` を
    計算する）にそのまま渡すと **二重に softmax を取ってしまい**、
    `fit_calibrator()` が校正前提としていた生スコアとは無関係な値になる
    （softmax は加法シフトに対して不変だが `T≠1` の除算と組み合わさると
    保存されない）。そのため `runner.fold_valid_scores`（`predict_fold()`
    が保持する校正前の生スコア）を `(race_id, horse_id)` で結合し直し、
    その生スコアに対して校正を適用する。

    `runner.fold_calibrators` に無い `fold_index` の行（`walk_forward_out`
    にだけ存在する場合）は校正できないため**その fold の行を返さない**
    （黙って未校正のまま混ぜない）。`run_p3_completion_check()` のように
    同じ `runner` を `run_walk_forward()` に渡した直後に呼ぶ使い方では
    起こらない。
    """
    if walk_forward_out.is_empty():
        return walk_forward_out

    race_ids = walk_forward_out["race_id"].unique().to_list()
    grades = conn.execute(
        "SELECT race_id, grade FROM races WHERE race_id = ANY(?)", [race_ids]
    ).pl()
    g1_ids = set(grades.filter(pl.col("grade") == "G1")["race_id"].to_list())

    parts: list[pl.DataFrame] = []
    for fold_index, cal in runner.fold_calibrators.items():
        sub = walk_forward_out.filter(
            (pl.col("fold_index") == fold_index) & pl.col("race_id").is_in(g1_ids)
        )
        raw_scores = runner.fold_valid_scores.get(fold_index)
        if sub.is_empty() or raw_scores is None:
            continue
        to_calibrate = sub.select(["race_id", "horse_id"]).join(
            raw_scores, on=["race_id", "horse_id"], how="inner"
        )
        if to_calibrate.is_empty():
            continue
        scored = cal.apply(to_calibrate)
        parts.append(
            scored.select("race_id", "horse_id", pl.col("win_prob").alias("y_pred"))
            .join(sub.select(["race_id", "horse_id", "fold_index", "y_true"]), on=["race_id", "horse_id"])
        )

    if not parts:
        return pl.DataFrame(schema=walk_forward_out.schema)
    return pl.concat(parts).select(walk_forward_out.columns)


# ---------------------------------------------------------------------------
# P-3 完了確認（R-023）の最小実装
# ---------------------------------------------------------------------------
#
# `010-backtest.md`（P-4、未作成）が指標計算を正式に引き取るまでのつなぎ。
# `R-023`（A判定: 開発用検証のG1でLogLossが市場確率を下回るか）だけに
# 答える最小限の実装で、`010` が持つはずの他の指標（Brier・回収率など）
# は扱わない。

def _race_logloss_from_walk_forward(df: pl.DataFrame) -> float:
    """`race_id, horse_id, y_true, y_pred` から `005-baseline.md` と同じ
    定義（レース単位平均）で LogLoss を計算する。`y_true` は
    `Stage2FoldRunner.predict_fold()` が既に `D-074`（1着同着の等分）を
    適用した値であること。
    """
    if df.is_empty():
        return float("nan")
    n_races = df["race_id"].n_unique()
    eps = 1e-15
    total = df.select(
        (pl.col("y_true") * pl.col("y_pred").clip(eps, 1.0).log()).sum()
    ).item()
    return -total / n_races


@dataclass(frozen=True)
class P3CompletionResult:
    """`R-023`（A判定）の判定結果。

    **`g1_n_races == 0` のとき `passes_r023` は `False` になるが、
    「不合格」ではなく「判定不能」を意味する**（`g1_model_logloss` /
    `g1_market_logloss` は `NaN` で `NaN < x` は常に `False` になるため）。
    `passes_r023` を読む前に必ず `g1_n_races > 0` を確認すること。
    """

    g1_model_logloss: float
    g1_market_logloss: float
    g1_n_races: int
    all_model_logloss: float
    all_market_logloss: float
    all_n_races: int
    passes_r023: bool  # g1_model_logloss < g1_market_logloss（g1_n_races==0 なら判定不能）


def run_p3_completion_check(
    conn: duckdb.DuckDBPyConnection, *, today: date, sealed_years: int = DEFAULT_SEALED_YEARS,
    runner: Stage2FoldRunner | None = None,
) -> P3CompletionResult:
    """`R-023`（A判定）を判定する。

    `g1` 母集団は校正後（`D-099`）、`all` 母集団は校正前（生スコア）の
    `y_pred` を使う。市場確率は `baseline.probability_metrics()` を
    **同じレース集合に対して計算し直す**（`D-075`）。

    封印セット（`D-017`）は2箇所で効く: `run_walk_forward()`
    （`make_folds()` 経由）が fold の**境界年**を封印除外後の母集団から
    決め、`Stage2FoldRunner`（`_race_ids_in_range()`）が fold の日付範囲
    から実際の `race_id` を確定する際に封印G1を除外する。**両者の
    `sealed_years` は必ず一致させること**（本関数は `runner` を渡さない
    限り自動で揃える）。`runner` を明示的に渡す場合は、その
    `runner.sealed_years` が本関数の `sealed_years` と一致しているか
    呼び出し側の責任で確認すること。
    """
    from umagic.baseline import probability_metrics

    runner = runner or Stage2FoldRunner(today=today, sealed_years=sealed_years)
    raw = run_walk_forward(
        conn, predict_fold=runner.predict_fold, today=today, sealed_years=sealed_years,
    )

    g1_calibrated = apply_g1_calibration(conn, raw, runner)
    all_model_logloss = _race_logloss_from_walk_forward(raw)
    g1_model_logloss = _race_logloss_from_walk_forward(g1_calibrated)

    all_race_ids = raw["race_id"].unique().to_list()
    g1_race_ids = g1_calibrated["race_id"].unique().to_list()

    all_market = probability_metrics(conn, all_race_ids, population="all")
    g1_market = probability_metrics(conn, g1_race_ids, population="g1")

    return P3CompletionResult(
        g1_model_logloss=g1_model_logloss,
        g1_market_logloss=g1_market.log_loss,
        g1_n_races=len(g1_race_ids),
        all_model_logloss=all_model_logloss,
        all_market_logloss=all_market.log_loss,
        all_n_races=len(all_race_ids),
        passes_r023=g1_model_logloss < g1_market.log_loss,
    )
