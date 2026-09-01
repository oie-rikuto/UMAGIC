"""運用推論用モデルの学習・キャッシュ（`D-183`、9/6実運用に向けた高速化）。

`predict_race`（`D-181`/`D-182`）は呼ぶたびに全履歴（11年超・46万行超）の
特徴量を`build_features()`の履歴自己結合で毎回計算し直しており、実測で
数時間かかった（`D-183`）。ほとんどの計算は**過去の確定済みレースの
特徴量**で、対象レース（新規1件）を追加しても変わらない。

この重い計算を「本番DB更新時に1回だけ」（`build_production_cache()`）に
切り出し、個々の予測（`predict_with_cache()`）は**対象レース1件だけ**
特徴量計算してキャッシュ済みモデルで予測する。`build_features()`の
内部実装は変えていない——`race_ids`を1件に絞って呼ぶだけで、コストが
下がることを利用する（各特徴量は対象行ごとに独立した as-of 集計で、
同じバッチに他の行が何件あるかに依存しない設計のため、`D-051`等）。

**本番DBを更新したら`build_production_cache()`を再実行してキャッシュを
作り直すこと。** 古いキャッシュのままだと、対象レース直前の最新情報が
学習に反映されない。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

from umagic.orchestration import (
    CATEGORY_COLUMNS,
    Stage2FoldRunner,
    _feature_columns,
    assemble_stage2_matrix,
    stage1_fit_all,
)
from umagic.stage1 import LightGBMStage1Model, predict_f102
from umagic.stage2 import (
    CategoryMapping,
    apply_category_mappings,
    build_category_mappings,
    build_labels,
    fit_stage2,
    fit_stage2_fixed_rounds,
    predict_win_prob,
    race_group,
)
from umagic.training import Fold, load_model, sample_weights, save_model, verify_feature_order

STAGE1_SUBDIR = "stage1"
STAGE2_SUBDIR = "stage2"
CATEGORY_MAP_FILENAME = "category_mappings.json"
CACHE_META_FILENAME = "cache_meta.json"


def _save_category_mappings(mappings: dict[str, CategoryMapping], path: Path) -> None:
    payload = {
        col: {"column": m.column, "keep": sorted(m.keep), "other_code": m.other_code}
        for col, m in mappings.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_category_mappings(path: Path) -> dict[str, CategoryMapping]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        col: CategoryMapping(column=v["column"], keep=frozenset(v["keep"]), other_code=v["other_code"])
        for col, v in payload.items()
    }


def build_production_cache(
    db_path: str, cache_dir: Path, *, runner: Stage2FoldRunner | None = None,
) -> dict:
    """`db_path` を読み取り専用で開き、`build_production_cache_from_conn()`
    を呼ぶ薄いラッパー（`scripts/build_prediction_cache.py` 用）。"""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        meta = build_production_cache_from_conn(conn, cache_dir, runner=runner)
    finally:
        conn.close()
    meta["db_path"] = str(Path(db_path).resolve())
    (cache_dir / CACHE_META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return meta


def build_production_cache_from_conn(
    conn: duckdb.DuckDBPyConnection, cache_dir: Path, *, runner: Stage2FoldRunner | None = None,
) -> dict:
    """本番DB全履歴でStage1・Stage2を学習し、`cache_dir` に保存する。

    `D-017`の封印は学習データには適用しない（「学習データ（全レース）は
    封印対象ではない」）——ここで作るのは運用予測用の最終モデルであり、
    G1のLogLossを測って選定するための`Stage2FoldRunner.predict_fold()`の
    fold構造とは目的が違う。

    `conn`を直接受け取る（ファイルパスではなく）ことで、テストが
    ファイルI/O無しに合成DBで検証できるようにしている
    （`build_production_cache()` がファイル版のラッパー）。
    """
    runner = runner or Stage2FoldRunner(today=date.today(), sealed_years=0)

    train_ids = conn.execute("SELECT race_id FROM races ORDER BY race_id").pl()["race_id"].to_list()
    if not train_ids:
        raise ValueError(f"{db_path} にレースが1件もありません")
    min_date, max_date = conn.execute("SELECT MIN(date), MAX(date) FROM races").fetchone()

    # `valid_start`/`valid_end` はダミー（実際の対象レースは推論時に別途
    # 与える）。`fold.inner_valid_start`（学習期間末尾1年）の計算にだけ使う
    fold = Fold(index=0, train_start=min_date, train_end=max_date,
                valid_start=max_date + timedelta(days=1), valid_end=max_date + timedelta(days=1),
                seed=42)

    f102_train_oof, stage1_full = stage1_fit_all(conn, fold, train_ids, n_blocks=runner.n_blocks)
    x_all = assemble_stage2_matrix(
        conn, train_ids, as_of=fold.valid_start, f102=f102_train_oof.select(["race_id", "f102"]),
        horse_effects=None,
    )
    labels_all = build_labels(conn, train_ids)
    dated = conn.execute(
        "SELECT race_id, date FROM races WHERE race_id = ANY(?)", [train_ids],
    ).pl()

    data = (
        x_all.join(labels_all, on=["race_id", "horse_id"], how="inner")
        .join(dated, on="race_id", how="left")
        .sort(["race_id", "horse_id"])
    )

    mappings = build_category_mappings(data, min_count=runner.min_category_count, columns=CATEGORY_COLUMNS)
    data = apply_category_mappings(data, mappings)

    sw = sample_weights(conn, train_ids, class_weights=runner.class_weights)
    data = data.join(sw, on="race_id", how="left").sort(["race_id", "horse_id"])

    feature_cols = _feature_columns(data)
    cat_idx = [feature_cols.index(c) for c in CATEGORY_COLUMNS if c in feature_cols]
    params = {**runner.extra_lgb_params, "categorical_feature": cat_idx}

    inner_train = data.filter(pl.col("date") < fold.inner_valid_start).sort(["race_id", "horse_id"])
    inner_valid = data.filter(pl.col("date") >= fold.inner_valid_start).sort(["race_id", "horse_id"])

    selection_booster, inner_metrics = fit_stage2(
        objective=runner.objective,
        x=inner_train.select(feature_cols), label=inner_train["label"], group=race_group(inner_train),
        sample_weight=inner_train["sample_weight"], seed=fold.seed, params=params,
        inner_x=inner_valid.select(feature_cols), inner_label=inner_valid["label"],
        inner_group=race_group(inner_valid),
        num_boost_round=runner.num_boost_round, early_stopping_rounds=runner.early_stopping_rounds,
    )
    n_rounds = max(1, selection_booster.best_iteration)

    final_booster = fit_stage2_fixed_rounds(
        objective=runner.objective,
        x=data.select(feature_cols), label=data["label"], group=race_group(data),
        sample_weight=data["sample_weight"], seed=fold.seed, params=params, num_boost_round=n_rounds,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    stage1_full.save(cache_dir / STAGE1_SUBDIR, {"trained_through": str(max_date)})
    save_model(
        final_booster,
        {"feature_names": feature_cols, "trained_through": str(max_date),
         "n_rounds": n_rounds, "fold_seed": fold.seed, "objective": runner.objective,
         "inner_metrics": inner_metrics},
        cache_dir / STAGE2_SUBDIR,
    )
    _save_category_mappings(mappings, cache_dir / CATEGORY_MAP_FILENAME)

    meta = {
        "trained_through": str(max_date),
        "n_train_races": len(train_ids), "n_rounds": n_rounds,
        "built_at": date.today().isoformat(),
    }
    (cache_dir / CACHE_META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return meta


def predict_with_cache(
    conn: duckdb.DuckDBPyConnection, race_id: int, target_date: date, cache_dir: Path,
) -> pl.DataFrame:
    """`build_production_cache()` が作ったキャッシュで、対象レース1件
    （`conn` の `races`/`runners` に既に重ね合わせ済みであること——
    `inference.build_overlay()`）だけ特徴量計算して予測する。

    戻り値: `race_id, horse_id, win_prob`。
    """
    stage1_model, _ = LightGBMStage1Model.load(cache_dir / STAGE1_SUBDIR)
    booster, meta = load_model(cache_dir / STAGE2_SUBDIR)
    mappings = _load_category_mappings(cache_dir / CATEGORY_MAP_FILENAME)
    feature_cols = meta["feature_names"]

    f102 = predict_f102(stage1_model, conn, [race_id], as_of=target_date)
    x = assemble_stage2_matrix(conn, [race_id], as_of=target_date, f102=f102, horse_effects=None)
    x = apply_category_mappings(x, mappings)

    verify_feature_order(meta, _feature_columns(x))
    pred = predict_win_prob(booster, x.select(feature_cols), x["race_id"], x["horse_id"])
    return pred.select(["race_id", "horse_id", "win_prob"])


# ---------------------------------------------------------------------------
# 予測の内訳（`D-192`）
# ---------------------------------------------------------------------------
#
# 特徴量名 → `F-xxx`（`docs/domain-knowledge.md`）の対応。`assemble_stage2_matrix`
# が作る169列は、1つの `F-xxx` が複数列（生値・`_z`・`_rank`・`_unavailable`）に
# 展開されたものなので、内訳を人間に見せるときは `F-xxx` 単位に畳む。
#
# 接頭辞で機械的に決まらないものだけを明示する（`last3f_*`→`F-303`、
# `fspd_*`→`F-304`、生ID4列→`F-201`）。残りは `fNNN` の数字がそのまま
# `F-NNN` を指す。
_EXPLICIT_FEATURE_FAMILY = {
    "sire_id": "F-201", "damsire_id": "F-201",
    "jockey_id": "F-201", "trainer_id": "F-201",
}

FEATURE_FAMILY_LABELS = {
    "F-101": "逃げ意欲スコア", "F-102": "想定ペース指標（Stage 1の出力）",
    "F-103": "ペース適性", "F-104": "展開交互作用",
    "F-201": "血統・騎手・厩舎のID（生カテゴリ）", "F-202": "種牡馬の条件別成績",
    "F-302": "補正タイムベースの能力値（既定無効・D-110）",
    "F-303": "上がり3F関連", "F-304": "中央値ベースの速度指数",
    "F-501": "当日の脚質バイアス", "F-503": "開催週次・柵移動",
    "F-601": "反動リスク（前走の着差・上がり順位）", "F-602": "ローテーション",
    "F-603": "馬体重", "F-701": "騎手の実力（人気帯で交絡除去）",
    "F-702": "乗り替わり", "F-703": "厩舎の勝負度", "F-704": "騎手のコース適性",
    "F-801": "枠順バイアス", "F-802": "コース・距離適性",
    "F-803": "レース基礎情報", "F-804": "当日の天候・馬場状態",
    "F-805": "出走馬の基礎情報（年齢・性別・斤量）", "F-806": "相手強度",
    "F-807": "前走からの条件替わり", "F-809": "馬のキャリア成績率",
}


def feature_family(column: str) -> str:
    """特徴量列名を `F-xxx` に対応づける。未知の形式は `"その他"` を返す。"""
    if column in _EXPLICIT_FEATURE_FAMILY:
        return _EXPLICIT_FEATURE_FAMILY[column]
    if column.startswith("last3f_"):
        return "F-303"
    if column.startswith("fspd_"):
        return "F-304"
    if len(column) >= 4 and column[0] == "f" and column[1:4].isdigit():
        return f"F-{column[1:4]}"
    return "その他"


def explain_with_cache(
    conn: duckdb.DuckDBPyConnection, race_id: int, target_date: date, cache_dir: Path,
    *, top_k: int = 6,
) -> pl.DataFrame:
    """`predict_with_cache()` と同じ入力で、各馬のスコアの内訳を返す。

    LightGBM の `pred_contrib=True`（SHAP値）を使い、169列の寄与を
    `F-xxx` 単位に合計してから、絶対値の大きい順に `top_k` 件を返す。
    寄与はスコア（softmax前の生margin）のスケールで、**レース内の相対
    比較にのみ意味がある**（`predict_win_prob` が `_softmax_by_race` で
    レース単位に正規化するため、絶対値そのものは確率に直接対応しない）。

    戻り値: `race_id, horse_id, family, label, contribution`（`contribution`
    降順ではなく、馬ごとに絶対値降順で `top_k` 件）。
    """
    stage1_model, _ = LightGBMStage1Model.load(cache_dir / STAGE1_SUBDIR)
    booster, meta = load_model(cache_dir / STAGE2_SUBDIR)
    mappings = _load_category_mappings(cache_dir / CATEGORY_MAP_FILENAME)
    feature_cols = meta["feature_names"]

    f102 = predict_f102(stage1_model, conn, [race_id], as_of=target_date)
    x = assemble_stage2_matrix(conn, [race_id], as_of=target_date, f102=f102, horse_effects=None)
    x = apply_category_mappings(x, mappings)
    verify_feature_order(meta, _feature_columns(x))

    # (n_rows, n_features + 1)。末尾はベース値（全体平均）で、内訳からは外す
    contrib = booster.predict(x.select(feature_cols).to_numpy(), pred_contrib=True)

    rows = []
    for i, horse_id in enumerate(x["horse_id"].to_list()):
        by_family: dict[str, float] = {}
        for j, col in enumerate(feature_cols):
            fam = feature_family(col)
            by_family[fam] = by_family.get(fam, 0.0) + float(contrib[i][j])
        for fam, value in sorted(by_family.items(), key=lambda kv: -abs(kv[1]))[:top_k]:
            rows.append({
                "race_id": race_id, "horse_id": horse_id, "family": fam,
                "label": FEATURE_FAMILY_LABELS.get(fam, fam), "contribution": value,
            })
    return pl.DataFrame(rows)
