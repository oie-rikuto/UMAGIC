"""`P-3` Stage 2 適性照合ランカー（`docs/spec/007-stage2-ranker.md`）。

`003-features.md` の全特徴量と Stage 1 の出力（`F-102`）を入力に、
出走各馬のスコアを出し、レース内 softmax で未校正の勝率に変換する。
`D-007` の2段階アーキテクチャの後段。確率の校正は行わない
（`015-calibration.md` の責務、`D-095`）。

**入力特徴量の組み立て**（`003-features.md` の `build_features()` の
全列 + `F-102` の結合）は本モジュールの外側（学習の orchestration 層）
の責務とする。`007-stage2-ranker.md` の型シグネチャ一覧自体、そのための
関数を持たない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import duckdb
import numpy as np
import polars as pl

_CATEGORY_COLS = ["sire_id", "damsire_id", "jockey_id", "trainer_id"]


@dataclass(frozen=True)
class CategoryMapping:
    """低頻度カテゴリの丸め表（`D-092`）。fold の学習期間から作る。"""

    column: str
    keep: frozenset  # 元の値の集合。この集合の要素だけ独自コードを持つ
    other_code: int  # 「その他」に割り当てる整数コード


def build_category_mappings(
    train: pl.DataFrame, *, min_count: int,
) -> dict[str, CategoryMapping]:
    """fold の学習期間だけで出現回数を数える（`R-019` / 原則7）。"""
    mappings: dict[str, CategoryMapping] = {}
    for col in _CATEGORY_COLS:
        if col not in train.columns:
            continue
        counts = (
            train.filter(pl.col(col).is_not_null())
            .group_by(col)
            .len()
        )
        keep = sorted(counts.filter(pl.col("len") >= min_count)[col].to_list())
        mappings[col] = CategoryMapping(column=col, keep=frozenset(keep), other_code=len(keep))
    return mappings


def apply_category_mappings(
    df: pl.DataFrame, mappings: dict[str, CategoryMapping],
) -> pl.DataFrame:
    """カテゴリ列を整数コードに変換する。

    **`null` はそのまま `null` として残す**（丸め対象にしない）。`D-058` の
    「補完しない」に沿い、LightGBM のネイティブな欠損分岐に委ねる。
    対応表に無い非 `null` の値（`min_count` 未満、または検証期間・推論時
    にのみ現れる未知の値）は `other_code` に落ちる。例外にしない。
    """
    out = df
    for col, mapping in mappings.items():
        if col not in out.columns:
            continue
        code_map = {v: i for i, v in enumerate(sorted(mapping.keep))}
        other = mapping.other_code
        out = out.with_columns(
            pl.col(col)
            .map_elements(
                lambda v, m=code_map, o=other: (None if v is None else m.get(v, o)),
                return_dtype=pl.Int32,
            )
            .alias(col)
        )
    return out


_LABELS_SQL = """
SELECT race_id, horse_id,
    CASE
        WHEN finish_pos = 1 THEN 3
        WHEN finish_pos = 2 THEN 2
        WHEN finish_pos = 3 THEN 1
        ELSE 0
    END AS label
FROM runners
WHERE race_id = ANY(?) AND status IN ('出走', '降着', '競走中止', '失格')
"""

_LABELS_SCHEMA = {"race_id": pl.Int64, "horse_id": pl.Int64, "label": pl.Int64}


def build_labels(conn: duckdb.DuckDBPyConnection, race_ids: list[int]) -> pl.DataFrame:
    """`lambdarank` のラベルを作る（`D-093` / `D-094`）。

    `出走取消` / `競走除外` は行ごと現れない。`競走中止` / `失格`
    （`finish_pos IS NULL`）はラベル `0` で含める。
    """
    if not race_ids:
        return pl.DataFrame(schema=_LABELS_SCHEMA)
    return conn.execute(_LABELS_SQL, [race_ids]).pl()


def race_group(df: pl.DataFrame) -> pl.Series:
    """`(race_id, horse_id)` で整列した `df` から LightGBM の `group` を作る（3節）。

    `df` は呼び出し側で `sort(["race_id", "horse_id"])` 済みであること
    （`D-055` の決定的順序）。
    """
    return df.group_by("race_id", maintain_order=True).len()["len"]


# ---------------------------------------------------------------------------
# レース内 softmax（D-095。温度は導入しない）とレース単位の評価指標
# ---------------------------------------------------------------------------

def _softmax_by_race(df: pl.DataFrame, *, score_col: str = "score") -> pl.DataFrame:
    out = df.with_columns(
        (pl.col(score_col) - pl.col(score_col).max().over("race_id")).alias("_shifted")
    )
    out = out.with_columns(pl.col("_shifted").exp().alias("_exp"))
    out = out.with_columns(
        (pl.col("_exp") / pl.col("_exp").sum().over("race_id")).alias("win_prob")
    )
    return out.drop(["_shifted", "_exp"])


def _group_slices(group: Iterable[int]) -> list[slice]:
    slices = []
    start = 0
    for g in group:
        slices.append(slice(start, start + g))
        start += g
    return slices


def race_metrics(
    scores: np.ndarray, group: list[int], label: np.ndarray, *, k: int = 3,
) -> tuple[float, float]:
    """レース単位で `(LogLoss, NDCG@k)` を計算し、レース数で平均する（6節）。

    LogLoss は `label == 3`（1着、同着は `D-074` と同じく等分）を正解とした
    レース内 softmax の対数損失。`R-023`（A判定）の判定指標と揃える。
    """
    total_logloss = 0.0
    total_ndcg = 0.0
    n_races = 0
    for sl in _group_slices(group):
        s = scores[sl]
        y_label = label[sl]
        n = len(s)
        if n == 0:
            continue
        m = s.max()
        exp_s = np.exp(s - m)
        p = exp_s / exp_s.sum()

        winners = y_label == 3
        n_winners = winners.sum()
        y = np.where(winners, 1.0 / n_winners, 0.0) if n_winners > 0 else np.zeros(n)
        eps = 1e-15
        total_logloss += -float(np.sum(y * np.log(np.clip(p, eps, 1.0))))

        k_eff = min(k, n)
        order = np.argsort(-s)[:k_eff]
        discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
        dcg = float(np.sum(y_label[order] * discounts))
        ideal_order = np.argsort(-y_label)[:k_eff]
        idcg = float(np.sum(y_label[ideal_order] * discounts))
        total_ndcg += (dcg / idcg) if idcg > 0 else 0.0

        n_races += 1

    if n_races == 0:
        return float("nan"), float("nan")
    return total_logloss / n_races, total_ndcg / n_races


def _feval(preds: np.ndarray, train_data) -> list[tuple[str, float, bool]]:
    group = list(train_data.get_group())
    label = train_data.get_label()
    logloss, ndcg3 = race_metrics(preds, group, label, k=3)
    # first_metric_only=True が最初の要素（race_logloss）で選択するよう先頭に置く
    return [("race_logloss", logloss, False), ("ndcg@3", ndcg3, True)]


def fit_stage2(
    x: pl.DataFrame,
    label: pl.Series,
    group: pl.Series,
    *,
    sample_weight: pl.Series | None,
    seed: int,
    params: dict,
    inner_x: pl.DataFrame,
    inner_label: pl.Series,
    inner_group: pl.Series,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
) -> tuple[object, dict]:
    """`lambdarank` を学習する。戻り値: `(Booster, inner検証の指標)`（6節）。

    early stopping はレース内 softmax 後の LogLoss（`race_logloss`）で行い、
    `NDCG@3` は記録のみ（`R-023` の判定指標と選択指標を一致させる）。
    """
    import lightgbm as lgb

    cat_cols = params.get("categorical_feature")
    extra_params = {k: v for k, v in params.items() if k != "categorical_feature"}

    group_list = list(group)
    inner_group_list = list(inner_group)

    train_ds = lgb.Dataset(
        x.to_numpy(), label=label.to_numpy(), group=group_list,
        weight=sample_weight.to_numpy() if sample_weight is not None else None,
        categorical_feature=cat_cols if cat_cols is not None else "auto",
        free_raw_data=False,
    )
    valid_ds = lgb.Dataset(
        inner_x.to_numpy(), label=inner_label.to_numpy(), group=inner_group_list,
        reference=train_ds, free_raw_data=False,
    )

    full_params = {
        "objective": "lambdarank", "metric": "None", "verbose": -1,
        "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
        "deterministic": True, "force_row_wise": True, "num_threads": 1,
        "min_data_in_leaf": 1,
        **extra_params,
    }

    booster = lgb.train(
        full_params, train_ds, num_boost_round=num_boost_round,
        valid_sets=[valid_ds], valid_names=["inner"],
        feval=_feval,
        callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)],
    )

    inner_scores = booster.predict(inner_x.to_numpy(), num_iteration=booster.best_iteration)
    inner_logloss, inner_ndcg3 = race_metrics(
        inner_scores, inner_group_list, inner_label.to_numpy(), k=3,
    )
    metrics = {"inner_logloss": inner_logloss, "inner_ndcg3": inner_ndcg3}
    return booster, metrics


def predict_win_prob(
    booster, x: pl.DataFrame, race_id: pl.Series, horse_id: pl.Series,
) -> pl.DataFrame:
    """列: `race_id, horse_id, score, win_prob`（`D-095`）。

    **`horse_id` は `007-stage2-ranker.md` の型シグネチャに無いが追加した。**
    出力が `horse_id` を持つと仕様書自身が書いており（本関数のdocstring
    と同じ文）、`x`/`race_id` だけでは `horse_id` を復元できないため。
    """
    scores = booster.predict(x.to_numpy())
    df = pl.DataFrame({
        "race_id": list(race_id), "horse_id": list(horse_id), "score": scores,
    })
    return _softmax_by_race(df).select(["race_id", "horse_id", "score", "win_prob"])
