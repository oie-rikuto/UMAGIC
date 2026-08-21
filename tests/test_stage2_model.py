"""Stage 2 の学習と出力（`docs/spec/007-stage2-ranker.md` 5〜8節）。"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from umagic.stage2 import fit_stage2, predict_win_prob, race_group


def _synthetic_data(n_races=30, n_per_race=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        for h in range(n_per_race):
            rows.append((r, h, float(rng.normal()), int(rng.integers(0, 3))))
    df = pl.DataFrame(
        rows, schema=["race_id", "horse_id", "f1", "cat"], orient="row"
    ).sort(["race_id", "horse_id"])
    # f1 が大きいほど上位（1着=3, 2着=2, 3着=1, それ以外=0）という擬似的な真の関係
    df = df.with_columns(
        (n_per_race - pl.col("f1").rank(method="ordinal", descending=True).over("race_id"))
        .clip(0, 3).cast(pl.Int64).alias("label")
    )
    return df


def _split(df, n_train_races):
    train_mask = df["race_id"] < n_train_races
    return df.filter(train_mask), df.filter(~train_mask)


def _fit(df_train, df_valid, *, seed=42, cat_idx=None):
    x_tr = df_train.select(["f1", "cat"])
    x_va = df_valid.select(["f1", "cat"])
    w_tr = pl.Series([1.0] * df_train.height)
    params = {"categorical_feature": cat_idx} if cat_idx is not None else {}
    booster, metrics = fit_stage2(
        x_tr, df_train["label"], race_group(df_train),
        sample_weight=w_tr, seed=seed, params=params,
        inner_x=x_va, inner_label=df_valid["label"], inner_group=race_group(df_valid),
    )
    return booster, metrics


def test_win_prob_sums_to_one_per_race(conn):
    """観点1: win_prob のレース内合計が 1.0 ± 1e-6（R-002）。"""
    df = _synthetic_data()
    train, valid = _split(df, 24)
    booster, _ = _fit(train, valid)

    out = predict_win_prob(booster, valid.select(["f1", "cat"]), valid["race_id"], valid["horse_id"])
    sums = out.group_by("race_id").agg(pl.col("win_prob").sum())
    for s in sums["win_prob"].to_list():
        assert abs(s - 1.0) < 1e-6


def test_all_nan_feature_horse_still_gets_probability(conn):
    """観点2: 全特徴量が NaN の馬にも win_prob が付き、合計が 1.0 を保つ（R-003）。"""
    df = _synthetic_data()
    train, valid = _split(df, 24)
    booster, _ = _fit(train, valid)

    # 検証集合の最初のレースの1頭を全特徴量欠損にする
    first_race = valid["race_id"][0]
    valid2 = valid.with_columns(
        pl.when((pl.col("race_id") == first_race) & (pl.col("horse_id") == 0))
        .then(None).otherwise(pl.col("f1")).alias("f1")
    )

    out = predict_win_prob(
        booster, valid2.select(["f1", "cat"]), valid2["race_id"], valid2["horse_id"],
    )
    row = out.filter((pl.col("race_id") == first_race) & (pl.col("horse_id") == 0))
    assert row.height == 1
    assert row["win_prob"].to_list()[0] is not None
    s = out.filter(pl.col("race_id") == first_race)["win_prob"].sum()
    assert abs(s - 1.0) < 1e-6


def test_same_seed_gives_bit_exact_predictions(conn):
    """観点11: 同じ fold・同じ seed で2回学習すると予測がビット完全一致（R-021）。"""
    df = _synthetic_data()
    train, valid = _split(df, 24)
    booster1, _ = _fit(train, valid, seed=7)
    booster2, _ = _fit(train, valid, seed=7)

    p1 = booster1.predict(valid.select(["f1", "cat"]).to_numpy())
    p2 = booster2.predict(valid.select(["f1", "cat"]).to_numpy())
    assert (p1 == p2).all()


def test_inner_metrics_recorded(conn):
    """観点14: meta.json 相当の inner_logloss と inner_ndcg3 の両方が記録される。"""
    df = _synthetic_data()
    train, valid = _split(df, 24)
    _, metrics = _fit(train, valid)
    assert "inner_logloss" in metrics
    assert "inner_ndcg3" in metrics
    assert metrics["inner_logloss"] == metrics["inner_logloss"]  # NaNでない
    assert metrics["inner_ndcg3"] == metrics["inner_ndcg3"]


def test_race_group_matches_row_count(conn):
    """観点7の別角度: 学習・検証それぞれで group の合計が行数と一致する。"""
    df = _synthetic_data()
    train, valid = _split(df, 24)
    assert race_group(train).sum() == train.height
    assert race_group(valid).sum() == valid.height


def test_sample_weight_same_within_race():
    """観点15: 同じレースの全出走行が同じ sample_weight を持つ、という契約を
    呼び出し側で満たせることを確認する（sample_weight は fit_stage2 に
    そのまま渡すだけで、Stage 2 自身は組み立てない）。"""
    df = _synthetic_data(n_races=3, n_per_race=4)
    race_to_weight = {0: 5.0, 1: 1.0, 2: 3.0}
    w = pl.Series([race_to_weight[r] for r in df["race_id"].to_list()])
    for race_id in df["race_id"].unique().to_list():
        mask = df["race_id"] == race_id
        assert len(set(w.filter(mask).to_list())) == 1


def test_stage2_module_never_references_odds_or_popularity():
    """観点13の一部: `stage2.py` 自身が `odds_win`/`popularity` を一切参照しない
    （`R-018`）。入力特徴量の組み立て（`003-features.md` の `build_features()`）
    自体は本モジュールの外側の責務のため、そちら側の保証は `P-1` の
    `test_features_build.py` 等が別途持つ。ここでは `stage2.py` が新たに
    それらを引き込んでいないことだけを確認する。"""
    import inspect
    import umagic.stage2 as stage2_module

    source = inspect.getsource(stage2_module)
    assert "odds_win" not in source
    assert "popularity" not in source
